# fiaptc-purchase-intention

Tech Challenge da Fase 2 da pós-graduação em ML Engineering (FIAP). O projeto prevê a
propensão de compra de um usuário de e-commerce a partir do comportamento de navegação,
para que o time de negócio consiga priorizar ações (ofertas, remarketing) durante a
sessão, e não só depois dela.

É um problema de classificação binária com dados tabulares, onde cada linha representa
uma sessão de navegação. O dataset é o
[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI), com 17 features e o alvo `Revenue`. A fonte dos dados é a versão do UCI. Este repositório cobre a análise exploratória,
o pipeline de dados e treinamento, o rastreamento de experimentos e a API de inferência,
que consome o modelo promovido no MLflow Model Registry.

---

## Estrutura do Repositório

```
configs/           configurações do pipeline
data/              dados brutos e processados
docs/              documentação e Model Card
models/            artefatos de modelo
notebooks/         EDA e experimentação
src/api/routes/    rotas da API (/health e /predict)
src/core/          configurações, CORS, logging e lifespan
src/inference/     carga do modelo no MLflow e predição
src/schemas/       contratos de entrada e saída
tests/             testes automatizados da API
main.py            ponto de entrada da aplicação
```

---

## Setup

O projeto usa Python 3.12 e [Poetry](https://python-poetry.org/) para gerenciar o
ambiente e as dependências (produção e desenvolvimento separadas):

```bash
poetry install
```

As versões ficam fixadas no `pyproject.toml` e no `poetry.lock`, que são a única fonte
de dependências do projeto. Para instalar só as dependências de produção:
`poetry install --only main`.

**Variáveis de ambiente:**

```bash
cp .env.example .env
```

| Variável               | Padrão                          | Descrição                                   |
| ---------------------- | ------------------------------- | ------------------------------------------- |
| `PROJECT_NAME`         | `fiaptc-purchase-intention`     | Nome do serviço                             |
| `PROJECT_DESCRIPTION`  | descrição da API                | Descrição exibida no OpenAPI                |
| `VERSION`              | `0.1.0`                         | Versão da API                               |
| `DEBUG`                | `false`                         | Ativa logs de debug e reload                |
| `DOCS_ENABLED`         | `true`                          | Expõe `/docs` e `/redoc`                    |
| `HOST`                 | `0.0.0.0`                       | Host do servidor                            |
| `PORT`                 | `8075`                          | Porta do servidor                           |
| `MLFLOW_TRACKING_URI`  | `http://localhost:5000`         | Servidor MLflow                             |
| `MODEL_NAME`           | `purchase-intention-classifier` | Nome do modelo no Model Registry            |
| `MODEL_ALIAS`          | `champion`                      | Alias do modelo promovido                   |
| `PREDICTION_THRESHOLD` | `0.5`                           | Threshold de decisão sobre a probabilidade  |

---

## API

O código fica em `src/api` (rotas), `src/core` (configuração, CORS, logging e lifespan),
`src/inference` (carga do modelo e predição) e `src/schemas` (contratos). Para subir
localmente:

```bash
poetry run python main.py
```

A documentação interativa fica em `/docs` (Swagger) e `/redoc`.

**Endpoints:**

- `GET /health` — status da API e `model_loaded`, que indica se o modelo está carregado.
- `POST /predict` — recebe as 17 features de uma sessão e retorna a probabilidade de
  compra e a decisão pelo threshold.

```bash
curl -X POST http://localhost:8075/predict \
  -H "Content-Type: application/json" \
  -d '{
    "administrative": 9,
    "administrative_duration": 111.5,
    "informational": 1,
    "informational_duration": 48.5,
    "product_related": 49,
    "product_related_duration": 1868.819697,
    "bounce_rates": 0.0,
    "exit_rates": 0.020708874,
    "page_values": 1.706014966,
    "special_day": 0.0,
    "month": "Mar",
    "operating_systems": 2,
    "browser": 2,
    "region": 7,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
    "weekend": false
  }'
```

Resposta (valores ilustrativos):

```json
{
  "purchase_probability": 0.76,
  "purchase_prediction": true,
  "threshold": 0.5,
  "model_name": "purchase-intention-classifier",
  "model_version": "1"
}
```

Os campos do JSON usam snake_case e são mapeados para as colunas originais do dataset
(ex.: `page_values` vira `PageValues`) em `FEATURE_COLUMNS`, em
`src/schemas/predict_schema.py`. Os domínios das categóricas e as faixas numéricas
seguem os valores observados no dataset, incluindo a grafia `June` para o mês de junho.
Payloads inválidos retornam `422`.

**Disponibilidade do modelo:** o modelo é carregado na subida da API. Se o MLflow ou o
modelo não estiverem disponíveis, a API sobe mesmo assim, registra o erro em log e o
`/health` informa `model_loaded: false`. Cada chamada a `/predict` tenta carregar o
modelo de novo, com intervalo mínimo de 10 segundos entre tentativas, e responde `503`
imediatamente enquanto o modelo não estiver disponível. Isso permite subir o MLflow
depois da API.

---

## Modelo no MLflow Model Registry

A API resolve a versão apontada por `{MODEL_NAME}@{MODEL_ALIAS}` (por padrão
`purchase-intention-classifier@champion`) e carrega exatamente essa versão
(`models:/{MODEL_NAME}/{versão}`), de modo que `model_version` na resposta sempre
corresponde ao modelo servido. O modelo é desserializado com pickle, então o
Registry precisa ser um servidor confiável. Premissas que o treinamento precisa
respeitar ao registrar o modelo:

- O modelo é registrado com o flavor `sklearn` (`mlflow.sklearn.log_model`).
- É um `sklearn.Pipeline` completo, com o pré-processamento incluso.
- Recebe um DataFrame com as 17 colunas brutas, com os nomes originais do dataset
  (`Administrative`, ..., `VisitorType`, `Weekend`), sem o alvo `Revenue`.
- Expõe `predict_proba`, e a coluna 1 do resultado é a probabilidade da classe de compra.
- O alias definido em `MODEL_ALIAS` aponta para a versão a ser servida.
- A versão do `scikit-learn` do treino deve ser compatível com a do `poetry.lock`.

---

## Testes

Os testes usam `pytest` e o `TestClient` do FastAPI e não dependem de MLflow rodando nem
de modelo treinado: o modelo é substituído por um estimador falso.

```bash
poetry run pytest
```

## Lint

```bash
poetry run ruff check .
poetry run ruff format --check .
```

---

## Pipeline de Dados e DVC

O ciclo de vida dos dados e artefatos de modelo é orquestrado e versionado via [DVC](https://dvc.org/), garantindo reprodutibilidade ponta a ponta e separação rígida entre código (Git) e dados/binários (DVC).

- O dataset bruto reside em `data/raw/online_shoppers_intention.csv` e é versionado pelo ponteiro `data/raw/online_shoppers_intention.csv.dvc`.
- O arquivo de orquestração `dvc.yaml` define quatro estágios interdependentes:
  1. **`prepare`**: Executa `src.pipeline.data_preparation`, realizando o split estratificado (seed `42`, 80% treino / 20% teste) e salvando `data/processed/train.csv` e `test.csv`.
  2. **`train`**: Executa `src.pipeline.train`, ajustando o pipeline Scikit-Learn com pré-processamento e Random Forest (`class_weight='balanced'`), gerando o artefato `models/model.pkl` e registrando parâmetros e modelo no MLflow.
  3. **`evaluate`**: Executa `src.pipeline.evaluate`, calculando acurácia, precisão, recall, F1-Score e ROC-AUC sobre a partição de teste e gravando em `reports/metrics.json`.
  4. **`register`**: Executa `src.pipeline.register`, promovendo a versão do modelo treinado no MLflow Model Registry sob o alias `champion`.

Comandos do DVC:

```bash
# Executar ou reproduzir o pipeline completo
dvc repro

# Visualizar as métricas obtidas no terminal
dvc metrics show
```

---

## Treinamento e MLflow Tracking

O treinamento utiliza o `scikit-learn` encapsulado em um `sklearn.pipeline.Pipeline` completo (com `StandardScaler` para variáveis contínuas e `OneHotEncoder` para variáveis categóricas).

Para inicializar a interface do MLflow localmente:

```bash
mlflow ui --port 5000
```

Ao executar o pipeline (`dvc repro` ou `python -m src.pipeline.train`), os experimentos são logados no experimento `purchase-intention-challenge`:
- **Parâmetros logados:** `n_estimators`, `max_depth`, `min_samples_split`, `seed`.
- **Métricas avaliadas (`reports/metrics.json`):**
  - **ROC-AUC:** ~0.92
  - **Recall:** ~81.4%
  - **Acurácia:** ~85.8%
  - **F1-Score:** ~0.64
- **Model Registry:** O modelo aprovado recebe a tag `@champion`, permitindo que a API de inferência o carregue dinamicamente na inicialização.

---

## Model Card

| Item | Especificação |
| --- | --- |
| **Nome do Modelo** | `purchase-intention-classifier` |
| **Tipo de Modelo** | Random Forest Classifier com pré-processamento acoplado (`ColumnTransformer`) |
| **Objetivo** | Previsão em tempo real da probabilidade de compra (`Revenue=True`) de uma sessão de e-commerce. |
| **Entradas** | 17 variáveis de sessão (páginas visitadas, duração, taxas de rejeição/saída, PageValues, mês, SO, navegador, região, tipo de tráfego, tipo de visitante e fim de semana). |
| **Saída** | `purchase_probability` (0.0 a 1.0) e `purchase_prediction` (True/False dado o threshold 0.5). |
| **Tratamento de Desbalanceamento** | `class_weight='balanced'` aplicado para priorizar recall da classe minoritária compradora (~15.5% da base). |
| **Limitações** | Desempenho condicionado à integridade dos eventos de navegação; novos valores em variáveis categóricas são ignorados de forma segura (`handle_unknown='ignore'`). |

## Docker

A imagem usa build multi-stage (dependências instaladas com Poetry no estágio de build) e roda como usuário não-root na porta 8075. O `.env` não é copiado para a imagem; as variáveis são passadas em runtime.

```bash
# Build
docker build -t fiaptc-purchase-intention-api .

# Run (variáveis a partir do .env, criado a partir do .env.example)
docker run -d --rm -p 8075:8075 --env-file .env --name pi-api fiaptc-purchase-intention-api

# Verificação
curl http://localhost:8075/health
docker inspect --format '{{.State.Health.Status}}' pi-api

# Parar
docker stop pi-api
```

Dentro do container, `localhost` é o próprio container. Para usar um MLflow rodando no host (Docker Desktop), aponte `MLFLOW_TRACKING_URI` para `host.docker.internal`:

```bash
docker run -d --rm -p 8075:8075 --env-file .env \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  --name pi-api fiaptc-purchase-intention-api
```

Em Linux, adicione `--add-host=host.docker.internal:host-gateway`. Sem MLflow acessível a API sobe, `/health` retorna `model_loaded: false` e `/predict` responde 503.
