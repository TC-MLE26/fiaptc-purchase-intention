# syntax=docker/dockerfile:1

# =============================================================================
# Estágio 1: builder — instala as dependências de runtime com Poetry
# =============================================================================
FROM python:3.12-slim AS builder

# Poetry com versão fixa; venv dentro do projeto (/app/.venv) para copiar depois
ENV POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir poetry==2.5.1

WORKDIR /app

# Copia só os manifestos de dependência primeiro (aproveita o cache do Docker)
COPY pyproject.toml poetry.lock ./

# Instala APENAS as dependências de runtime, como resolvidas no poetry.lock.
# --only main ignora o grupo dev (pytest, ruff...); --no-root não instala o
# projeto em si (package-mode = false; a API roda via python main.py)
RUN poetry install --only main --no-root

# =============================================================================
# Estágio 2: runtime — imagem final enxuta, sem Poetry e sem root
# =============================================================================
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="fiaptc-purchase-intention-api" \
      org.opencontainers.image.description="API FastAPI de previsão de intenção de compra (modelo via MLflow Model Registry)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Usuário não-root (sem shell). O --chown nos COPY evita um `chown -R`
# posterior, que duplicaria camadas grandes.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Venv pronto do builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Apenas o código necessário para servir a API
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src/ ./src/

USER appuser

EXPOSE 8075

# Consulta /health com a stdlib (sem curl); usa a PORT configurada
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8075'), timeout=3)"]

CMD ["python", "main.py"]
