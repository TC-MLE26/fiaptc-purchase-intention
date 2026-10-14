"""Purchase intention prediction endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_predictor
from src.inference.predictor import PurchasePredictor
from src.schemas.predict_schema import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run model inference",
    description="Recebe as features da sessão e retorna a predição do modelo",
    responses={503: {"description": "Modelo indisponível no MLflow Model Registry"}},
)
def predict(
    payload: PredictRequest,
    predictor: Annotated[PurchasePredictor, Depends(get_predictor)],
) -> PredictResponse:
    """Run purchase intention inference for one session.

    The function is synchronous on purpose: FastAPI runs it in the threadpool,
    so pandas/sklearn do not block the event loop.
    """
    response = predictor.predict(payload)

    logger.info(
        "Prediction made: purchase_probability=%.4f purchase_prediction=%s",
        response.purchase_probability,
        response.purchase_prediction,
    )

    return response
