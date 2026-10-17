"""Unit tests for the training and data pipeline modules."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.pipeline.config_loader import load_params
from src.pipeline.data_preparation import DataPreparer
from src.pipeline.evaluate import ModelEvaluator
from src.pipeline.model_pipeline import build_model_pipeline
from src.pipeline.register import ModelRegistrar


@pytest.fixture()
def sample_dataset() -> pd.DataFrame:
    """Generate a minimal DataFrame matching dataset structure."""
    return pd.DataFrame(
        {
            "Administrative": [1, 2, 0, 1, 3],
            "Administrative_Duration": [10.0, 20.0, 0.0, 5.0, 30.0],
            "Informational": [0, 1, 0, 0, 1],
            "Informational_Duration": [0.0, 10.0, 0.0, 0.0, 15.0],
            "ProductRelated": [5, 10, 2, 8, 12],
            "ProductRelated_Duration": [100.0, 200.0, 50.0, 150.0, 300.0],
            "BounceRates": [0.01, 0.02, 0.05, 0.0, 0.01],
            "ExitRates": [0.03, 0.04, 0.08, 0.02, 0.03],
            "PageValues": [5.0, 0.0, 0.0, 10.0, 2.0],
            "SpecialDay": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Month": ["Feb", "Mar", "May", "Nov", "Dec"],
            "OperatingSystems": [1, 2, 2, 1, 3],
            "Browser": [1, 2, 1, 2, 1],
            "Region": [1, 1, 2, 3, 1],
            "TrafficType": [1, 2, 2, 1, 2],
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
                "Returning_Visitor",
                "New_Visitor",
                "Returning_Visitor",
            ],
            "Weekend": [False, True, False, False, True],
            "Revenue": [True, False, False, True, False],
        }
    )


def test_load_params() -> None:
    """Test loading params.yaml configuration."""
    params = load_params("configs/params.yaml")
    assert "prepare" in params
    assert "train" in params
    assert "evaluate" in params


def test_data_preparer_split(sample_dataset: pd.DataFrame) -> None:
    """Test DataPreparer split stratification and shapes."""
    preparer = DataPreparer(seed=42, test_size=0.4, target_col="Revenue")
    train_df, test_df = preparer.split_data(sample_dataset)

    assert len(train_df) + len(test_df) == len(sample_dataset)
    assert len(test_df) == 2


def test_model_pipeline_fit_predict(sample_dataset: pd.DataFrame) -> None:
    """Test building, training and predicting with sklearn pipeline."""
    pipeline = build_model_pipeline(seed=42, n_estimators=5, max_depth=3)
    x = sample_dataset.drop(columns=["Revenue"])
    y = sample_dataset["Revenue"]

    pipeline.fit(x, y)
    probs = pipeline.predict_proba(x)

    assert probs.shape == (len(sample_dataset), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_model_evaluator_metrics(sample_dataset: pd.DataFrame) -> None:
    """Test calculation of evaluation metrics."""
    pipeline = build_model_pipeline(seed=42, n_estimators=5, max_depth=3)
    x = sample_dataset.drop(columns=["Revenue"])
    y = sample_dataset["Revenue"]
    pipeline.fit(x, y)

    evaluator = ModelEvaluator(
        tracking_uri="http://localhost:5000", experiment_name="test"
    )
    metrics = evaluator.calculate_metrics(pipeline, x, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_model_registrar_get_latest_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ModelRegistrar finds latest run."""
    mock_client = MagicMock()
    mock_exp = MagicMock(experiment_id="exp_123")
    mock_run = MagicMock(info=MagicMock(run_id="run_456"))

    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [mock_run]

    registrar = ModelRegistrar(tracking_uri="http://localhost:5000")
    registrar.client = mock_client

    run_id = registrar.get_latest_run_id("test-experiment")
    assert run_id == "run_456"
