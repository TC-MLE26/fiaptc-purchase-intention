"""Tests for the run ID handoff between train, evaluate and register stages."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline import evaluate as evaluate_module
from src.pipeline import register as register_module
from src.pipeline import train as train_module
from src.pipeline.console import configure_utf8_console
from src.pipeline.evaluate import ModelEvaluator, log_metrics_if_possible
from src.pipeline.register import (
    ModelRegistrar,
    require_train_run_id,
    run_registration,
)
from src.pipeline.run_id import (
    LOCAL_RUN_ID,
    read_run_id,
    run_id_path,
    write_run_id,
)
from src.pipeline.train import ModelTrainer, build_model_params

METRICS = {"accuracy": 0.9, "f1": 0.7}


def make_registrar() -> ModelRegistrar:
    """Build a registrar whose MLflow client is a mock."""
    registrar = ModelRegistrar(tracking_uri="http://localhost:5000")
    registrar.client = MagicMock()
    return registrar


def test_run_id_roundtrip(tmp_path: Path) -> None:
    """The run ID written by train is read back unchanged."""
    path = tmp_path / "models" / "mlflow_run_id.txt"
    write_run_id("abc123", path)
    assert read_run_id(path) == "abc123"


def test_read_run_id_missing_or_empty(tmp_path: Path) -> None:
    """A missing or empty file means there is no run ID."""
    path = tmp_path / "run_id.txt"
    assert read_run_id(path) is None
    path.write_text("  \n", encoding="utf-8")
    assert read_run_id(path) is None


def test_run_id_path_uses_model_dir() -> None:
    """The run ID file lives next to the model."""
    assert run_id_path({"model_dir": "out"}) == Path("out/mlflow_run_id.txt")
    assert run_id_path({}) == Path("models/mlflow_run_id.txt")


def test_build_model_params_defaults() -> None:
    """Hyperparameters come from params.yaml with defaults."""
    assert build_model_params({"seed": 7, "train": {"max_depth": 3}}) == {
        "seed": 7,
        "n_estimators": 150,
        "max_depth": 3,
        "min_samples_split": 5,
    }


def test_train_returns_local_run_when_mlflow_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without MLflow, train keeps the model locally and flags a local run."""
    trainer = ModelTrainer(tracking_uri="http://localhost:5000", experiment_name="t")
    monkeypatch.setattr(train_module, "build_model_pipeline", lambda **_: MagicMock())
    monkeypatch.setattr(
        trainer, "log_to_mlflow", MagicMock(side_effect=RuntimeError("down"))
    )
    monkeypatch.setattr(trainer, "save_local_artifact", MagicMock())
    run_id = trainer.train_and_log(MagicMock(), MagicMock(), {}, tmp_path / "m.pkl")
    assert run_id == LOCAL_RUN_ID


def test_evaluate_logs_metrics_in_train_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Evaluate reopens the train run instead of creating another one."""
    fake_mlflow = MagicMock()
    monkeypatch.setattr(evaluate_module, "mlflow", fake_mlflow)
    run_id_file = tmp_path / "run_id.txt"
    write_run_id("train-run", run_id_file)
    evaluator = ModelEvaluator(tracking_uri="http://x", experiment_name="t")

    log_metrics_if_possible(evaluator, METRICS, run_id_file)

    fake_mlflow.start_run.assert_called_once_with(run_id="train-run")
    fake_mlflow.log_metrics.assert_called_once_with(METRICS)


@pytest.mark.parametrize("content", [None, LOCAL_RUN_ID])
def test_evaluate_skips_mlflow_without_train_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str | None
) -> None:
    """Offline train (or no handoff file) means evaluate does not touch MLflow."""
    fake_mlflow = MagicMock()
    monkeypatch.setattr(evaluate_module, "mlflow", fake_mlflow)
    run_id_file = tmp_path / "run_id.txt"
    if content:
        write_run_id(content, run_id_file)
    evaluator = ModelEvaluator(tracking_uri="http://x", experiment_name="t")

    log_metrics_if_possible(evaluator, METRICS, run_id_file)

    fake_mlflow.start_run.assert_not_called()


def test_register_uses_train_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Registration uses the run ID from train, not the latest run."""
    fake_mlflow = MagicMock()
    fake_mlflow.register_model.return_value = MagicMock(version=4)
    monkeypatch.setattr(register_module, "mlflow", fake_mlflow)
    run_id_file = tmp_path / "run_id.txt"
    write_run_id("train-run", run_id_file)
    registrar = make_registrar()

    code = run_registration(registrar, run_id_file, "model", "champion")

    assert code == 0
    fake_mlflow.register_model.assert_called_once_with(
        model_uri="runs:/train-run/model", name="model"
    )
    registrar.client.set_registered_model_alias.assert_called_once_with(
        name="model", alias="champion", version="4"
    )
    registrar.client.search_runs.assert_not_called()


def test_register_unreachable_mlflow_warns_and_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Offline evaluators still get exit code 0 with a warning."""
    registrar = make_registrar()
    registrar.client.search_experiments.side_effect = ConnectionError("refused")

    code = run_registration(registrar, tmp_path / "none.txt", "model", "champion")

    assert code == 0
    assert "MLflow unreachable" in caplog.text


def test_register_failure_with_reachable_mlflow_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A registry error with MLflow up must fail the stage."""
    fake_mlflow = MagicMock()
    fake_mlflow.register_model.side_effect = RuntimeError("model not found")
    monkeypatch.setattr(register_module, "mlflow", fake_mlflow)
    run_id_file = tmp_path / "run_id.txt"
    write_run_id("train-run", run_id_file)

    code = run_registration(make_registrar(), run_id_file, "model", "champion")

    assert code == 1
    assert "Registration FAILED" in caplog.text


def test_register_without_train_run_fails_when_reachable(tmp_path: Path) -> None:
    """MLflow up but train ran offline: fail instead of silently skipping."""
    run_id_file = tmp_path / "run_id.txt"
    write_run_id(LOCAL_RUN_ID, run_id_file)

    assert run_registration(make_registrar(), run_id_file, "m", "champion") == 1
    with pytest.raises(ValueError, match="did not log a run"):
        require_train_run_id(run_id_file)


def test_register_main_exits_with_registration_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage entrypoint turns the return code into the process exit code."""
    monkeypatch.setattr(register_module, "run_registration", lambda *a, **k: 1)
    monkeypatch.setattr(register_module, "ModelRegistrar", MagicMock())
    with pytest.raises(SystemExit) as exc:
        register_module.main()
    assert exc.value.code == 1


def test_configure_utf8_console_replaces_unencodable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After configuring, an emoji on a cp1252 stream no longer raises."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    configure_utf8_console()
    print("\U0001f3c3 View run")
    stream.flush()

    assert "View run" in raw.getvalue().decode("utf-8")


def test_configure_utf8_console_ignores_streams_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured or replaced streams are left alone."""
    monkeypatch.setattr(sys, "stdout", object())
    monkeypatch.setattr(sys, "stderr", object())
    configure_utf8_console()
