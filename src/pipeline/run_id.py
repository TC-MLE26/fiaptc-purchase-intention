"""Handoff of the MLflow run ID from the train stage to the later stages."""

from pathlib import Path
from typing import Any

RUN_ID_FILENAME = "mlflow_run_id.txt"
LOCAL_RUN_ID = "local_run"


def run_id_path(train_cfg: dict[str, Any]) -> Path:
    """Return where the train stage stores the MLflow run ID.

    Args:
        train_cfg: The ``train`` section of params.yaml.

    Returns:
        Path of the run ID file, next to the trained model.
    """
    return Path(train_cfg.get("model_dir", "models")) / RUN_ID_FILENAME


def write_run_id(run_id: str, path: str | Path) -> None:
    """Persist the run ID so evaluate and register reuse the train run.

    Args:
        run_id: MLflow run ID, or ``LOCAL_RUN_ID`` when MLflow was unavailable.
        path: Destination file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_id, encoding="utf-8")


def read_run_id(path: str | Path) -> str | None:
    """Read the run ID written by the train stage.

    Args:
        path: File written by ``write_run_id``.

    Returns:
        The run ID, or None when the file is missing or empty.
    """
    source = Path(path)
    if not source.exists():
        return None
    return source.read_text(encoding="utf-8").strip() or None
