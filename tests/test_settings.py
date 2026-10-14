"""Tests for the application settings."""

import pytest
from pydantic import ValidationError

from src.core.settings import Settings


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_settings_reject_threshold_outside_unit_interval(threshold):
    """PREDICTION_THRESHOLD must be a probability."""
    with pytest.raises(ValidationError):
        Settings(PREDICTION_THRESHOLD=threshold, _env_file=None)


@pytest.mark.parametrize("threshold", [0.0, 0.35, 1.0])
def test_settings_accept_threshold_inside_unit_interval(threshold):
    """Boundaries of the unit interval are valid thresholds."""
    settings = Settings(PREDICTION_THRESHOLD=threshold, _env_file=None)

    assert threshold == settings.PREDICTION_THRESHOLD
