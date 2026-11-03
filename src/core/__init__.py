"""Application core: app factory and CORS setup."""

from src.core.config import create_app
from src.core.cors import setup_cors

__all__ = [
    "create_app",
    "setup_cors",
]
