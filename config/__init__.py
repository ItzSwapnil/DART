"""Configuration package for DART trading platform.

Re-exports modern Pydantic v2 configuration from dart_config.
"""

from config.dart_config import DARTSettings, get_config, load_config

__all__ = [
    "DARTSettings",
    "get_config",
    "load_config",
]
