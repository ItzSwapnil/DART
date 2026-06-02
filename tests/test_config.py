"""Tests for configuration validation and management."""

import pytest
from pydantic import ValidationError

from config.dart_config import DARTSettings, get_config


def test_config_singleton() -> None:
    """Test that get_config returns a singleton instance."""
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2


def test_validate_percentages_valid() -> None:
    """Test valid risk settings percentages pass validation."""
    settings = DARTSettings()
    settings.risk.max_portfolio_risk = 0.05
    settings.risk.max_position_size = 0.15
    settings.risk.max_drawdown = 0.25

    # Should not raise validation error
    assert settings.risk.validate_percentages(0.05) == 0.05


def test_validate_percentages_invalid() -> None:
    """Test invalid risk settings percentages fail validation."""
    with pytest.raises(ValidationError):
        # max_portfolio_risk must be between 0 and 1
        DARTSettings(risk={"max_portfolio_risk": 1.5})

    with pytest.raises(ValidationError):
        DARTSettings(risk={"max_position_size": -0.1})


def test_validate_granularity_valid() -> None:
    """Test valid granularity options pass validation."""
    for valid_g in [15, 30, 60, 120, 300, 600, 900, 1800, 3600]:
        settings = DARTSettings(trading={"granularity": valid_g})
        assert settings.trading.granularity == valid_g


def test_validate_granularity_invalid() -> None:
    """Test invalid granularity options raise validation error."""
    with pytest.raises(ValidationError):
        DARTSettings(trading={"granularity": 45})


def test_validate_configuration_warnings() -> None:
    """Test validate_configuration returns False when critical credentials are missing."""
    settings = DARTSettings()
    settings.api.deriv_access_token = ""
    # Should report warning and return False
    assert settings.validate_configuration() is False


def test_validate_configuration_success() -> None:
    """Test validate_configuration returns True when critical credentials are set."""
    settings = DARTSettings()
    settings.api.deriv_access_token = "valid_oauth_token"
    settings.api.deriv_account_id = "CR12345"
    assert settings.validate_configuration() is True
