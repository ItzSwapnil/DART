"""Tests for API utility helpers and decorators."""

import asyncio

import pytest

from utils.api_utils import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    calculate_backoff_delay,
    is_retryable_error,
    retry_async,
)


def test_retry_config() -> None:
    """Test RetryConfig initialization."""
    config = RetryConfig(max_retries=5, base_delay=2.0)
    assert config.max_retries == 5
    assert config.base_delay == 2.0


def test_calculate_backoff_delay() -> None:
    """Test exponential backoff calculations with and without jitter."""
    config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

    # 1.0 * (2.0 ^ 0) = 1.0
    assert calculate_backoff_delay(0, config) == 1.0
    # 1.0 * (2.0 ^ 2) = 4.0
    assert calculate_backoff_delay(2, config) == 4.0

    # Cap at max delay
    config.max_delay = 5.0
    assert calculate_backoff_delay(3, config) == 5.0  # 1 * 8 capped at 5.0


def test_is_retryable_error() -> None:
    """Test retryable error classification logic."""
    # Retryable errors
    assert is_retryable_error(Exception("Websocket connection closed")) is True
    assert is_retryable_error(Exception("Timeout occurred")) is True
    assert is_retryable_error(Exception("Rate limit exceeded")) is True

    # Non-retryable errors
    assert is_retryable_error(Exception("Invalid Token")) is False
    assert is_retryable_error(Exception("Insufficient Balance")) is False


@pytest.mark.asyncio
async def test_circuit_breaker_transitions() -> None:
    """Test circuit breaker transitions between CLOSED, OPEN, and HALF-OPEN states."""
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    # Initial state
    assert await breaker.can_execute() is True
    assert breaker.state == CircuitState.CLOSED

    # Record 1 failure
    await breaker.record_failure()
    assert await breaker.can_execute() is True
    assert breaker.state == CircuitState.CLOSED

    # Record 2nd failure -> Circuit opens
    await breaker.record_failure()
    assert await breaker.can_execute() is False
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.15)

    # Check transitions to half-open
    assert breaker.state == CircuitState.HALF_OPEN
    assert await breaker.can_execute() is True  # Allows 1 test call

    # Next call is blocked while testing
    assert await breaker.can_execute() is False

    # Record success -> Circuit closes
    await breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert await breaker.can_execute() is True


@pytest.mark.asyncio
async def test_retry_async_decorator() -> None:
    """Test that retry_async decorator performs retries on failure."""
    calls = 0

    @retry_async(config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False))
    async def failing_function() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("Temporary network timeout")
        return "Success"

    res = await failing_function()
    assert res == "Success"
    assert calls == 3  # Called 3 times (1 initial + 2 retries)


@pytest.mark.asyncio
async def test_retry_async_decorator_non_retryable() -> None:
    """Test that retry_async decorator raises immediately on non-retryable error."""
    calls = 0

    @retry_async(config=RetryConfig(max_retries=2, base_delay=0.01))
    async def failing_function() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("Invalid token provided")

    with pytest.raises(ValueError):
        await failing_function()

    # Should fail fast and not retry
    assert calls == 1
