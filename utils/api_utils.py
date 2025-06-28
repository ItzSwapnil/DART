"""
API Utilities for Robust Error Handling

This module provides retry logic, circuit breaker pattern, and connection management
for the Deriv API integration in the DART trading application.
"""

import asyncio
import logging
import random
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("api_utils")

# Type variable for generic async functions
T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


class CircuitState(Enum):
    """States for the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests go through
    OPEN = "open"  # Failure threshold reached, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.

    When too many failures occur, the circuit "opens" and blocks further requests
    for a recovery period. After the recovery period, it allows a test request
    through ("half-open") to see if the service has recovered.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, updating if recovery time has passed."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (
                time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
        return self._state

    async def can_execute(self) -> bool:
        """Check if a request can be executed."""
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.CLOSED:
                return True

            if current_state == CircuitState.OPEN:
                return False

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker closing after successful test request")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker re-opening after failed test request")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures. "
                    f"Blocking requests for {self.recovery_timeout}s",
                )

    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        logger.info("Circuit breaker manually reset")


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> float:
    """
    Calculate delay for exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter (0-50% of delay)
    if config.jitter:
        jitter_amount = delay * random.uniform(0, 0.5)
        delay += jitter_amount

    return delay


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception that occurred

    Returns:
        True if the error is retryable
    """
    error_str = str(error).lower()

    # Non-retryable errors - fail fast
    non_retryable = [
        "invalid token",
        "authorization failed",
        "invalid appid",
        "permission denied",
        "account blocked",
        "insufficient balance",
    ]

    for phrase in non_retryable:
        if phrase in error_str:
            return False

    # Retryable errors
    retryable = [
        "connection",
        "timeout",
        "temporary",
        "rate limit",
        "too many requests",
        "service unavailable",
        "internal error",
        "sorry, an error occurred",  # Generic Deriv API error
        "websocket",
        "network",
    ]

    for phrase in retryable:
        if phrase in error_str:
            return True

    # Default: retry for unknown errors (conservative approach)
    return True


def retry_async(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
):
    """
    Decorator for async functions that implements retry with exponential backoff.

    Args:
        config: Retry configuration (uses default if not provided)
        circuit_breaker: Optional circuit breaker instance

    Usage:
        @retry_async()
        async def my_api_call():
            ...

        @retry_async(config=RetryConfig(max_retries=5))
        async def my_important_api_call():
            ...
    """
    if config is None:
        config = DEFAULT_RETRY_CONFIG

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Check circuit breaker if provided
            if circuit_breaker:
                if not await circuit_breaker.can_execute():
                    raise CircuitBreakerError(
                        f"Circuit breaker is open for {func.__name__}",
                    )

            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)

                    # Record success with circuit breaker
                    if circuit_breaker:
                        await circuit_breaker.record_success()

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if this error is retryable
                    if not is_retryable_error(e):
                        logger.warning(
                            f"{func.__name__} failed with non-retryable error: {e}",
                        )
                        if circuit_breaker:
                            await circuit_breaker.record_failure()
                        raise

                    # Check if we have retries left
                    if attempt >= config.max_retries:
                        logger.error(
                            f"{func.__name__} failed after {attempt + 1} attempts: {e}",
                        )
                        if circuit_breaker:
                            await circuit_breaker.record_failure()
                        raise

                    # Calculate delay and log retry
                    delay = calculate_backoff_delay(attempt, config)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s...",
                    )

                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class ConnectionManager:
    """
    Manages API connection state and provides reconnection logic.

    Tracks connection health and provides methods for reconnection
    with proper state management.
    """

    def __init__(self, max_reconnect_attempts: int = 5, reconnect_delay: float = 2.0):
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._is_connected = False
        self._last_successful_connection: Optional[float] = None
        self._reconnect_attempts = 0
        self._connection_callbacks: list[Callable[[bool], None]] = []
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Get current connection status."""
        return self._is_connected

    def register_callback(self, callback: Callable[[bool], None]):
        """Register a callback to be notified of connection status changes."""
        self._connection_callbacks.append(callback)

    def _notify_callbacks(self, is_connected: bool):
        """Notify all registered callbacks of connection status change."""
        for callback in self._connection_callbacks:
            try:
                callback(is_connected)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    async def mark_connected(self):
        """Mark the connection as established."""
        async with self._lock:
            was_connected = self._is_connected
            self._is_connected = True
            self._last_successful_connection = time.time()
            self._reconnect_attempts = 0

            if not was_connected:
                logger.info("Connection established")
                self._notify_callbacks(True)

    async def mark_disconnected(self):
        """Mark the connection as lost."""
        async with self._lock:
            was_connected = self._is_connected
            self._is_connected = False

            if was_connected:
                logger.warning("Connection lost")
                self._notify_callbacks(False)

    async def attempt_reconnect(self, reconnect_func: Callable[[], Any]) -> bool:
        """
        Attempt to reconnect using the provided function.

        Args:
            reconnect_func: Async function that attempts to establish connection

        Returns:
            True if reconnection was successful
        """
        async with self._lock:
            self._reconnect_attempts += 1

            if self._reconnect_attempts > self.max_reconnect_attempts:
                logger.error(
                    f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded",
                )
                return False

        try:
            delay = self.reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            delay = min(delay, 60.0)  # Cap at 60 seconds

            logger.info(
                f"Attempting reconnection (attempt {self._reconnect_attempts}/{self.max_reconnect_attempts}) "
                f"in {delay:.1f}s...",
            )

            await asyncio.sleep(delay)
            await reconnect_func()

            await self.mark_connected()
            return True

        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            return False

    def reset(self):
        """Reset connection state for fresh start."""
        self._is_connected = False
        self._last_successful_connection = None
        self._reconnect_attempts = 0
