"""API package for DART trading platform.

Provides the modern Deriv API client with OAuth 2.0 authentication.
"""

from api.deriv_client import DerivClient, DerivOAuth

__all__ = [
    "DerivClient",
    "DerivOAuth",
]
