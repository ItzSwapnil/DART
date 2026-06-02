"""Tests for the Deriv client and OAuth authentication flows."""

from api.deriv_client import DerivClient, DerivOAuth


def test_oauth_pkce_generation() -> None:
    """Test OAuth PKCE verifier and challenge generation."""
    oauth = DerivOAuth(client_id="test_client")
    oauth.generate_pkce()

    assert oauth.code_verifier is not None
    assert oauth.state is not None
    assert len(oauth.code_verifier) >= 43
    assert len(oauth.code_verifier) <= 128


def test_oauth_authorization_url() -> None:
    """Test OAuth URL contains expected client ID and PKCE properties."""
    oauth = DerivOAuth(client_id="72212")
    oauth.generate_pkce()

    url = oauth.get_authorization_url(scope="trade")

    assert "client_id=72212" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "scope=trade" in url


def test_is_pat() -> None:
    """Test detection of Personal Access Tokens (PATs) vs standard OAuth tokens."""
    # Standard PAT tokens are short alphanumeric strings (e.g. 15 chars)
    client_pat = DerivClient(access_token="abc123XYZ789000")
    assert client_pat._is_pat() is True

    # Empty access token should not be detected as PAT
    client_empty = DerivClient(access_token="")
    assert client_empty._is_pat() is False

    # Standard OAuth access tokens are long JWTs or similar (e.g. 40+ characters)
    long_token = "eyJhY2NvdW50X2lkIjogIkNSMTIzNDUiLCAic2NvcGUiOiAidHJhZGUifQ.token"
    client_oauth = DerivClient(access_token=long_token)
    assert client_oauth._is_pat() is False
