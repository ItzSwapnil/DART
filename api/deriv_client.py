"""
DART v3.1 — Modern Deriv API Client
OAuth 2.0 PKCE authentication + OTP-based WebSocket trading.

Authentication flow:
  1. OAuth 2.0 Authorization Code + PKCE → access_token
  2. POST /accounts/{id}/otp → OTP WebSocket URL
  3. Connect to OTP WebSocket URL for authenticated trading
  4. Public data uses wss://api.derivws.com/trading/v1/options/ws/public (no auth)
"""

import asyncio
import hashlib
import http.server
import json
import logging
import os
import re
import secrets
import threading
import urllib.parse
import webbrowser
from base64 import urlsafe_b64encode
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import websockets

logger = logging.getLogger("deriv_client")


# ---------------------------------------------------------------------------
# OAuth 2.0 PKCE Helper
# ---------------------------------------------------------------------------

class DerivOAuth:
    """OAuth 2.0 Authorization Code flow with PKCE for Deriv."""

    AUTH_URL = "https://auth.deriv.com/oauth2/auth"
    TOKEN_URL = "https://auth.deriv.com/oauth2/token"

    def __init__(self, client_id: str, redirect_uri: str = "http://localhost:8080/callback"):
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self._code_verifier: Optional[str] = None
        self._state: Optional[str] = None

    # -- PKCE helpers -------------------------------------------------------

    @staticmethod
    def _generate_code_verifier(length: int = 64) -> str:
        """Generate a cryptographically random code_verifier (43-128 chars)."""
        charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
        return "".join(secrets.choice(charset) for _ in range(length))

    @staticmethod
    def _derive_code_challenge(verifier: str) -> str:
        """Derive code_challenge = BASE64URL(SHA256(code_verifier))."""
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def generate_pkce(self) -> tuple[str, str, str]:
        """Generate PKCE parameters and random state.

        Returns:
            (code_verifier, code_challenge, state)
        """
        self._code_verifier = self._generate_code_verifier()
        challenge = self._derive_code_challenge(self._code_verifier)
        self._state = secrets.token_hex(16)
        return self._code_verifier, challenge, self._state

    # -- Authorization URL --------------------------------------------------

    def get_authorization_url(self, scope: str = "trade") -> str:
        """Build the Deriv OAuth authorization URL.

        Call ``generate_pkce()`` first to populate verifier/state.
        """
        if self._code_verifier is None or self._state is None:
            self.generate_pkce()

        challenge = self._derive_code_challenge(self._code_verifier)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": self._state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    # -- Token exchange -----------------------------------------------------

    async def exchange_code_for_token(self, code: str) -> dict[str, Any]:
        """Exchange authorization code for access_token.

        Args:
            code: Authorization code received from Deriv callback.

        Returns:
            Token response dict: {"access_token": "...", "expires_in": ..., "token_type": "Bearer"}
        """
        if self._code_verifier is None:
            raise RuntimeError("PKCE code_verifier missing — call generate_pkce() first")

        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "code_verifier": self._code_verifier,
            "redirect_uri": self.redirect_uri,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                self.TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            return resp.json()

    # -- Local callback server (desktop flow) -------------------------------

    def login_interactive(self, scope: str = "trade", timeout: int = 120) -> Optional[str]:
        """Open browser for OAuth login and capture the callback code.

        Spins up a tiny local HTTP server, opens the authorization URL in the
        default browser, and waits for the redirect with the auth code.

        Returns:
            The authorization code, or None on timeout/failure.
        """
        self.generate_pkce()
        auth_url = self.get_authorization_url(scope=scope)

        captured: dict[str, Optional[str]] = {"code": None}
        parsed_redirect = urllib.parse.urlparse(self.redirect_uri)
        port = parsed_redirect.port or 8080

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_inner):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self_inner.path).query)
                captured["code"] = qs.get("code", [None])[0]
                self_inner.send_response(200)
                self_inner.send_header("Content-Type", "text/html")
                self_inner.end_headers()
                self_inner.wfile.write(
                    b"<html><body><h2>Authentication successful!</h2>"
                    b"<p>You can close this tab and return to DART.</p></body></html>"
                )

            def log_message(self_inner, *args):  # noqa: D401 — suppress noisy logs
                pass

        server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
        server.timeout = timeout

        logger.info("Opening browser for Deriv OAuth login...")
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        return captured["code"]

    @property
    def code_verifier(self) -> Optional[str]:
        return self._code_verifier

    @property
    def state(self) -> Optional[str]:
        return self._state


# ---------------------------------------------------------------------------
# Deriv Trading Client
# ---------------------------------------------------------------------------

class DerivClient:
    """Modern Deriv API client — OAuth 2.0 + OTP WebSocket.

    Authentication hierarchy:
      1. ``access_token`` + ``account_id`` → REST OTP → authenticated WebSocket
      2. Public WebSocket (no auth) for market data only
    """

    BASE_REST = "https://api.derivws.com"
    PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
    ACCOUNTS_URL = f"{BASE_REST}/trading/v1/options/accounts"
    OTP_URL = f"{BASE_REST}/trading/v1/options/accounts/{{account_id}}/otp"
    HEALTH_URL = f"{BASE_REST}/v1/health"

    def __init__(
        self,
        app_id: str = "72212",
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
        http_timeout: int = 20,
    ):
        self.app_id = str(app_id)
        self.access_token = access_token
        self.account_id = account_id
        self.http_timeout = http_timeout

        self.active_trades: dict[str, dict[str, Any]] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.consecutive_losses = 0
        self.is_connected = False
        self.account_info: Optional[dict[str, Any]] = None

    # -- REST helpers -------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Standard headers for authenticated REST calls."""
        headers = {"Deriv-App-ID": self.app_id}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    # -- WebSocket helpers --------------------------------------------------

    async def _send_and_receive(
        self,
        ws,
        message: dict[str, Any],
        *,
        match_key: Optional[str] = None,
        timeout: int = 12,
    ) -> dict[str, Any]:
        """Send one request and wait for a matching response."""
        req_id = message.get("req_id")
        await ws.send(json.dumps(message))

        loop = asyncio.get_running_loop()
        end_time = loop.time() + timeout

        while True:
            if loop.time() > end_time:
                raise asyncio.TimeoutError("Timed out waiting for Deriv WebSocket response")

            raw = await ws.recv()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                return data
            if match_key and match_key in data:
                return data
            if req_id is not None and data.get("req_id") == req_id:
                return data
            if match_key is None:
                return data

    # -- OTP-based authenticated WebSocket ----------------------------------

    async def _get_otp_ws_url(self) -> Optional[str]:
        """Obtain an OTP-authenticated WebSocket URL via REST."""
        if not (self.access_token and self.account_id):
            return None

        url = self.OTP_URL.format(account_id=self.account_id)
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.post(url, headers=self._auth_headers())
                resp.raise_for_status()
                return resp.json().get("data", {}).get("url")
        except Exception as e:
            logger.error("Failed to fetch OTP WebSocket URL: %s", e)
            return None

    async def _open_trading_ws(self):
        """Open an authenticated trading WebSocket via OTP.

        Falls back to the public endpoint (no trading) if OAuth is not configured.
        """
        otp_url = await self._get_otp_ws_url()
        if otp_url:
            try:
                return await websockets.connect(otp_url)
            except Exception as e:
                logger.error("OTP WebSocket connection failed: %s", e)

        # Fallback: public endpoint (market data only, no trading)
        logger.warning("No OAuth credentials — connecting to public WebSocket (read-only)")
        try:
            return await websockets.connect(self.PUBLIC_WS)
        except Exception as e:
            logger.error("Public WebSocket connection failed: %s", e)
            return None

    # -- Market Data (public, no auth) --------------------------------------

    async def get_active_symbols(self) -> dict[str, str]:
        """Fetch active market symbols from public WebSocket."""
        try:
            async with websockets.connect(self.PUBLIC_WS) as ws:
                resp = await self._send_and_receive(
                    ws,
                    {"active_symbols": "brief", "req_id": 1},
                    match_key="active_symbols",
                )
                if "error" in resp:
                    logger.error("active_symbols failed: %s", resp.get("error"))
                    return {}

                symbols = {}
                for item in resp.get("active_symbols", []):
                    market_name = f"{item.get('market_display_name')} - {item.get('display_name')}"
                    if not item.get("exchange_is_open", True):
                        market_name = f"{market_name} [CLOSED]"
                    symbols[market_name] = item.get("symbol")
                return symbols
        except Exception as e:
            logger.error("Error fetching active symbols: %s", e)
            return {}

    async def get_candles(self, symbol: str, granularity: int, count: int = 50) -> list[dict[str, Any]]:
        """Retrieve candle history for a symbol (public endpoint)."""
        try:
            async with websockets.connect(self.PUBLIC_WS) as ws:
                resp = await self._send_and_receive(
                    ws,
                    {
                        "ticks_history": symbol,
                        "count": count,
                        "end": "latest",
                        "style": "candles",
                        "granularity": granularity,
                        "req_id": 2,
                    },
                    timeout=15,
                )
                if "error" in resp:
                    logger.error("ticks_history failed: %s", resp.get("error"))
                    return []
                return resp.get("candles") or resp.get("history") or []
        except Exception as e:
            logger.error("Error fetching candles: %s", e)
            return []

    async def get_historical_data(self, symbol: str, granularity: int, days: int = 7) -> list[dict[str, Any]]:
        """Retrieve historical candle data with bounded count."""
        seconds_per_day = 24 * 60 * 60
        total_seconds = max(1, days) * seconds_per_day
        count = int(min(total_seconds // max(1, granularity), 5000))
        return await self.get_candles(symbol, granularity, count=count)

    # -- Trading (authenticated via OTP) ------------------------------------

    async def get_contract_proposal(
        self,
        symbol: str,
        contract_type: str,
        amount: float,
        duration: int,
        duration_unit: str = "s",
    ) -> dict[str, Any]:
        """Get a contract price proposal."""
        ws = await self._open_trading_ws()
        if not ws:
            return {"error": {"message": "Unable to establish Deriv WebSocket"}}

        try:
            req = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "underlying_symbol": symbol,
                "req_id": 3,
            }
            resp = await self._send_and_receive(ws, req, match_key="proposal")
            if "error" in resp:
                return resp
            return resp.get("proposal", {})
        except Exception as e:
            logger.error("Error getting contract proposal: %s", e)
            return {"error": {"message": str(e)}}
        finally:
            await ws.close()

    async def buy_contract(
        self,
        symbol: str,
        contract_type: str,
        amount: float,
        duration: int,
        duration_unit: str = "s",
        price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Buy a contract: proposal → buy in one flow."""
        proposal = await self.get_contract_proposal(
            symbol=symbol,
            contract_type=contract_type,
            amount=amount,
            duration=duration,
            duration_unit=duration_unit,
        )
        if "error" in proposal:
            return proposal

        proposal_id = proposal.get("id")
        if not proposal_id:
            return {"error": {"message": "Missing proposal id from Deriv response"}}

        buy_price = float(price) if price is not None else float(proposal.get("ask_price", amount))

        ws = await self._open_trading_ws()
        if not ws:
            return {"error": {"message": "Unable to establish Deriv WebSocket"}}

        try:
            buy_req = {
                "buy": proposal_id,
                "price": buy_price,
                "req_id": 4,
            }
            resp = await self._send_and_receive(ws, buy_req, match_key="buy")
            if "error" in resp:
                return resp

            buy_data = resp.get("buy", {})
            contract_id = buy_data.get("contract_id")
            if contract_id is not None:
                self.active_trades[str(contract_id)] = {
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "amount": amount,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "buy_response": resp,
                    "status": "open",
                    "start_time": datetime.now(timezone.utc),
                }

            return resp
        except Exception as e:
            logger.error("Error buying contract: %s", e)
            return {"error": {"message": str(e)}}
        finally:
            await ws.close()

    async def check_trade_status(self, contract_id: str) -> Optional[dict[str, Any]]:
        """Check the status of an open contract."""
        if str(contract_id) not in self.active_trades:
            return None

        ws = await self._open_trading_ws()
        if not ws:
            return None

        try:
            resp = await self._send_and_receive(
                ws,
                {
                    "proposal_open_contract": 1,
                    "contract_id": int(contract_id),
                    "req_id": 5,
                },
                match_key="proposal_open_contract",
            )
            if "error" in resp:
                logger.error("proposal_open_contract failed: %s", resp.get("error"))
                return None

            info = resp.get("proposal_open_contract", {})
            trade_key = str(contract_id)
            self.active_trades[trade_key]["current_info"] = info

            if info.get("status") in ["sold", "expired"]:
                record = self.active_trades.pop(trade_key)
                record["status"] = "closed"
                record["end_time"] = datetime.now(timezone.utc)

                buy_price = float(info.get("buy_price", 0) or 0)
                sell_price = float(info.get("sell_price", 0) or 0)
                profit = sell_price - buy_price
                record["profit"] = profit
                self.trade_history.append(record)

                if profit < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

            return info
        except Exception as e:
            logger.error("Error checking trade status: %s", e)
            return None
        finally:
            await ws.close()

    async def check_all_trades(self) -> dict[str, Optional[dict[str, Any]]]:
        """Check all currently tracked active trades."""
        results = {}
        for cid in list(self.active_trades.keys()):
            results[cid] = await self.check_trade_status(cid)
        return results

    # -- Account management (REST) ------------------------------------------

    async def get_account_info(self) -> Optional[dict[str, Any]]:
        """Get account info via REST (OAuth required)."""
        if not self.access_token:
            logger.warning("OAuth access_token required for account info")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.get(self.ACCOUNTS_URL, headers=self._auth_headers())
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if isinstance(data, list) and data:
                    if self.account_id:
                        matched = next(
                            (x for x in data if x.get("account_id") == self.account_id),
                            data[0],
                        )
                        self.account_info = matched
                    else:
                        self.account_info = data[0]
                        self.account_id = self.account_info.get("account_id")
                    self.is_connected = True
                    return self.account_info
        except Exception as e:
            logger.error("REST account info failed: %s", e)

        self.is_connected = False
        return None

    async def get_account_balance(self) -> Optional[dict[str, Any]]:
        """Get account balance from cached or freshly loaded account info."""
        if self.account_info is None:
            await self.get_account_info()

        if self.account_info:
            balance = self.account_info.get("balance")
            currency = self.account_info.get("currency")
            if balance is not None and currency:
                return {"balance": balance, "currency": currency}

        return None

    # -- Connection health --------------------------------------------------

    async def check_connection(self) -> bool:
        """Check connectivity via WebSocket ping."""
        try:
            async with websockets.connect(self.PUBLIC_WS) as ws:
                resp = await self._send_and_receive(ws, {"ping": 1, "req_id": 6}, match_key="ping")
                self.is_connected = resp.get("ping") == "pong"
                return self.is_connected
        except Exception as e:
            logger.error("Connection check failed: %s", e)
            self.is_connected = False
            return False

    async def check_health(self) -> bool:
        """Check Deriv REST API health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                resp = await client.get(self.HEALTH_URL)
                return resp.status_code == 200
        except Exception:
            return False

    # -- Contract metadata --------------------------------------------------

    async def get_available_durations(self, symbol: str, contract_type: str) -> Optional[list[int]]:
        """Get available standard durations for a symbol + contract type."""
        if not symbol:
            logger.error("Symbol is required to get available durations")
            return None

        try:
            async with websockets.connect(self.PUBLIC_WS) as ws:
                resp = await self._send_and_receive(
                    ws,
                    {"contracts_for": symbol, "currency": "USD", "req_id": 8},
                    match_key="contracts_for",
                )
                if "error" in resp:
                    logger.error("Error getting contracts: %s", resp.get("error"))
                    return None

                contracts = resp.get("contracts_for", {}).get("available", [])
                available_durations: list[int] = []

                for contract in contracts:
                    if contract.get("contract_type") != contract_type:
                        continue

                    min_duration = contract.get("min_contract_duration")
                    max_duration = contract.get("max_contract_duration")

                    if not (min_duration and max_duration):
                        continue

                    min_seconds = self._duration_to_seconds(min_duration)
                    max_seconds = self._duration_to_seconds(max_duration)

                    standard = [30, 60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400]
                    for dur in standard:
                        if min_seconds <= dur <= max_seconds:
                            available_durations.append(dur)

                available_durations = sorted(set(available_durations))
                if not available_durations:
                    logger.warning(
                        "No available durations found for %s with contract type %s",
                        symbol,
                        contract_type,
                    )
                return available_durations

        except Exception as e:
            logger.error("Error getting available durations: %s", e)
            return None

    @staticmethod
    def _duration_to_seconds(duration_str: str) -> int:
        """Convert a duration string like 5m, 2h, 1d, 10t to seconds."""
        if not duration_str:
            return 0

        match = re.match(r"^(\d+)([smhdtSMHDT])$", str(duration_str).strip())
        if not match:
            return 0

        value = int(match.group(1))
        unit = match.group(2).lower()

        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "t": 1}
        return value * multipliers.get(unit, 0)
