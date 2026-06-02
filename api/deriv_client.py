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
import re
import secrets
import urllib.parse
import webbrowser
from base64 import urlsafe_b64encode
from datetime import UTC, datetime
from typing import Any

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
        self._code_verifier: str | None = None
        self._state: str | None = None

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

        assert self._code_verifier is not None
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

    def login_interactive(self, scope: str = "trade", timeout: int = 120) -> str | None:
        """Open browser for OAuth login and capture the callback code.

        Spins up a tiny local HTTP server, opens the authorization URL in the
        default browser, and waits for the redirect with the auth code.

        Returns:
            The authorization code, or None on timeout/failure.
        """
        self.generate_pkce()
        auth_url = self.get_authorization_url(scope=scope)

        captured: dict[str, str | None] = {"code": None}
        parsed_redirect = urllib.parse.urlparse(self.redirect_uri)
        port = parsed_redirect.port or 8080

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                captured["code"] = qs.get("code", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication successful!</h2>"
                    b"<p>You can close this tab and return to DART.</p></body></html>"
                )

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
        server.timeout = timeout

        logger.info("Opening browser for Deriv OAuth login...")
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        return captured["code"]

    @property
    def code_verifier(self) -> str | None:
        return self._code_verifier

    @property
    def state(self) -> str | None:
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
        access_token: str | None = None,
        account_id: str | None = None,
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
        self.account_info: dict[str, Any] | None = None
        self.active_symbols_raw: list[dict[str, Any]] = []

        self._public_ws = None
        self._trading_ws = None
        self._public_ws_lock = asyncio.Lock()
        self._trading_ws_lock = asyncio.Lock()
        self._ws_connect_lock = asyncio.Lock()

    def _is_ws_open(self, ws) -> bool:
        """Check if WebSocket connection is open (compatible with old/new websockets versions)."""
        if ws is None:
            return False
        state = getattr(ws, "state", None)
        if state is not None:
            try:
                import websockets

                return state == websockets.State.OPEN
            except Exception:
                return state == 1
        return getattr(ws, "open", False)

    async def _get_public_ws(self):
        """Get or establish a persistent public WebSocket connection."""
        async with self._ws_connect_lock:
            if not self._is_ws_open(self._public_ws):
                logger.info("Opening persistent public WebSocket connection...")
                self._public_ws = await websockets.connect(self.PUBLIC_WS)
            return self._public_ws

    async def _get_trading_ws(self):
        """Get or establish a persistent authenticated trading WebSocket connection."""
        async with self._ws_connect_lock:
            if not self._is_ws_open(self._trading_ws):
                logger.info("Opening persistent trading WebSocket connection...")
                ws = await self._open_trading_ws()
                if ws is None:
                    raise RuntimeError("Unable to establish authenticated Deriv trading WebSocket")
                self._trading_ws = ws
            return self._trading_ws

    async def _send_and_receive_public(
        self,
        message: dict[str, Any],
        *,
        match_key: str | None = None,
        timeout: int = 12,
    ) -> dict[str, Any]:
        """Send a message and receive a response using the persistent public WebSocket, safely locked."""
        async with self._public_ws_lock:
            ws = await self._get_public_ws()
            return await self._send_and_receive(ws, message, match_key=match_key, timeout=timeout)

    async def _send_and_receive_trading(
        self,
        message: dict[str, Any],
        *,
        match_key: str | None = None,
        timeout: int = 12,
    ) -> dict[str, Any]:
        """Send a message and receive a response using the persistent trading WebSocket, safely locked."""
        async with self._trading_ws_lock:
            ws = await self._get_trading_ws()
            return await self._send_and_receive(ws, message, match_key=match_key, timeout=timeout)

    def _is_pat(self) -> bool:
        """Check if access_token is a Personal Access Token (PAT).
        Typically, standard API tokens are 15-character alphanumeric strings,
        whereas OAuth 2.0 access tokens are much longer.
        """
        if not self.access_token:
            return False
        return len(self.access_token) < 40

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
        match_key: str | None = None,
        timeout: int = 12,
    ) -> dict[str, Any]:
        """Send one request and wait for a matching response."""
        req_id = message.get("req_id")
        await ws.send(json.dumps(message))

        loop = asyncio.get_running_loop()
        end_time = loop.time() + timeout

        while True:
            remaining = end_time - loop.time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for Deriv WebSocket response")

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                raise TimeoutError("Timed out waiting for Deriv WebSocket response") from None
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                raise

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

    async def _get_otp_ws_url(self) -> str | None:
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
        """Open an authenticated trading WebSocket.

        If it is a PAT, connects to the standard ws.derivws.com and sends authorize.
        Otherwise, falls back to the OTP WebSocket or public endpoint.
        """
        if self._is_pat():
            url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
            try:
                ws = await websockets.connect(url)
                # Authorize immediately
                auth_resp = await self._send_and_receive(
                    ws, {"authorize": self.access_token, "req_id": 10}, match_key="authorize"
                )
                # Cache account info
                auth_data = auth_resp.get("authorize", {})
                self.account_info = {
                    "account_id": auth_data.get("loginid"),
                    "balance": str(auth_data.get("balance", "0")),
                    "currency": auth_data.get("currency", "USD"),
                    "email": auth_data.get("email"),
                    "fullname": auth_data.get("fullname"),
                    "is_virtual": auth_data.get("is_virtual"),
                }
                self.account_id = self.account_info["account_id"]
                self.is_connected = True
                return ws
            except Exception as e:
                logger.error("PAT WebSocket connection failed: %s", e)
                return None

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
            resp = await self._send_and_receive_public(
                {"active_symbols": "brief", "req_id": 1},
                match_key="active_symbols",
            )
            if "error" in resp:
                logger.error("active_symbols failed: %s", resp.get("error"))
                return {}
            self.active_symbols_raw = resp.get("active_symbols", [])
            symbols = {}
            for item in self.active_symbols_raw:
                symbol = item.get("symbol") or item.get("underlying_symbol")
                display_name = item.get("display_name") or item.get("underlying_symbol_name") or symbol
                market_name = item.get("market_display_name") or item.get("market") or ""

                name = f"{market_name} - {display_name}" if market_name else str(display_name)
                if not item.get("exchange_is_open", True):
                    name = f"{name} [CLOSED]"

                if symbol:
                    symbols[name] = symbol
                    symbols[str(display_name)] = symbol
                    symbols[str(symbol)] = symbol
            return symbols
        except Exception as e:
            logger.error("Error fetching active symbols: %s", e)
            return {}

    async def get_candles(self, symbol: str, granularity: int, count: int = 50) -> list[dict[str, Any]]:
        """Retrieve candle history for a symbol (public endpoint)."""
        try:
            resp = await self._send_and_receive_public(
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
        retry_count: int = 0,
    ) -> dict[str, Any]:
        """Get a contract price proposal.
        
        Automatically scales down the stake amount if the proposal fails with a payout limit error.
        """
        try:
            # Round stake to 2 decimal places as Deriv API rejects higher precision stakes
            amount = round(amount, 2)
            req = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": duration,
                "duration_unit": duration_unit,
                "req_id": 3,
            }
            if self._is_pat():
                req["symbol"] = symbol
            else:
                req["underlying_symbol"] = symbol
            resp = await self._send_and_receive_trading(req, match_key="proposal")
            if "error" in resp:
                err_msg = resp["error"].get("message", "")
                if "maximum payout" in err_msg.lower() and retry_count < 2:
                    # Parse limit and current values from message
                    match = re.search(
                        r"maximum payout of\s+([0-9.,]+).+?Current payout is\s+([0-9.,]+)",
                        err_msg,
                        re.IGNORECASE,
                    )
                    if match:
                        max_payout = float(match.group(1).replace(",", "").rstrip("."))
                        current_payout = float(match.group(2).replace(",", "").rstrip("."))
                        if current_payout > 0:
                            # Scale down the stake
                            factor = max_payout / current_payout
                            # Apply a safety margin to stay slightly below maximum payout
                            adjusted_amount = round(amount * factor * 0.99, 2)
                            # Ensure we don't fall below minimum stake (usually 0.50)
                            if adjusted_amount >= 0.50:
                                logger.info(
                                    f"Scaling down stake from {amount:.2f} to {adjusted_amount:.2f} "
                                    f"due to maximum payout limit ({max_payout:.2f})"
                                )
                                return await self.get_contract_proposal(
                                    symbol=symbol,
                                    contract_type=contract_type,
                                    amount=adjusted_amount,
                                    duration=duration,
                                    duration_unit=duration_unit,
                                    retry_count=retry_count + 1,
                                )
                return resp
            return resp.get("proposal", {})
        except Exception as e:
            logger.error("Error getting contract proposal: %s", e)
            return {"error": {"message": str(e)}}

    async def buy_contract(
        self,
        symbol: str,
        contract_type: str,
        amount: float,
        duration: int,
        duration_unit: str = "s",
        price: float | None = None,
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

        try:
            buy_req = {
                "buy": proposal_id,
                "price": buy_price,
                "req_id": 4,
            }
            resp = await self._send_and_receive_trading(buy_req, match_key="buy")
            if "error" in resp:
                return resp

            buy_data = resp.get("buy", {})
            contract_id = buy_data.get("contract_id")
            if contract_id is not None:
                self.active_trades[str(contract_id)] = {
                    "id": str(contract_id),
                    "symbol": symbol,
                    "contract_type": contract_type,
                    "amount": buy_price,
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "buy_response": resp,
                    "status": "open",
                    "start_time": datetime.now(UTC),
                }

            return resp
        except Exception as e:
            logger.error("Error buying contract: %s", e)
            return {"error": {"message": str(e)}}

    async def check_trade_status(self, contract_id: str) -> dict[str, Any] | None:
        """Check the status of an open contract."""
        if str(contract_id) not in self.active_trades:
            return None

        try:
            resp = await self._send_and_receive_trading(
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
            if not isinstance(info, dict):
                return None
            trade_key = str(contract_id)
            self.active_trades[trade_key]["current_info"] = info

            if info.get("status") in ["sold", "expired", "won", "lost"]:
                record = self.active_trades.pop(trade_key)
                record["status"] = "closed"
                record["end_time"] = datetime.now(UTC)

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

    async def check_all_trades(self) -> dict[str, dict[str, Any] | None]:
        """Check all currently tracked active trades."""
        results = {}
        for cid in list(self.active_trades.keys()):
            results[cid] = await self.check_trade_status(cid)
        return results

    # -- Account management (REST) ------------------------------------------

    async def get_account_info(self) -> dict[str, Any] | None:
        """Get account info via REST or WebSocket (depends on token type)."""
        if not self.access_token:
            logger.warning("access_token required for account info")
            return None

        if self._is_pat():
            try:
                # Trigger persistent trading WS which will authorize and cache account_info automatically
                await self._get_trading_ws()
                return self.account_info
            except Exception as e:
                logger.error("PAT account info failed: %s", e)
                self.is_connected = False
                return None

        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                http_resp = await client.get(self.ACCOUNTS_URL, headers=self._auth_headers())
                http_resp.raise_for_status()
                data = http_resp.json().get("data", [])
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

    async def get_account_balance(self) -> dict[str, Any] | None:
        """Get account balance from cached or freshly loaded account info."""
        if self._is_pat() or self.account_info is None:
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
            resp = await self._send_and_receive_public({"ping": 1, "req_id": 6}, match_key="ping")
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

    async def get_available_durations(self, symbol: str, contract_type: str) -> list[int] | None:
        """Get available standard durations for a symbol + contract type."""
        if not symbol:
            logger.error("Symbol is required to get available durations")
            return None

        try:
            resp = await self._send_and_receive_public(
                {"contracts_for": symbol, "req_id": 8},
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

                standard = [15, 30, 45, 60, 90, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400]
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
