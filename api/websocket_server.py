"""
DART v3.2 — WebSocket Server for Real-Time UI Updates
Launches a local WebSocket server to broadcast trade status, balance updates,
and performance metrics from the AutoTrader to the Streamlit UI.
"""

import asyncio
import json
import logging
import threading
from typing import Any

import websockets

logger = logging.getLogger("websocket_server")


class DARTWebSocketServer:
    """
    Lightweight WebSocket server that runs alongside the AutoTrader.
    Broadcasts real-time events to all connected clients.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set[Any] = set()
        self.server = None
        self._loop = None
        self._lock = threading.Lock()

    async def _register(self, websocket):
        """Register a new client connection."""
        with self._lock:
            self.clients.add(websocket)
        logger.info(f"WebSocket client connected from {websocket.remote_address}")

    async def _unregister(self, websocket):
        """Unregister a client connection."""
        with self._lock:
            if websocket in self.clients:
                self.clients.remove(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

    async def _handler(self, websocket, path=None):
        """Handle client lifecycle and messages."""
        await self._register(websocket)
        try:
            # Keep client connection alive, listen for messages (optional)
            async for _ in websocket:
                # Echo or handle incoming messages if needed in the future
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._unregister(websocket)

    async def start(self):
        """Start the WebSocket server in the current event loop."""
        self._loop = asyncio.get_running_loop()
        try:
            self.server = await websockets.serve(self._handler, self.host, self.port)
            logger.info(f"DART WebSocket Server started on wss://{self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server on port {self.port}: {e}")
            return False

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("DART WebSocket Server stopped")

    def broadcast(self, message: dict[str, Any]):
        """Broadcast a message to all registered clients. Safe to call from any thread."""
        if not self._loop or not self.server:
            return

        payload = json.dumps(message)
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    async def _broadcast_async(self, payload: str):
        """Asynchronous implementation of broadcast."""
        with self._lock:
            targets = list(self.clients)

        if not targets:
            return

        # Broadcast concurrently to all connected clients
        await asyncio.gather(*[client.send(payload) for client in targets], return_exceptions=True)
