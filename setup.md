# 🛠️ DART v3.1 — Detailed Setup & Onboarding Guide

Welcome to the **Deep Adaptive Reinforcement Trader (DART) v3.1** installation and configuration guide. This document is written specifically to help you set up and obtain everything you need to run DART in a state-of-the-art AI trading configuration.

As per system instructions, this setup guide assumes you are running on **Windows** using **WSL (Windows Subsystem for Linux)** with **uv** as the primary package manager. All runtime commands must be executed within your active WSL Linux environment.

---

## 🗺️ Table of Contents
1. [Prerequisites & WSL Setup](#1-prerequisites--wsl-setup)
2. [Installing uv & Project Dependencies](#2-installing-uv--project-dependencies)
3. [Obtaining Deriv Developer Credentials (OAuth 2.0 PKCE)](#3-obtaining-deriv-developer-credentials-oauth-2-0-pkce)
4. [Setting up the AI Engine (vLLM & Qwen)](#4-setting-up-the-ai-engine-vllm--qwen)
5. [Configuring Environment Variables (.env)](#5-configuring-environment-variables-env)
6. [Running the System Components](#6-running-the-system-components)
7. [Running Tests & Verifying Installation](#7-running-tests--verifying-installation)

---

## 1. Prerequisites & WSL Setup

To ensure optimal performance, native compatibility with deep learning libraries (like PyTorch), and SOTA execution speeds, DART runs inside **Windows Subsystem for Linux (WSL)**.

### Step 1.1: Install WSL 2 (if not already installed)
1. Open PowerShell on Windows as Administrator.
2. Run the following command:
   ```powershell
   wsl --install
   ```
3. Restart your computer if prompted. This installs the default **Ubuntu** Linux distribution.
4. Set up your Unix username and password.

### Step 1.2: Open WSL Terminal
1. Open your terminal of choice (e.g., Windows Terminal, VS Code Terminal with WSL profile, or launch `Ubuntu` from the Start Menu).
2. Ensure you are inside the DART project directory:
   ```bash
   cd /mnt/c/Users/Swapnil/Projects/DART
   ```
   *(Note: `/mnt/c/` corresponds to the `C:\` drive in Windows)*

---

## 2. Installing uv & Project Dependencies

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It manages virtual environments and dependencies seamlessly.

### Step 2.1: Install uv in WSL
Run the official installation script inside your WSL terminal:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your WSL shell or run `source $HOME/.local/bin/env` to add `uv` to your PATH. Verify the installation:
```bash
uv --version
```

### Step 2.2: Create and Activate the Virtual Environment
Create a virtual environment pinned to **Python 3.14.5**:
```bash
uv venv .venv --python 3.14.5
```
Activate the virtual environment:
```bash
source .venv/bin/activate
```
> [!IMPORTANT]
> Always ensure your virtual environment is active before running any scripts. Your terminal prompt should show `(.venv)` at the beginning.

### Step 2.3: Install DART Dependencies
Install the package in editable mode along with all optional dependency groups (development, deep-learning, LLM, market data, and monitoring):
```bash
uv pip install -e ".[all]"
```
This reads the configuration defined in [pyproject.toml](file:///c:/Users/Swapnil/Projects/DART/pyproject.toml) and installs everything needed.

---

## 3. Obtaining Deriv Developer Credentials (OAuth 2.0 PKCE)

DART v3.1 uses **OAuth 2.0 Authorization Code Flow with PKCE (Proof Key for Code Exchange)** for security, deprecating static API tokens.

### Step 3.1: Register a Deriv Application
1. Go to the [Deriv Developer Portal Portal](https://developers.deriv.com/) and log in with your Deriv account.
2. Go to **Manage App / Register App** (or click [here](https://developers.deriv.com/api-registration/)).
3. Fill in the application details:
   - **Application Name**: `DART Algorithmic Trader`
   - **Redirect URL**: `http://localhost:8080/callback` (This redirect URL matches the default redirect URI used by [DerivClient](file:///c:/Users/Swapnil/Projects/DART/api/deriv_client.py#L182))
   - **Scopes**: Tick **Read**, **Trade**, and **Trading Information** (you can optionally select *Payments* and *Admin* if you require full dashboard capabilities).
4. Click **Register**.
5. Copy your generated **App ID** and **OAuth Client ID** (which is usually the same as App ID or a specific Client ID string).

### Step 3.2: Interactive Login and Obtaining Access Token
To perform live trading, DART requires an active OAuth `access_token` and `account_id` (a currency account e.g. `CR12345` or virtual account `VR12345`).

DART contains an interactive script that automatically performs the PKCE authentication flow, spins up a local server, handles the callback, and prints out your credentials:
1. Run the interactive login sequence inside WSL:
   ```bash
   python -c "import asyncio; from api.deriv_client import DerivOAuth; auth = DerivOAuth(client_id='YOUR_APP_ID'); code = auth.login_interactive(); print('Auth Code:', code)"
   ```
2. Your web browser will open a Deriv login page. Authenticate and authorize the application.
3. Upon success, the browser redirects back to `http://localhost:8080/callback`. The background HTTP server captures the code and terminates.
4. You will get the access token and account ID. Save these for the `.env` file.

---

## 4. Setting up the AI Engine (vLLM & Qwen)

DART uses **Soft Actor-Critic (SAC)** Deep Reinforcement Learning (defined in [SoftActorCriticV2](file:///c:/Users/Swapnil/Projects/DART/ml/deep_rl_agent.py#L21)) combined with market regime detection and **LLM-based Market Analysis** (defined in [LLMMarketAnalyzer](file:///c:/Users/Swapnil/Projects/DART/ml/llm_market_analyzer.py#L21)).

To power the LLM Market Analyzer, you must run an OpenAI-compatible local inference server. **vLLM** is recommended for high throughput and GPU acceleration on Windows (via WSL CUDA).

### Step 4.1: Start the local vLLM Server
1. DART is configured to look for the LLM at `http://localhost:8000/v1` and expects `Qwen/Qwen2.5-1.5B-Instruct` (a fast, lightweight, instruction-following model).
2. Inside WSL (with your GPU drivers and PyTorch configured), run:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
   ```
3. The server will download the model weights from Hugging Face and start serving on port `8000`.

> [!TIP]
> If you do not have a dedicated GPU or want to test without a local LLM server running, you can set `AI__REQUIRE_REAL_LLM=False` and `AI__USE_LLM_ANALYSIS=False` (or let it use fallback heuristics) in your configuration.

---

## 5. Configuring Environment Variables (.env)

The project utilizes Pydantic v2 settings (defined in [DARTSettings](file:///c:/Users/Swapnil/Projects/DART/config/dart_config.py#L103)) to load variables from a `.env` file located in the root directory.

Create a `.env` file in `/mnt/c/Users/Swapnil/Projects/DART/.env` with the following contents:

```env
# DART App General Settings
DEBUG_MODE=False

# API Credentials (OAuth 2.0 PKCE)
API__DERIV_APP_ID=72212
API__DERIV_OAUTH_CLIENT_ID=your_deriv_oauth_client_id
API__DERIV_OAUTH_REDIRECT_URI=http://localhost:8080/callback
API__DERIV_ACCESS_TOKEN=your_oauth_access_token
API__DERIV_ACCOUNT_ID=your_deriv_account_id

# AI & LLM Engine Settings
AI__USE_DEEP_RL=True
AI__USE_ENSEMBLE_ML=True
AI__USE_LLM_ANALYSIS=True
AI__REQUIRE_REAL_LLM=True
API__LLM_BASE_URL=http://localhost:8000/v1
AI__LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Risk Settings
RISK__MAX_DAILY_LOSS=100.0
RISK__MAX_PORTFOLIO_RISK=0.02
RISK__CONFIDENCE_THRESHOLD=0.40

# Trading Settings
TRADING__SYMBOL=R_75
TRADING__GRANULARITY=60
TRADING__BASE_TRADE_AMOUNT=10.0
TRADING__AUTO_TRADING_ENABLED=False

# Monitoring & Logs
MONITORING__WANDB_ENABLED=False
MONITORING__WANDB_PROJECT=DART-v3
MONITORING__LOG_LEVEL=INFO
```

---

## 6. Running the System Components

Once your virtual environment is active, credentials are set, and your LLM server is running, you can launch DART components:

### A. Run the Demo Trading & ML Pipeline
To verify training and data extraction, execute:
```bash
python main.py
```
This runs the code in [main.py](file:///c:/Users/Swapnil/Projects/DART/main.py) which initializes the model, feature extractor, and risk manager, trains on simulated market data, and prints live predictions.

### B. Launch the Web Dashboard (Streamlit)
To see the modern glassmorphism UI, real-time charts, uncertainty scores, and indicators, launch the Streamlit app:
```bash
streamlit run ui/modern_dashboard.py
```
Streamlit will output a local URL (e.g., `http://localhost:8501`). Open it in your Windows browser to interact with DART.

### C. Launch the Desktop GUI
*Note: Running CustomTkinter GUIs inside WSL requires an X-Server (like GWSL or VcXsrv) or WSLg configured on Windows 11.*
```bash
python ui/app.py
```

### D. Run the Auto-Trading Daemon
To start the fully automated background trading loop:
```bash
python ml/auto_trader.py
```
This runs the [AutoTrader](file:///c:/Users/Swapnil/Projects/DART/ml/auto_trader.py#L29) engine which connects to Deriv WebSockets, polls ticks, extracts features, queries the AI for signals, performs risk checks, and executes contracts.

---

## 7. Running Tests & Verifying Installation

To ensure everything is correctly configured, run the pytest test suite in WSL:
```bash
pytest
```
Expected output:
```
tests/test_imports.py::TestCoreImports::test_import_deriv_client PASSED
...
============================= 11 passed in 35.32s ==============================
```
This verifies that all deep learning modules, config schemas, and websocket clients import successfully and contain the correct OAuth fields.

---

## 💡 Troubleshooting Tips

- **WSL Port Forwarding**: Windows automatically forwards ports from `localhost:8080` (OAuth callback) and `localhost:8000` (vLLM) to WSL. If you encounter connectivity errors, check your Windows Firewall settings.
- **OutOfMemory (OOM) Errors**: If vLLM crashes with OOM, start the server with CPU execution only, or decrease the GPU utilization parameter:
  ```bash
  python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --gpu-memory-utilization 0.7
  ```
- **Authentication Failures**: Double check that your `API__DERIV_ACCESS_TOKEN` is fresh. OAuth tokens expire; run the interactive login helper to generate a new token and update your `.env` file.
