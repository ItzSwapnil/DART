# 🔐 Security Policy

## Project: [DART - Deep Adaptive Reinforcement Trader](https://github.com/ItzSwapnil/DART)

---

## 📅 Supported Versions

We actively support the latest stable release:

| Version | Supported          |
|---------|--------------------|
| latest  | ✅ Supported        |
| dev     | 🚧 Development Only |
| < prev  | ❌ Not Supported    |

---

## ⚠️ Disclaimer & Responsible Use

> ⚠️ **WARNING:** This bot performs real-time financial trading using AI algorithms and live market data. Trading in financial markets involves **significant risk**, including the potential loss of capital.  
>  
> **By using DART, you agree to the following**:
>
> - Use the bot **at your own risk**.
> - You are solely responsible for any trades executed by the bot.
> - You must comply with all applicable **regulations**, **exchange terms of service**, and **legal requirements** in your jurisdiction.
> - The maintainers of this project provide **no warranty or guarantee** of profitability, uptime, or data accuracy.
> - This software is **experimental** and intended for educational and research purposes only.

---

## 🛡️ Threat Model

Because DART operates in live environments with AI, WebSocket streams, and trading APIs, it is vulnerable to the following:

- API key or session token leakage
- Trade hijacking or spoofing
- Adversarial AI inputs or model poisoning
- Man-in-the-middle attacks over WebSockets
- Insecure GUI or code injection
- Risky model self-modification (code updates via LLMs)

---

## 🔒 Security Best Practices

### ✅ Credential Management
- Store secrets in `.env` or use a secret vault.
- Do **not** hard-code any sensitive keys.
- Never commit authentication tokens to Git.

### ✅ API & WebSocket Safety
- Always use `wss://` and verify SSL certificates.
- Validate and sanitize all inbound and outbound JSON.
- Limit API usage scopes and rotate keys frequently.

### ✅ Data Handling
- Separate testing and production data environments.
- Sanitize all streamed or stored inputs into InfluxDB/VectorDB.
- Monitor memory usage, as live model learning can be data-intensive.

### ✅ AI Model Protections
- Disable auto-retraining in production unless signed and verified.
- Avoid injecting untrusted data into LLMs or reinforcement learners.
- Implement logging and rollback for model updates.

### ✅ UI/Server Safety
- Lock local dashboards to `localhost` if not intended for web access.
- Implement rate-limiting and access control if deployed on a network.
- Disable code execution from the frontend UI.

---

## 📩 Responsible Disclosure

We value the security community. To report a vulnerability:

1. DO NOT include any sensitive information when creating a public GitHub issue.
2. Include:
   - Description and severity
   - Reproduction steps
   - Suggested mitigation (if any)

We aim to respond in **72 hours** and patch critical issues within **7 business days**.

---

## 🧪 Developer Security Tools

- ✅ `bandit` – static security scanner for Python
- ✅ `safety` – dependency vulnerability checker
- ✅ `dotenv` – for secure config management
- ✅ `pyright` – type checks to catch unsafe logic
- ✅ `pytest` – test framework for code integrity

---

## ❗ Known Vulnerabilities

None at this time in the current release.  
Check the [issues](https://github.com/ItzSwapnil/DART/issues) page for updates.

---

## 📌 Final Note

Use this software **wisely**, **securely**, and **ethically**.  
By running DART, you accept **full responsibility** for your trading outcomes and usage behavior.

---

