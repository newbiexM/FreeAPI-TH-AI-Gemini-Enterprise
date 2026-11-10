# Hermes Gemini Enterprise Provider Bridge

An unofficial standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent)
provider exclusively dedicated to **Gemini Enterprise** models (`gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, and `gemini-3.1-pro-preview`).

It exposes an authenticated local OpenAI-compatible endpoint, enabling Hermes to leverage Gemini Enterprise reasoning, extended context (1M tokens), and streaming across CLI, TUI, Desktop, and WebUI without interference from third-party commercial LLMs.

> [!NOTE]
> This bridge connects directly to Gemini Enterprise infrastructure with Mac-assisted authentication (Brave Browser session extraction on port 9222 and Keychain / `auth.json` fallback).

## Supported Models

| Model ID | Display Name | Role & Reasoning | Context |
| :--- | :--- | :--- | :--- |
| `gemini-4-argon` | Gemini 4 Argon | Flagship: High Reasoning & Agentic Workflows (`low`, `medium`, `high`, `max`) | 1M |
| `gemini-3.8-flash` | Gemini 3.8 Flash (**Default**) | Fast, versatile, primary agent model (`low`, `medium`, `high`) | 1M |
| `gemini-3.5-flash` | Gemini 3.5 Flash | Balanced general-purpose model (`low`, `medium`, `high`) | 1M |
| `gemini-3.5-flash-lite` | Gemini 3.5 Flash Lite | Ultra-fast zero-credit response (`low`, `medium`, `high`) | 1M |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro Preview | Complex coding, reasoning & mathematics (`low`, `medium`, `high`, `max`) | 1M |

## Requirements

- macOS or Linux Container with Mac-assisted authentication
- Hermes Agent (CLI, TUI, Desktop, or WebUI)
- Brave Browser (default, via remote debugging port 9222)
- An active account with Gemini Enterprise access

## Quickstart

```bash
# 1. Setup & Login via Brave Browser
hermes aipass setup --set-default
hermes aipass login

# 2. Run Hermes with Gemini Enterprise
hermes -m gemini-enterprise/gemini-3.8-flash
hermes -z "สรุป repository นี้ให้หน่อย"
```

Use `/model` inside Hermes to switch between Gemini Enterprise models.

## Management

```bash
hermes aipass status
hermes aipass models
hermes aipass doctor
hermes aipass restart
hermes aipass logout
```

The bridge daemon listens on `127.0.0.1:8792`. Credentials are stored securely in macOS Keychain with file fallback (`auth.json`) for container synchronization.

## Reasoning Levels

Hermes reasoning levels map cleanly to Gemini Enterprise thinking configurations: `low`, `medium`, `high`, and `max`. Unsupported values clamp to the nearest valid level.

## Incremental Streaming Architecture

The bridge features a zero-latency `StreamInterceptor` (adapted from `qwen-free-api`) with:
- **True Real-Time SSE**: Upstream tokens stream immediately to Hermes with zero perceived Time-to-First-Token (TTFT) latency.
- **Sliding Hold-back Window (`STREAM_KEEP = 32`)**: Prevents split tool markers from leaking into user-visible conversation text across chunk cuts.
- **Initial Refusal Gating**: Intercepts upfront model refusal messages before emitting HTTP headers, enabling seamless and transparent automatic retries.
- **Tolerant Multi-Syntax Parsing**: Robust parsing for `<tool_call>`, `<<<TOOL_CALL>>>`, and `@@HERMES_TOOL_CALL@@` with code fence stripping and bracket tolerance.

## Dual-Protocol Support (OpenAI & Anthropic)

The bridge provides parity across two major agent protocols on `127.0.0.1:8792`:
- **OpenAI Endpoint**: `POST /v1/chat/completions` (Hermes CLI, TUI, Desktop, WebUI, Cursor).
- **Anthropic Endpoints**: `POST /v1/messages` and `POST /v1/messages/count_tokens` (Claude Code, Cline, Roo-Code).

## Multi-Account Resilient Pool

Pass multiple tokens via `AIPASS_TOKENS="tok1,tok2,tok3"`:
- **Round-Robin Rotation**: Dispatches queries across healthy accounts.
- **Exponential Cooldown on 429**: Backs off on rate limit (`60s * 2^(streak - 1)` up to 30m) with automatic pre-stream failover to the next healthy account.

## License

MIT
