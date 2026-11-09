# Hermes AiPASS & Gemini Enterprise

An unofficial standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent)
provider for TH-AI Passport and Gemini Enterprise. It exposes authenticated
models as a local OpenAI-compatible endpoint, so Hermes can use text models,
stream replies, and run its normal tool loop across CLI, TUI, Desktop, and WebUI.

> [!WARNING]
> AiPASS does not publish this as a public API. This plugin follows the same
> internal routes as the web application. Those routes can change without
> notice. Use it only with your own account and review the applicable terms.

## Requirements

- macOS or Linux Container with Mac-assisted authentication
- Hermes Agent (CLI, TUI, Desktop, or WebUI)
- Brave Browser (default, via remote debugging port 9222) or Google Chrome
- An active TH-AI Passport account with Gemini Enterprise access

## Install

```bash
hermes plugins install newbiexM/Hermes_TH-AI-pass_Gemini-Enterprise --enable
hermes aipass setup --set-default
hermes aipass login
```

After login, authentication credentials are saved securely in macOS Keychain with `auth.json` file fallback for container portability. Start Hermes normally:

```bash
hermes
hermes -m aipass/gpt-6.1-sol
hermes -m aipass-gemini-enterprise/gemini-3.8-flash
hermes -z "Summarize this repository"
```

Use `/model` inside Hermes to select between **TH-AI Passport** models and **AiPASS Gemini Enterprise** models.

## Management

```bash
hermes aipass status
hermes aipass models --refresh
hermes aipass doctor
hermes aipass restart
hermes aipass logout
```

The bridge listens only on `127.0.0.1:8792`. Authentication secrets are stored
in macOS Keychain. Runtime state and redacted logs live below `HERMES_HOME`.

## Reasoning levels

Hermes reasoning levels are mapped to levels advertised by each AiPASS model:
`low`, `medium`, `high`, and `max`. Unsupported values are clamped to the
nearest available level. Models without thinking metadata use their web default.

## Troubleshooting

- **401 / session expired:** run `hermes aipass login` again.
- **403 / request denied:** the selected model may require paid credits or
  additional account access. Confirm that it works in the AiPASS web chat.
- **Bridge unavailable:** run `hermes aipass restart`, then `hermes aipass doctor`.
- **Missing model:** run `hermes aipass models --refresh`; unavailable and
  non-conversation models are intentionally hidden.
- **Protocol mismatch:** AiPASS may have changed its internal transport. Run
  `hermes aipass doctor` and include its redacted output in a bug report.

## Uninstall

```bash
hermes aipass stop
hermes plugins remove hermes-aipass
```

`logout` is deliberately separate so an update or uninstall does not silently
delete account credentials or conversation mappings.

## Known limitations

- macOS only in v0.1.
- One upstream request is processed at a time.
- Image, video, music, and workflow-specific deep-research models are excluded.
- Internal AiPASS route changes may temporarily break the plugin.

## License

MIT
