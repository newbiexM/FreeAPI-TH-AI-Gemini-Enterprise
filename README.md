# Hermes AiPASS

An unofficial standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent)
provider for TH-AI Passport. It exposes the authenticated AiPASS web transport
as a local OpenAI-compatible endpoint, so Hermes can use AiPASS text models,
stream replies, and run its normal tool loop from the terminal.

> [!WARNING]
> AiPASS does not publish this as a public API. This plugin follows the same
> internal routes as the web application. Those routes can change without
> notice. Use it only with your own account and review the applicable terms.

## Requirements

- macOS (the first release uses Keychain and LaunchAgent)
- Hermes Agent
- Google Chrome, needed only while signing in or renewing an expired session
- An active AiPASS account

## Install

```bash
hermes plugins install Yeet2042/hermes-aipass --enable
hermes aipass setup --set-default
hermes aipass login
```

After login, Chrome closes and is not used while chatting. Start Hermes normally:

```bash
hermes
hermes -m gpt-5.6-sol
hermes -z "Summarize this repository"
```

Use `/model` inside Hermes to select any currently available AiPASS text model.

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
