"""Hermes AiPASS plugin entrypoint."""

from .hermes_aipass.cli import dispatch, register_cli
from .hermes_aipass.middleware import attach_session_header


def register(ctx) -> None:
    """Register the native CLI surface and session-scoping middleware."""
    ctx.register_cli_command(
        name="aipass",
        help="Set up and manage the AiPASS model provider",
        setup_fn=register_cli,
        handler_fn=dispatch,
        description="Authenticate and use TH-AI Passport models from Hermes.",
    )
    ctx.register_middleware("llm_request", attach_session_header)

