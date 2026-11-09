"""AiPASS model discovery and conversation-model filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .codec import walk


_EXCLUDED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:image|seedream|seedance|veo|lyria|banana)",
        r"^gemini-.*-image",
        r"^gpt-image",
        r"deep-research",
    )
)


@dataclass(frozen=True)
class Model:
    id: str
    display_name: str
    provider: str
    provider_name: str = ""
    description: str = ""
    ready: bool = True
    selectable: bool = True
    thinking_levels: tuple[str, ...] = field(default_factory=tuple)
    default_thinking_level: str | None = None

    def openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "owned_by": self.provider or "aipass",
            "name": self.display_name,
            "description": self.description,
            "capabilities": {
                "reasoning": bool(self.thinking_levels),
                "thinking_levels": list(self.thinking_levels),
                "default_thinking_level": self.default_thinking_level,
            },
        }


GEMINI_ENTERPRISE_MODELS: tuple[Model, ...] = (
    Model(
        id="gemini-4-argon",
        display_name="Gemini 4 Argon",
        provider="google",
        provider_name="Google Gemini Enterprise",
        description="Gemini Enterprise Flagship: การคิดวิเคราะห์ขั้นสูงและงานเชิงตัวแทนอัตโนมัติ (High Reasoning & Agentic Workflows)",
        thinking_levels=("low", "medium", "high", "max"),
        default_thinking_level="high",
    ),
    Model(
        id="gemini-3.8-flash",
        display_name="Gemini 3.8 Flash",
        provider="google",
        provider_name="Google Gemini Enterprise",
        description="Gemini Enterprise Default: โมเดลหลักความเร็วสูง ฉลาด รอบด้าน บริบท 1M tokens",
        thinking_levels=("low", "medium", "high"),
        default_thinking_level="low",
    ),
    Model(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        provider="google",
        provider_name="Google Gemini Enterprise",
        description="Gemini Enterprise Balanced: ประมวลผลรวดเร็ว เสถียร สำหรับงานวิเคราะห์ทั่วไป",
        thinking_levels=("low", "medium", "high"),
        default_thinking_level="low",
    ),
    Model(
        id="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash Lite",
        provider="google",
        provider_name="Google Gemini Enterprise",
        description="Gemini Enterprise Ultra-Fast: ไม่หักเครดิต ความเร็วสูงสุด ตอบสนองทันที",
        thinking_levels=("low", "medium", "high"),
        default_thinking_level="low",
    ),
    Model(
        id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        provider="google",
        provider_name="Google Gemini Enterprise",
        description="Gemini Enterprise Pro: สำหรับงานแก้ปัญหาซับซ้อน งานคณิตศาสตร์ และโค้ดเชิงลึก",
        thinking_levels=("low", "medium", "high", "max"),
        default_thinking_level="medium",
    ),
)


def is_conversation_model(model_id: str) -> bool:
    if any(pattern.search(model_id) for pattern in _EXCLUDED_PATTERNS):
        return False
    return "gemini" in model_id.lower()


def _from_record(record: dict[str, Any]) -> Model | None:
    model_id = record.get("id")
    display_name = record.get("displayName")
    provider = record.get("provider")
    if not all(isinstance(item, str) and item for item in (model_id, display_name, provider)):
        return None
    if not is_conversation_model(model_id):
        return None
    thinking = record.get("thinkingConfig")
    levels: tuple[str, ...] = ()
    default: str | None = None
    if isinstance(thinking, dict):
        raw_levels = thinking.get("supportedLevels")
        if isinstance(raw_levels, list):
            levels = tuple(str(level) for level in raw_levels if isinstance(level, str))
        raw_default = thinking.get("defaultLevel")
        default = raw_default if isinstance(raw_default, str) else None
    return Model(
        id=model_id,
        display_name=display_name,
        provider=provider,
        provider_name=str(record.get("providerName") or ""),
        description=str(record.get("description") or ""),
        ready=record.get("ready") is not False,
        selectable=record.get("selectable") is not False,
        thinking_levels=levels,
        default_thinking_level=default,
    )


def extract_models(decoded: Any = None) -> list[Model]:
    if not decoded:
        return list(GEMINI_ENTERPRISE_MODELS)
    found: dict[str, Model] = {}
    for value in walk(decoded):
        if not isinstance(value, dict):
            continue
        model = _from_record(value)
        if model and model.ready and model.selectable and "gemini" in model.id.lower():
            found[model.id] = model
    if not found:
        return list(GEMINI_ENTERPRISE_MODELS)
    return list(found.values())


def map_reasoning(requested: str | None, model: Model) -> str | None:
    levels = list(model.thinking_levels)
    if not levels:
        return None
    if not requested:
        return model.default_thinking_level if model.default_thinking_level in levels else None
    requested = requested.lower()
    if requested in {"none", "minimal"}:
        return None
    desired = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "max",
        "ultra": "max",
    }.get(requested, requested)
    if desired in levels:
        return desired
    order = ["low", "medium", "high", "max"]
    available = [level for level in order if level in levels]
    if not available:
        return model.default_thinking_level if model.default_thinking_level in levels else levels[0]
    target = order.index(desired) if desired in order else order.index("medium")
    return min(available, key=lambda level: abs(order.index(level) - target))

