
"""Modular LLM service supporting Ollama and Hugging Face.

* Provider and model are configured through environment settings.
* Ollama runs locally without API credits.
* Hugging Face remains available as an optional provider.
* Retries with backoff on 429/5xx/timeouts.
* Maps authentication and model errors to application exceptions.
* ajson provides schema-validated structured output with one repair retry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading

from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.errors import (
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.utils.jsonparse import extract_json


log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRYABLE = {408, 425, 429, 500, 502, 503, 504}

_STATUS_RE = re.compile(
    r"\b([45]\d\d)\s+(?:Client|Server)\s+Error|"
    r"status(?:\s+code)?[:=\s]+([45]\d\d)",
    re.I,
)


def msgs(system: str, user: str) -> list[BaseMessage]:
    """Create system and user messages."""
    return [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]


def _status(exc: BaseException) -> int | None:
    """Extract HTTP status code from an exception."""

    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)

        if isinstance(value, int) and 400 <= value < 600:
            return value

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)

    if isinstance(value, int):
        return value

    match = _STATUS_RE.search(str(exc)[:400])

    return int(match.group(1) or match.group(2)) if match else None


def _is_retryable(
    exc: BaseException,
    code: int | None,
) -> bool:
    """Determine whether a failed request should be retried."""

    if code is not None:
        return code in RETRYABLE

    name = type(exc).__name__.lower()

    return (
        any(
            term in name
            for term in (
                "timeout",
                "connect",
                "readerror",
                "remoteprotocol",
            )
        )
        or isinstance(exc, (asyncio.TimeoutError, ConnectionError))
    )


def _text(content: Any) -> str:
    """Convert model response content into plain text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict)
            else str(block)
            for block in content
        )

    return str(content or "")


def _to_error(
    exc: BaseException,
    code: int | None,
    provider: str,
) -> LLMError:
    """Convert provider exceptions into application errors."""

    if code in (401, 403):
        if provider == "huggingface":
            return LLMConfigError(
                "Hugging Face rejected the request. "
                "Check your token permissions and model access."
            )

        return LLMConfigError(
            "Authentication or permission error from Ollama. "
            "Check your Ollama server configuration."
        )

    if code == 404:
        return LLMConfigError(
            "The requested model was not found. "
            "Check the model name and run ollama pull if using Ollama."
        )

    if code == 429:
        return LLMRateLimitError(
            "The LLM provider is rate limiting requests. "
            "Please retry shortly."
        )

    return LLMUnavailableError(
        "The LLM is temporarily unavailable. "
        "Check that the model server is running and try again."
    )


class LLMService:
    """LLM service with Ollama and Hugging Face support."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: Any = None,
    ):
        self.settings = settings or get_settings()
        self._model = model
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        """Build and cache the configured LLM."""

        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build()

        return self._model

    def _build(self) -> Any:
        """Initialize the selected LLM provider."""

        s = self.settings
        provider = s.llm_provider.lower().strip()

        # -----------------------------------------
        # Ollama - Local LLM
        # -----------------------------------------

        if provider == "ollama":

            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise LLMConfigError(
                    "Ollama integration is missing. "
                    "Install it using: pip install langchain-ollama"
                ) from exc

            log.info(
                "Initializing Ollama model: %s",
                s.ollama_model,
            )

            return ChatOllama(
                model=s.ollama_model,
                base_url=s.ollama_base_url,
                temperature=s.llm_temperature,
                num_predict=s.llm_max_new_tokens,
                client_kwargs={
                    "timeout": s.llm_timeout_seconds,
                },
            )

        # -----------------------------------------
        # Hugging Face - Cloud inference
        # -----------------------------------------

        if provider == "huggingface":

            if not s.hf_token:
                raise LLMConfigError(
                    "HF_TOKEN is not configured."
                )

            try:
                from langchain_huggingface import (
                    ChatHuggingFace,
                    HuggingFaceEndpoint,
                )
            except ImportError as exc:
                raise LLMConfigError(
                    "Install langchain-huggingface."
                ) from exc

            kwargs: dict[str, Any] = dict(
                repo_id=s.llm_model_id,
                task="conversational",
                huggingfacehub_api_token=s.hf_token,
                max_new_tokens=s.llm_max_new_tokens,
                temperature=max(s.llm_temperature, 0.01),
                timeout=s.llm_timeout_seconds,
            )

            if s.hf_provider:
                kwargs["provider"] = s.hf_provider

            endpoint = HuggingFaceEndpoint(**kwargs)

            return ChatHuggingFace(llm=endpoint)

        raise LLMConfigError(
            f"Unsupported LLM provider: {provider}"
        )

    # ---------------------------------------------
    # Async text generation
    # ---------------------------------------------

    async def ainvoke_text(
        self,
        messages: list[BaseMessage],
    ) -> str:

        attempts = self.settings.llm_max_retries + 1
        provider = self.settings.llm_provider.lower()

        for attempt in range(attempts):

            try:
                response = await self.model.ainvoke(messages)
                return _text(response.content)

            except LLMError:
                raise

            except Exception as exc:

                code = _status(exc)

                if (
                    attempt + 1 >= attempts
                    or not _is_retryable(exc, code)
                ):
                    log.warning(
                        "LLM call failed (%s): %s",
                        code,
                        exc,
                    )

                    raise _to_error(
                        exc,
                        code,
                        provider,
                    ) from exc

                delay = min(2**attempt, 8) + random.random() * 0.3

                log.warning(
                    "LLM retry %s/%s after %.2f seconds",
                    attempt + 1,
                    attempts,
                    delay,
                )

                await asyncio.sleep(delay)

        raise LLMUnavailableError()

    # ---------------------------------------------
    # Streaming text generation
    # ---------------------------------------------

    async def astream_text(
        self,
        messages: list[BaseMessage],
    ) -> AsyncIterator[str]:

        attempts = self.settings.llm_max_retries + 1
        provider = self.settings.llm_provider.lower()

        for attempt in range(attempts):

            started = False

            try:

                async for chunk in self.model.astream(messages):

                    piece = _text(chunk.content)

                    if piece:
                        started = True
                        yield piece

                return

            except LLMError:
                raise

            except Exception as exc:

                code = _status(exc)

                # Avoid retrying after tokens have been emitted.
                if (
                    started
                    or attempt + 1 >= attempts
                    or not _is_retryable(exc, code)
                ):
                    log.warning(
                        "LLM stream failed (%s): %s",
                        code,
                        exc,
                    )

                    raise _to_error(
                        exc,
                        code,
                        provider,
                    ) from exc

                delay = min(2**attempt, 8) + random.random() * 0.3

                await asyncio.sleep(delay)

    # ---------------------------------------------
    # Structured JSON generation
    # ---------------------------------------------

    async def ajson(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:

        instruction = (
            "\n\nRespond with ONLY one JSON object "
            "(no prose, no markdown fences) "
            "matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema())
        )

        messages = msgs(system + instruction, user)

        last_error = ""

        for attempt in range(2):

            raw = await self.ainvoke_text(messages)

            try:
                return schema.model_validate(
                    extract_json(raw)
                )

            except (ValueError, ValidationError) as exc:

                last_error = str(exc)[:300]

                messages = messages + [
                    AIMessage(content=raw[:3000]),
                    HumanMessage(
                        content=(
                            f"That was not valid for the schema "
                            f"({last_error}). "
                            "Reply again with ONLY the corrected "
                            "JSON object."
                        )
                    ),
                ]

        raise LLMOutputError(
            f"Model returned invalid structured output: {last_error}"
        )
