
from __future__ import annotations

import os

from fastapi import APIRouter

from app.config import get_settings
from app.database import mongo as m
from app.services.llm_service import LLMService


router = APIRouter(tags=["health"])


@router.get("/health")
async def health():

    settings = get_settings()

    # -----------------------------------------
    # MongoDB health check
    # -----------------------------------------

    try:
        await m.get_db().command("ping")
        mongodb_status = "ok"

    except Exception as exc:
        mongodb_status = f"error: {type(exc).__name__}"

    # -----------------------------------------
    # LLM configuration check
    # -----------------------------------------

    provider = settings.llm_provider.lower().strip()

    model_names = {
        "gemini": settings.gemini_model,
        "ollama": settings.ollama_model,
        "huggingface": settings.llm_model_id,
    }

    model_name = model_names.get(provider, "unknown")

    try:
        llm_service = LLMService(settings=settings)
        llm_configured = llm_service.configured

    except Exception:
        llm_configured = False

    # -----------------------------------------
    # LangSmith tracing
    # -----------------------------------------

    langsmith_enabled = (
        os.environ.get("LANGSMITH_TRACING", "").lower()
        == "true"
    )

    # -----------------------------------------
    # Overall application status
    # -----------------------------------------

    application_status = (
        "ok"
        if mongodb_status == "ok" and llm_configured
        else "degraded"
    )

    # -----------------------------------------
    # Health response
    # -----------------------------------------

    return {
        "status": application_status,
        "mongodb": mongodb_status,
        "llm_provider": provider,
        "model": model_name,
        "llm_configured": llm_configured,
        "vector_store": settings.vector_store,
        "reranker_enabled": settings.reranker_enabled,
        "mcp_mode": settings.mcp_mode,
        "checkpointer": settings.checkpointer,
        "langsmith_tracing": langsmith_enabled,
    }
