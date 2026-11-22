
from __future__ import annotations

import logging
import time
import uuid

from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import (
    analytics,
    auth,
    chat,
    documents,
    health,
    interviews,
    memory,
    resume,
)

from app.config import get_settings
from app.core.logging import setup_logging
from app.core.rate_limit import limiter
from app.core.tracing import configure_tracing
from app.database import mongo
from app.database.indexes import ensure_indexes
from app.errors import AppError
from app.graph.checkpointer import create_checkpointer
from app.services.container import get_services
from app.services.interview_service import InterviewService


log = logging.getLogger("app")


# --------------------------------------------------
# APPLICATION FACTORY
# --------------------------------------------------

def create_app() -> FastAPI:

    settings = get_settings()

    setup_logging(settings.log_level)

    # --------------------------------------------------
    # MCP CONFIGURATION
    # --------------------------------------------------

    mcp_apps: dict = {}

    if settings.mcp_mode == "embedded":

        from app.mcp.servers import SERVERS
        from app.mcp.servers._common import http_app

        mcp_apps = {
            name: http_app(
                server,
                settings.mcp_internal_token,
            )
            for name, server in SERVERS.items()
        }

    # --------------------------------------------------
    # LLM MODEL INFORMATION
    # --------------------------------------------------

    provider = settings.llm_provider.lower().strip()

    model_names = {
        "gemini": settings.gemini_model,
        "ollama": settings.ollama_model,
        "huggingface": settings.llm_model_id,
    }

    model_name = model_names.get(provider, "unknown")

    # --------------------------------------------------
    # APPLICATION LIFESPAN
    # --------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        log.info("Starting RAG Interview Preparation API...")

        # Validate production configuration.
        settings.validate_for_runtime()

        log.info(
            "LLM provider: %s",
            provider,
        )

        log.info(
            "LLM model: %s",
            model_name,
        )

        log.info(
            "LLM API key configured: %s",
            bool(settings.gemini_api_key.strip())
            if provider == "gemini"
            else bool(settings.hf_token.strip())
            if provider == "huggingface"
            else True,
        )

        # Configure LangSmith tracing.
        configure_tracing(settings)

        # Ensure MongoDB indexes.
        await ensure_indexes(mongo.get_db())

        # Initialize application services.
        services = get_services()

        # Initialize interview service and checkpointer.
        services.interviews = InterviewService(
            services,
            create_checkpointer(settings),
        )

        # Start embedded MCP servers.
        async with AsyncExitStack() as stack:

            for sub in mcp_apps.values():

                await stack.enter_async_context(
                    sub.lifespan(sub)
                )

            log.info(
                "RAG Interview Preparation API ready "
                "(mcp_mode=%s, provider=%s, model=%s)",
                settings.mcp_mode,
                provider,
                model_name,
            )

            yield

        # Close MongoDB connection.
        await mongo.close()

        log.info("RAG Interview Preparation API stopped.")

    # --------------------------------------------------
    # FASTAPI APPLICATION
    # --------------------------------------------------

    app = FastAPI(
        title="RAG-Based Personalized Interview Preparation System",
        description=(
            "AI-powered personalized interview preparation "
            "using Retrieval-Augmented Generation, "
            "LangGraph, MongoDB and vector search."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # --------------------------------------------------
    # RATE LIMITING
    # --------------------------------------------------

    app.state.limiter = limiter

    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,
    )

    app.add_middleware(
        SlowAPIMiddleware,
    )

    # --------------------------------------------------
    # CORS CONFIGURATION
    # --------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------
    # REQUEST LOGGING MIDDLEWARE
    # --------------------------------------------------

    @app.middleware("http")
    async def request_log(
        request: Request,
        call_next,
    ):

        request_id = uuid.uuid4().hex[:8]

        start = time.perf_counter()

        response = await call_next(request)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        log.info(
            "%s %s -> %s (%.0f ms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id

        return response

    # --------------------------------------------------
    # APPLICATION ERROR HANDLER
    # --------------------------------------------------

    @app.exception_handler(AppError)
    async def app_error(
        request: Request,
        exc: AppError,
    ):

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "code": exc.code,
            },
        )

    # --------------------------------------------------
    # GLOBAL ERROR HANDLER
    # --------------------------------------------------

    @app.exception_handler(Exception)
    async def unhandled(
        request: Request,
        exc: Exception,
    ):

        log.exception(
            "Unhandled error on %s %s",
            request.method,
            request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "code": "internal_error",
            },
        )

    # --------------------------------------------------
    # REGISTER API ROUTERS
    # --------------------------------------------------

    for router in (
        health.router,
        auth.router,
        resume.router,
        documents.router,
        interviews.router,
        chat.router,
        memory.router,
        analytics.router,
    ):

        app.include_router(router)

    # --------------------------------------------------
    # MOUNT EMBEDDED MCP SERVERS
    # --------------------------------------------------

    for name, sub in mcp_apps.items():

        app.mount(
            f"/mcp-servers/{name}",
            sub,
        )

    return app


# --------------------------------------------------
# APPLICATION INSTANCE
# --------------------------------------------------

app = create_app()
