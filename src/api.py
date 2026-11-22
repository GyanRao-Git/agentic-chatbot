"""FastAPI chatbot server"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from agents import call_agent
from chatbot import build_chatbot
from config import (
    ALLOWED_MODEL_NAMES,
    DEFAULT_MODEL_NAME,
    get_database_pool_limits,
    get_database_url,
    get_log_retention_days,
    get_log_user_text,
)
from conversation_history import (
    delete_conversation_history,
    get_conversation_messages,
)
from conversations_repository import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    list_conversations,
    update_conversation_title,
)
from exceptions import AgentExecutionError
from logging_config import configure_logging
from model_factory import create_chat_model
from models import (
    ConversationMessage,
    ConversationMessagesResponse,
    ConversationResponse,
    ConversationTitleRequest,
    MessageRequest,
    MessageResponse,
)
from tools import CHAT_TOOLS

# Each module gets its own logger; the name shows which module wrote the line.
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources at startup and close them at shutdown."""
    # Logging is set up first, so every startup message below is recorded.
    configure_logging(retention_days=get_log_retention_days())

    # Read settings first. If one is missing, log it, then stop startup.
    try:
        database_url = get_database_url()
        # Pool sizes come from DB_POOL_MIN_SIZE and DB_POOL_MAX_SIZE, with defaults.
        min_size, max_size = get_database_pool_limits()
    except RuntimeError:
        logger.exception("startup failed, check environment variables")
        raise

    # A pool keeps database connections ready and lends one to each request.
    database_pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        conninfo=database_url,
        # kwargs are settings applied to each new connection pool creates
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=min_size,
        max_size=max_size,
        open=False,
    )

    with database_pool:
        checkpointer = PostgresSaver(database_pool)
        # Build one graph per allowed model. Each request picks one by name.
        chatbots: dict[str, CompiledStateGraph] = {}

        for model_name in ALLOWED_MODEL_NAMES:
            model = create_chat_model(provider="gemini", model_name=model_name)

            # bind_tools is a method of BaseChatModel, It gives Gemini their names, descriptions, and input structure.
            model_with_tools = model.bind_tools(CHAT_TOOLS)
            chatbots[model_name] = build_chatbot(
                model=model_with_tools,
                checkpointer=checkpointer,
                tools=CHAT_TOOLS,
            )

        # Routes retrieve these shared objects through request.app.state.
        app.state.database_pool = database_pool
        app.state.checkpointer = checkpointer
        app.state.chatbots = chatbots
        logger.info("database pool ready min_size=%s max_size=%s", min_size, max_size)

        yield

        logger.info("shutdown started, closing database pool")

# FastAPI app that Uvicorn will run.
app = FastAPI(
    title="Agentic Chatbot API",
    lifespan=lifespan,
)


# Middleware runs around every request. It logs once per request, after the route finishes.
@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log method, path, status code and time taken for each request."""
    # perf_counter is a precise clock, good for measuring how long something takes.
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000

    # The query string is not logged, so user text is kept out of the file.
    logger.info(
        "request finished method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# Visiting GET /health confirms that the server is running.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Creates a conversation and returns its PostgreSQL-generated UUIDv7.
@app.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_conversation(
    request: Request,
    conversation_request: ConversationTitleRequest | None = None,
) -> ConversationResponse:
    title = None

    if conversation_request is not None:
        title = conversation_request.title

    # The connection returns to the shared pool when this block ends.
    with request.app.state.database_pool.connection() as connection:
        conversation_id = create_conversation(connection, title)

    # The title text is not logged, only the ID and whether a title was given.
    logger.info(
        "conversation created conversation_id=%s has_title=%s",
        conversation_id,
        title is not None,
    )

    return ConversationResponse(
        conversation_id=conversation_id,
        title=title,
    )


# Returns every conversation, with the newest one first.
@app.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
def get_conversations(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[ConversationResponse]:
    with request.app.state.database_pool.connection() as connection:
        conversation_rows = list_conversations(
            connection,
            limit=limit,
            offset=offset,
        )

    conversations: list[ConversationResponse] = []

    for conversation_id, title in conversation_rows:
        conversation = ConversationResponse(
            conversation_id=conversation_id,
            title=title,
        )
        conversations.append(conversation)

    return conversations


# Updates or clears the title displayed by a frontend.
@app.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def rename_conversation(
    conversation_id: UUID,
    conversation_request: ConversationTitleRequest,
    request: Request,
) -> ConversationResponse:
    with request.app.state.database_pool.connection() as connection:
        if not conversation_exists(connection, conversation_id):
            logger.warning("rename failed, not found conversation_id=%s", conversation_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation does not exist: {conversation_id}",
            )

        update_conversation_title(
            connection=connection,
            conversation_id=conversation_id,
            title=conversation_request.title,
        )

    logger.info("conversation renamed conversation_id=%s", conversation_id)

    return ConversationResponse(
        conversation_id=conversation_id,
        title=conversation_request.title,
    )


# Deletes both the saved LangGraph history and our conversation row.
@app.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_conversation(conversation_id: UUID, request: Request) -> Response:
    with request.app.state.database_pool.connection() as connection:
        if not conversation_exists(connection, conversation_id):
            logger.warning("delete failed, not found conversation_id=%s", conversation_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation does not exist: {conversation_id}",
            )

        delete_conversation_history(
            conversation_id=conversation_id,
            checkpointer=request.app.state.checkpointer,
        )
        delete_conversation(connection, conversation_id)

    logger.info("conversation deleted conversation_id=%s", conversation_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Returns messages already saved by LangGraph for one conversation.
@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
def get_messages(
    conversation_id: UUID,
    request: Request,
    debug: bool = False,
    limit: int = Query(default=50, ge=1, le=100),
    before: int | None = Query(default=None, ge=0),
) -> ConversationMessagesResponse:
    with request.app.state.database_pool.connection() as connection:
        try:
            saved_messages, next_before = get_conversation_messages(
                conversation_id=conversation_id,
                database_connection=connection,
                checkpointer=request.app.state.checkpointer,
                debug=debug,
                limit=limit,
                before=before,
            )
        except ValueError as error:
            logger.warning("messages requested for missing conversation_id=%s", conversation_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    conversation_messages: list[ConversationMessage] = []

    for message in saved_messages:
        conversation_message = ConversationMessage(
            role=message["role"],
            content=message["content"],
        )
        conversation_messages.append(conversation_message)

    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=conversation_messages,
        next_before=next_before,
    )


# Sends one message to an existing conversation.
@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
)
def chat(
    conversation_id: UUID,
    message_request: MessageRequest,
    request: Request,
) -> MessageResponse:
    # Off by default. Turn on LOG_USER_TEXT=true only for local debugging.
    if get_log_user_text():
        logger.info(
            "user message conversation_id=%s text=%s",
            conversation_id,
            message_request.message,
        )

    # call_agent borrows its own short-lived connection, so the Gemini call
    # below does not keep a pooled connection checked out.
    # The request may name a model. If it does not, use the default model.
    model_name = message_request.model or DEFAULT_MODEL_NAME
    chatbot = request.app.state.chatbots[model_name]

    try:
        agent_response = call_agent(
            conversation_id=conversation_id,
            user_input=message_request.message,
            database_pool=request.app.state.database_pool,
            chatbot=chatbot,
            model_name=model_name,
        )
    except ValueError as error:
        logger.warning("message rejected, not found conversation_id=%s", conversation_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except AgentExecutionError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return MessageResponse(
        conversation_id=agent_response["conversation_id"],
        answer=agent_response["answer"],
    )
