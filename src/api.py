"""FastAPI chatbot server"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response, status
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from agents import call_agent
from chatbot import build_chatbot
from config import get_database_url
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

"""
    @asynccontextmanager
    async def lifespan(app):
        print("STARTUP")
        yield
        print("SHUTDOWN")
"""
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources at startup and close them at shutdown."""
    database_url = get_database_url()

    # A pool keeps database connections ready and lends one to each request.
    database_pool: ConnectionPool[Connection[DictRow]] = ConnectionPool(
        conninfo=database_url,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=1,
        max_size=10,
        open=False,
    )

    with database_pool:
        checkpointer = PostgresSaver(database_pool)
        model = create_chat_model(provider="gemini")

        # bind_tools is a method of BaseChatModel, It gives Gemini their names, descriptions, and input structure.
        model_with_tools = model.bind_tools(CHAT_TOOLS)
        chatbot = build_chatbot(
            model=model_with_tools,
            checkpointer=checkpointer,
            tools=CHAT_TOOLS,
        )

        # Routes retrieve these shared objects through request.app.state.
        app.state.database_pool = database_pool
        app.state.checkpointer = checkpointer
        app.state.chatbot = chatbot

        yield

# FastAPI app that Uvicorn will run.
app = FastAPI(
    title="Agentic Chatbot API",
    lifespan=lifespan,
)


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

    return ConversationResponse(
        conversation_id=conversation_id,
        title=title,
    )


# Returns every conversation, with the newest one first.
@app.get(
    "/conversations",
    response_model=list[ConversationResponse],
)
def get_conversations(request: Request) -> list[ConversationResponse]:
    with request.app.state.database_pool.connection() as connection:
        conversation_rows = list_conversations(connection)

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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation does not exist: {conversation_id}",
            )

        update_conversation_title(
            connection=connection,
            conversation_id=conversation_id,
            title=conversation_request.title,
        )

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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation does not exist: {conversation_id}",
            )

        delete_conversation_history(
            conversation_id=conversation_id,
            checkpointer=request.app.state.checkpointer,
        )
        delete_conversation(connection, conversation_id)

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
) -> ConversationMessagesResponse:
    with request.app.state.database_pool.connection() as connection:
        try:
            saved_messages = get_conversation_messages(
                conversation_id=conversation_id,
                database_connection=connection,
                checkpointer=request.app.state.checkpointer,
                debug=debug,
            )
        except ValueError as error:
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
    with request.app.state.database_pool.connection() as connection:
        try:
            agent_response = call_agent(
                conversation_id=conversation_id,
                user_input=message_request.message,
                database_connection=connection,
                chatbot=request.app.state.chatbot,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    return MessageResponse(
        conversation_id=agent_response["conversation_id"],
        answer=agent_response["answer"],
    )
