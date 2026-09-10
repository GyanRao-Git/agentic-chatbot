"""FastAPI chatbot server"""

from uuid import UUID

import psycopg
from fastapi import FastAPI, HTTPException, Response, status

from agents import call_agent
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
from models import (
    ConversationMessage,
    ConversationMessagesResponse,
    ConversationResponse,
    ConversationTitleRequest,
    MessageRequest,
    MessageResponse,
)

# FastAPI app that Uvicorn will run.
app = FastAPI(title="Agentic Chatbot API")


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
    conversation_request: ConversationTitleRequest | None = None,
) -> ConversationResponse:
    database_url = get_database_url()
    title = None

    if conversation_request is not None:
        title = conversation_request.title

    # This connection exists only while the request is being handled.
    with psycopg.connect(database_url, autocommit=True) as connection:
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
def get_conversations() -> list[ConversationResponse]:
    database_url = get_database_url()

    with psycopg.connect(database_url, autocommit=True) as connection:
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
) -> ConversationResponse:
    database_url = get_database_url()

    with psycopg.connect(database_url, autocommit=True) as connection:
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
def remove_conversation(conversation_id: UUID) -> Response:
    database_url = get_database_url()

    with psycopg.connect(database_url, autocommit=True) as connection:
        if not conversation_exists(connection, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation does not exist: {conversation_id}",
            )

        delete_conversation_history(conversation_id)
        delete_conversation(connection, conversation_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Returns messages already saved by LangGraph for one conversation.
@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
def get_messages(conversation_id: UUID) -> ConversationMessagesResponse:
    try:
        saved_messages = get_conversation_messages(conversation_id)
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
) -> MessageResponse:
    try:
        agent_response = call_agent(
            conversation_id=conversation_id,
            user_input=message_request.message,
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
