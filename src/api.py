"""FastAPI chatbot server"""

import os
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from agents import call_agent
from conversations_repository import create_conversation
from models import ConversationResponse, MessageRequest, MessageResponse

load_dotenv()


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
def start_conversation() -> ConversationResponse:
    database_url = os.environ["DATABASE_URL"]

    # This connection exists only while the request is being handled.
    with psycopg.connect(database_url, autocommit=True) as connection:
        conversation_id = create_conversation(connection)

    return ConversationResponse(conversation_id=conversation_id)


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
