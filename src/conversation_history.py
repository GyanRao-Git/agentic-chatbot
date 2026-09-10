"""
    Read conversation messages saved by LangGraph.
    Because checkpoint table is managed by langGraph (it contains individual messages), we manage the Conversations Table    
"""

import os
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

# Base class for thread config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver

from conversations_repository import conversation_exists


def get_conversation_messages(conversation_id: UUID) -> list[dict[str, str]]:
    """Read the latest saved messages without calling the model."""
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]

    with (
        PostgresSaver.from_conn_string(database_url) as checkpointer,
        psycopg.connect(database_url, autocommit=True) as database_connection,
    ):
        if not conversation_exists(
            connection=database_connection,
            conversation_id=conversation_id,
        ):
            raise ValueError(f"Conversation does not exist: {conversation_id}")

        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(conversation_id),
            }
        }
        checkpoint = checkpointer.get(config)

    # A new conversation has no checkpoint until its first message is sent.
    if checkpoint is None:
        return []

    saved_messages: list[BaseMessage] = checkpoint["channel_values"].get(
        "messages",
        [],
    )
    roles = {
        "human": "user",
        "ai": "assistant",
    }

    formatted_messages: list[dict[str, str]] = []

    for message in saved_messages:
        formatted_message = {
            "role": roles.get(message.type, message.type),
            "content": message.text,
        }
        formatted_messages.append(formatted_message)

    return formatted_messages
