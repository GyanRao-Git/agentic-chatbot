"""
    Read conversation messages saved by LangGraph.
    Because checkpoint table is managed by langGraph (it contains individual messages), we manage the Conversations Table    
"""

from uuid import UUID

import psycopg
from langchain_core.messages import BaseMessage

# Base class for thread config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver

from config import get_database_url
from conversations_repository import conversation_exists


def delete_conversation_history(conversation_id: UUID) -> None:
    """Delete all LangGraph checkpoints for one conversation."""
    database_url = get_database_url()

    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.delete_thread(str(conversation_id))


def get_conversation_messages(conversation_id: UUID) -> list[dict[str, str]]:
    """Read the latest saved messages without calling the model."""
    database_url = get_database_url()

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
