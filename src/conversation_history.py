"""
    Read conversation messages saved by LangGraph.
    Because checkpoint table is managed by langGraph (it contains individual messages), we manage the Conversations Table    
"""

from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# Base class for thread config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow

from conversations_repository import conversation_exists


def delete_conversation_history(
    conversation_id: UUID,
    checkpointer: PostgresSaver,
) -> None:
    """Delete all LangGraph checkpoints for one conversation."""
    checkpointer.delete_thread(str(conversation_id))


def get_conversation_messages(
    conversation_id: UUID,
    database_connection: Connection[DictRow],
    checkpointer: PostgresSaver,
    debug: bool = False,
) -> list[dict[str, str]]:
    """Read the latest saved messages without calling the model."""
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
    formatted_messages: list[dict[str, str]] = []
    # We send role too in our structured output for frontend to render with diff graphics , IF FUTURE ME WANRS TO
    for message in saved_messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            # A tool request is an internal step, not a final assistant answer.
            if message.tool_calls or not message.text:
                continue
            role = "assistant"
        elif isinstance(message, ToolMessage):
            if not debug:
                continue
            role = "tool"
        else:
            continue

        formatted_message = {
            "role": role,
            "content": message.text,
        }
        formatted_messages.append(formatted_message)

    return formatted_messages
