"""
    Runs one chatbot message with checkpointing
"""

from uuid import UUID

import psycopg
from langchain_core.messages import HumanMessage

# Persistent state using PostgreSQL checkpointing
from langgraph.checkpoint.postgres import PostgresSaver

from chatbot import ChatState, build_chatbot
from config import get_database_url
from conversations_repository import conversation_exists
from model_factory import create_chat_model


def call_agent(conversation_id: UUID, user_input: str) -> dict[str, str]:
    """Send one message and return structured data for the API."""
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

        model = create_chat_model(provider="gemini")
        chatbot = build_chatbot(
            model=model,
            checkpointer=checkpointer,
        )

        # do not append, checkpointer will append, only send new message to avoid duplicates
        state: ChatState = {
            "messages": [HumanMessage(content=user_input)]
        }
        config = {
            "configurable": {
                "thread_id": str(conversation_id)
            }
        }
        final_state = chatbot.invoke(state, config=config)

    return {
        "conversation_id": str(conversation_id),
        "answer": final_state["messages"][-1].text,
    }
