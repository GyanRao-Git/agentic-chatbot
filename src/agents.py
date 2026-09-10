"""
    Runs one chatbot message with checkpointing
"""

from uuid import UUID

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from psycopg import Connection
from psycopg.rows import DictRow

from chatbot import ChatState
from conversations_repository import conversation_exists


def call_agent(
    conversation_id: UUID,
    user_input: str,
    database_connection: Connection[DictRow],
    chatbot: CompiledStateGraph,
) -> dict[str, str]:
    """Send one message and return structured data for the API."""
    if not conversation_exists(
        connection=database_connection,
        conversation_id=conversation_id,
    ):
        raise ValueError(f"Conversation does not exist: {conversation_id}")

    # do not append, checkpointer will append, only send new message to avoid duplicates
    state: ChatState = {
        "messages": [HumanMessage(content=user_input)]
    }
    config: RunnableConfig = {
        "configurable": {
            "thread_id": str(conversation_id)
        }
    }
    final_state = chatbot.invoke(state, config=config)

    return {
        "conversation_id": str(conversation_id),
        "answer": final_state["messages"][-1].text,
    }
