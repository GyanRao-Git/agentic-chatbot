"""
    Runs one chatbot message with checkpointing
"""

import logging
import time
from uuid import UUID

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from psycopg import Connection
from psycopg.rows import DictRow
from psycopg_pool import ConnectionPool

from chatbot import ChatState
from config import DEFAULT_MODEL_NAME
from conversations_repository import conversation_exists
from exceptions import AgentExecutionError


logger = logging.getLogger(__name__)


def call_agent(
    conversation_id: UUID,
    user_input: str,
    database_pool: ConnectionPool[Connection[DictRow]],
    chatbot: CompiledStateGraph,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, str]:
    """Send one message and return structured data for the API."""
    # Borrow a connection only for this quick check. It returns to the pool
    # before the slow model call, so Gemini latency does not block other requests.
    with database_pool.connection() as connection:
        conversation_found = conversation_exists(
            connection=connection,
            conversation_id=conversation_id,
        )

    if not conversation_found:
        raise ValueError(f"Conversation does not exist: {conversation_id}")

    logger.info("agent turn started conversation_id=%s model=%s", conversation_id, model_name)
    started = time.perf_counter()

    # do not append, checkpointer will append, only send new message to avoid duplicates
    state: ChatState = {
        "messages": [HumanMessage(content=user_input)]
    }
    config: RunnableConfig = {
        "configurable": {
            "thread_id": str(conversation_id)
        }
    }
    try:
        final_state = chatbot.invoke(state, config=config)
        answer = final_state["messages"][-1].text
    except Exception as error:
        # Do not log the user's message or the third-party error text because
        # either could contain private information.
        logger.error(
            "Agent execution failed for conversation_id=%s; error_type=%s",
            conversation_id,
            type(error).__name__,
        )
        raise AgentExecutionError() from error

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "agent turn finished conversation_id=%s model=%s duration_ms=%.1f",
        conversation_id,
        model_name,
        duration_ms,
    )

    return {
        "conversation_id": str(conversation_id),
        "answer": answer,
    }
