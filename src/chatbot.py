"""
    LangGraph chatbot construction and state definitions.
"""

from typing import Annotated, Protocol, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# protocol tells typechecker if the object class passed has invoke function or not
class ChatModel(Protocol):
    """The small model interface the graph needs, including test doubles."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """Return a model response for the supplied conversation messages."""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_chatbot(model: ChatModel):
    """Build a chatbot graph using the supplied model implementation."""
    graph = StateGraph(ChatState)

    def chat_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=MemorySaver())
