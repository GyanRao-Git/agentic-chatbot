"""
    LangGraph chatbot construction and state definitions.
"""

from typing import Annotated, Protocol, TypedDict

from langchain_core.messages import BaseMessage
# Base class for all LangChain tools
from langchain_core.tools import BaseTool

# base class for checkpointer type
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

"""
    LLM
    ↓
    AIMessage with tool_calls
    ↓
    ToolNode
    ├── find the requested tool
    ├── extract arguments
    ├── call the tool
    ├── handle multiple tool calls
    └── create ToolMessage with results
    ↓
    LLM again

    without toolNode we would have to manage this ourselves

    AND tools_condition provides that routing logic.
"""
from langgraph.prebuilt import ToolNode, tools_condition

# protocol tells typechecker if the object class passed has invoke function or not
class ChatModel(Protocol):
    """The small model interface the graph needs, including test doubles."""

    def invoke(self, messages: list[BaseMessage] , /) -> BaseMessage:
        """Return a model response for the supplied conversation messages."""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_chatbot(
    model: ChatModel,
    checkpointer: BaseCheckpointSaver,
    tools: list[BaseTool],
):
    """Build a chatbot graph using the supplied model and tools."""
    graph = StateGraph(ChatState)

    def chat_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat_node")

    # Tool calls go to ToolNode. Normal model responses end the graph.
    graph.add_conditional_edges(
        "chat_node",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)
