from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from chatbot import TOOL_ERROR_MESSAGE, build_chatbot
from prompts import SYSTEM_PROMPT
from tools import CHAT_TOOLS


class FakeModel:
    def __init__(self) -> None:
        self.received_messages: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        self.received_messages.append(messages)
        return AIMessage(content=f"Model received {len(messages)} messages")


class ToolCallingFakeModel:
    """Request the clock once, then answer using its result."""

    def __init__(self) -> None:
        self.received_messages: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        self.received_messages.append(messages)

        if not isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_datetime",
                        "args": {"timezone": "Asia/Kolkata"},
                        "id": "clock-call",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content=f"Tool said: {messages[-1].text}")


class CalculatorCallingFakeModel:
    """Request multiplication once, then answer using its result."""

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        if not isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator_tool",
                        "args": {
                            "operation": "multiply",
                            "num1": 6,
                            "num2": 7,
                        },
                        "id": "calculator-call",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content=f"The answer is based on: {messages[-1].text}")


@tool
def failing_tool() -> str:
    """Raise an internal error so ToolNode error handling can be tested."""
    raise RuntimeError("private tool failure")


class FailingToolFakeModel:
    """Request the failing tool once, then explain its safe result."""

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        if not isinstance(messages[-1], ToolMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "failing_tool",
                        "args": {},
                        "id": "failing-tool-call",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content=f"Tool response: {messages[-1].text}")


# pytest is the program that finds and executes functions beginning with test_
def test_chatbot_accepts_a_fake_model() -> None:
    chatbot = build_chatbot(FakeModel(), MemorySaver(), CHAT_TOOLS)

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        {"configurable": {"thread_id": "test-thread"}},
    )

    assert result["messages"][-1].content == "Model received 2 messages"


def test_system_prompt_is_sent_once_but_not_checkpointed() -> None:
    model = FakeModel()
    chatbot = build_chatbot(model, MemorySaver(), CHAT_TOOLS)
    config = {"configurable": {"thread_id": "system-prompt-thread"}}

    chatbot.invoke(
        {"messages": [HumanMessage(content="Hello")]},
        config,
    )

    messages_sent_to_model = model.received_messages[0]
    assert isinstance(messages_sent_to_model[0], SystemMessage)
    assert messages_sent_to_model[0].content == SYSTEM_PROMPT

    system_message_count = 0
    for message in messages_sent_to_model:
        if isinstance(message, SystemMessage):
            system_message_count += 1

    assert system_message_count == 1

    saved_messages = chatbot.get_state(config).values["messages"]
    for message in saved_messages:
        assert not isinstance(message, SystemMessage)


def test_chatbot_restores_messages_for_the_same_thread() -> None:
    chatbot = build_chatbot(FakeModel(), MemorySaver(), CHAT_TOOLS)
    config = {"configurable": {"thread_id": "same-thread"}}

    chatbot.invoke(
        {"messages": [HumanMessage(content="First message")]},
        config,
    )
    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Second message")]},
        config,
    )

    assert [message.content for message in result["messages"]] == [
        "First message",
        "Model received 2 messages",
        "Second message",
        "Model received 4 messages",
    ]


def test_chatbot_keeps_different_threads_isolated() -> None:
    chatbot = build_chatbot(FakeModel(), MemorySaver(), CHAT_TOOLS)

    chatbot.invoke(
        {"messages": [HumanMessage(content="Message for thread A")]},
        {"configurable": {"thread_id": "thread-a"}},
    )
    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Message for thread B")]},
        {"configurable": {"thread_id": "thread-b"}},
    )

    assert [message.content for message in result["messages"]] == [
        "Message for thread B",
        "Model received 2 messages",
    ]


def test_chatbot_executes_tool_and_returns_to_model() -> None:
    model = ToolCallingFakeModel()
    chatbot = build_chatbot(model, MemorySaver(), CHAT_TOOLS)

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="What time is it in India?")]},
        {"configurable": {"thread_id": "tool-thread"}},
    )

    messages = result["messages"]
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "get_current_datetime"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].text.startswith("Current date and time in Asia/Kolkata:")
    assert messages[3].text.startswith("Tool said: Current date and time")

    # The model runs before and after the tool, receiving one system prompt each time.
    assert len(model.received_messages) == 2
    for model_call_messages in model.received_messages:
        system_message_count = 0
        for message in model_call_messages:
            if isinstance(message, SystemMessage):
                system_message_count += 1

        assert system_message_count == 1


def test_chatbot_executes_calculator_and_returns_to_model() -> None:
    chatbot = build_chatbot(
        CalculatorCallingFakeModel(),
        MemorySaver(),
        CHAT_TOOLS,
    )

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="What is 6 multiplied by 7?")]},
        {"configurable": {"thread_id": "calculator-thread"}},
    )

    messages = result["messages"]
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "calculator_tool"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].text == "Calculation result: 42.0"
    assert messages[3].text == "The answer is based on: Calculation result: 42.0"


def test_chatbot_converts_unexpected_tool_failure_into_safe_result() -> None:
    chatbot = build_chatbot(
        FailingToolFakeModel(),
        MemorySaver(),
        [failing_tool],
    )

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Run the failing tool")]},
        {"configurable": {"thread_id": "failing-tool-thread"}},
    )

    messages = result["messages"]
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].status == "error"
    assert messages[2].text == TOOL_ERROR_MESSAGE
    assert "private tool failure" not in messages[2].text
    assert messages[3].text == f"Tool response: {TOOL_ERROR_MESSAGE}"
