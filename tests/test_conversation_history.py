from unittest.mock import Mock, patch
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation_history import (
    delete_conversation_history,
    get_conversation_messages,
)


def test_delete_conversation_history_deletes_langgraph_thread() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    checkpointer = Mock()

    delete_conversation_history(
        conversation_id=conversation_id,
        checkpointer=checkpointer,
    )

    checkpointer.delete_thread.assert_called_once_with(str(conversation_id))


def test_get_conversation_messages_reads_latest_checkpoint() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_connection = Mock()
    checkpointer = Mock()
    checkpointer.get.return_value = {
        "channel_values": {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_current_datetime",
                            "args": {"timezone": "UTC"},
                            "id": "clock-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Current date and time in UTC: 2026-09-11T00:00:00+00:00",
                    tool_call_id="clock-call",
                    name="get_current_datetime",
                ),
                AIMessage(content="It is midnight in UTC."),
            ]
        }
    }

    with patch(
        "conversation_history.conversation_exists",
        return_value=True,
    ):
        result = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
        )

    assert result == (
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "It is midnight in UTC."},
        ],
        None,
    )
    checkpointer.get.assert_called_once_with(
        {"configurable": {"thread_id": str(conversation_id)}}
    )


def test_get_conversation_messages_returns_empty_for_new_conversation() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_connection = Mock()
    checkpointer = Mock()
    checkpointer.get.return_value = None

    with patch(
        "conversation_history.conversation_exists",
        return_value=True,
    ):
        result = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
        )

    assert result == ([], None)


def test_get_conversation_messages_includes_tool_output_in_debug_mode() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_connection = Mock()
    checkpointer = Mock()
    checkpointer.get.return_value = {
        "channel_values": {
            "messages": [
                HumanMessage(content="What time is it?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_current_datetime",
                            "args": {"timezone": "UTC"},
                            "id": "clock-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Current date and time in UTC: 2026-09-11T00:00:00+00:00",
                    tool_call_id="clock-call",
                    name="get_current_datetime",
                ),
                AIMessage(content="It is midnight in UTC."),
            ]
        }
    }

    with patch(
        "conversation_history.conversation_exists",
        return_value=True,
    ):
        result = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
            debug=True,
        )

    assert result == (
        [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "tool",
                "content": "Current date and time in UTC: 2026-09-11T00:00:00+00:00",
            },
            {"role": "assistant", "content": "It is midnight in UTC."},
        ],
        None,
    )


def test_get_conversation_messages_returns_newest_page_in_chronological_order() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_connection = Mock()
    checkpointer = Mock()
    checkpointer.get.return_value = {
        "channel_values": {
            "messages": [
                HumanMessage(content="Question one"),
                AIMessage(content="Answer one"),
                HumanMessage(content="Question two"),
                AIMessage(content="Answer two"),
                HumanMessage(content="Question three"),
                AIMessage(content="Answer three"),
            ]
        }
    }

    with patch("conversation_history.conversation_exists", return_value=True):
        newest_messages, next_before = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
            limit=2,
        )
        older_messages, older_next_before = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
            limit=2,
            before=next_before,
        )

    assert newest_messages == [
        {"role": "user", "content": "Question three"},
        {"role": "assistant", "content": "Answer three"},
    ]
    assert next_before == 4
    assert older_messages == [
        {"role": "user", "content": "Question two"},
        {"role": "assistant", "content": "Answer two"},
    ]
    assert older_next_before == 2


def test_message_pagination_counts_only_messages_visible_in_selected_mode() -> None:
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    database_connection = Mock()
    checkpointer = Mock()
    checkpointer.get.return_value = {
        "channel_values": {
            "messages": [
                HumanMessage(content="What time is it?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_current_datetime",
                            "args": {"timezone": "UTC"},
                            "id": "clock-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="Clock result",
                    tool_call_id="clock-call",
                    name="get_current_datetime",
                ),
                AIMessage(content="It is midnight."),
            ]
        }
    }

    with patch("conversation_history.conversation_exists", return_value=True):
        normal_messages, normal_next_before = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
            limit=1,
        )
        debug_messages, debug_next_before = get_conversation_messages(
            conversation_id=conversation_id,
            database_connection=database_connection,
            checkpointer=checkpointer,
            debug=True,
            limit=2,
        )

    assert normal_messages == [
        {"role": "assistant", "content": "It is midnight."}
    ]
    assert normal_next_before == 1
    assert debug_messages == [
        {"role": "tool", "content": "Clock result"},
        {"role": "assistant", "content": "It is midnight."},
    ]
    assert debug_next_before == 1
