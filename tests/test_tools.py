from datetime import datetime, timedelta

import pytest

from tools import CHAT_TOOLS, calculator_tool, get_current_datetime


@pytest.mark.parametrize(
    ("operation", "num1", "num2", "expected_result"),
    [
        ("add", 8, 2, "Calculation result: 10.0"),
        ("subtract", 8, 2, "Calculation result: 6.0"),
        ("multiply", 8, 2, "Calculation result: 16.0"),
        ("divide", 8, 2, "Calculation result: 4.0"),
    ],
)
def test_calculator_tool_performs_basic_operations(
    operation: str,
    num1: float,
    num2: float,
    expected_result: str,
) -> None:
    result = calculator_tool.invoke(
        {
            "operation": operation,
            "num1": num1,
            "num2": num2,
        }
    )

    assert result == expected_result


def test_calculator_tool_handles_division_by_zero() -> None:
    result = calculator_tool.invoke(
        {
            "operation": "divide",
            "num1": 8,
            "num2": 0,
        }
    )

    assert result == "Cannot divide by zero"


def test_calculator_tool_is_available_to_the_chatbot() -> None:
    tool_names: list[str] = []

    for chat_tool in CHAT_TOOLS:
        tool_names.append(chat_tool.name)

    assert "calculator_tool" in tool_names


def test_get_current_datetime_defaults_to_asia_kolkata() -> None:
    result = get_current_datetime.invoke({})
    timestamp = result.removeprefix(
        "Current date and time in Asia/Kolkata: "
    )
    parsed_datetime = datetime.fromisoformat(timestamp)

    assert parsed_datetime.utcoffset() == timedelta(hours=5, minutes=30)


def test_get_current_datetime_returns_utc_time() -> None:
    result = get_current_datetime.invoke({"timezone": "UTC"})
    timestamp = result.removeprefix("Current date and time in UTC: ")
    parsed_datetime = datetime.fromisoformat(timestamp)

    assert parsed_datetime.utcoffset() == timedelta(0)


def test_get_current_datetime_returns_asia_kolkata_time() -> None:
    result = get_current_datetime.invoke({"timezone": "Asia/Kolkata"})
    timestamp = result.removeprefix(
        "Current date and time in Asia/Kolkata: "
    )
    parsed_datetime = datetime.fromisoformat(timestamp)

    assert parsed_datetime.utcoffset() == timedelta(hours=5, minutes=30)


def test_get_current_datetime_handles_unknown_timezone() -> None:
    result = get_current_datetime.invoke({"timezone": "Not/A-Timezone"})

    assert result == (
        "Unknown timezone: Not/A-Timezone. "
        "Use an IANA timezone such as Asia/Kolkata or UTC."
    )
