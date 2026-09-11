"""Small tools that the chatbot model is allowed to call."""

from typing import Literal

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool


@tool
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """
        Return the live date and time for an IANA timezone.

        Infer the IANA timezone from the location in the user's message.
        For example, use Europe/London for London and America/New_York for New York.
    """
    try:
        selected_timezone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return (
            f"Unknown timezone: {timezone}. "
            "Use an IANA timezone such as Asia/Kolkata or UTC."
        )

    current_datetime = datetime.now(selected_timezone)
    formatted_datetime = current_datetime.isoformat(timespec="seconds")

    return f"Current date and time in {timezone}: {formatted_datetime}"


@tool
def calculator_tool(operation: Literal["add", "subtract", "multiply", "divide"], num1: float, num2: float) -> str:
    """
        Perform an arithmetic operation on two numbers.

        Args:
            operation: The arithmetic operation to perform. Supported operations
                are "add", "subtract", "multiply", and "divide".
            num1: The first number.
            num2: The second number.

        Returns:
            The result of the arithmetic operation as an int or float,
            or an error message as a string if the operation is unsupported.

        Raises:
            ZeroDivisionError: If the operation is "divide" and num2 is zero.
    """
    op: str = operation.strip().lower()
    
    if op == "add":
        res = num1 + num2
    elif op == "subtract":
        res =  num1 - num2
    elif op == "multiply":
        res =  num1*num2
    elif op == "divide":
        if num2 == 0:
            return "Cannot divide by zero"
        res =  num1/num2
    else:
        return "Unsupported operation"

    return f"Calculation result: {res}"


# The same list is used when binding tools to Gemini and building the ToolNode.
CHAT_TOOLS: list[BaseTool] = [
    get_current_datetime,
    calculator_tool
]
