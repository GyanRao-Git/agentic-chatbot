"""Small tools that the chatbot model is allowed to call."""

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


# The same list is used when binding tools to Gemini and building the ToolNode.
CHAT_TOOLS: list[BaseTool] = [get_current_datetime]
