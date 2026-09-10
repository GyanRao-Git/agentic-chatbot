"""Construction of production chat-model clients."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_gemini_api_key


def create_chat_model(provider: str = "gemini") -> BaseChatModel:
    """Create the configured chat model for the requested provider."""
    if provider != "gemini":
        raise ValueError(f"Unsupported model provider: {provider}")

    api_key = get_gemini_api_key()

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
    )
