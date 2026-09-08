"""Construction of production chat-model clients."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def create_chat_model(provider: str = "gemini") -> BaseChatModel:
    """Create the configured chat model for the requested provider."""
    if provider != "gemini":
        raise ValueError(f"Unsupported model provider: {provider}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to use the Gemini model.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
    )
