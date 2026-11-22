"""Construction of production chat-model clients."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import ALLOWED_MODEL_NAMES, DEFAULT_MODEL_NAME, get_gemini_api_key

logger = logging.getLogger(__name__)


def create_chat_model(
    provider: str = "gemini",
    model_name: str = DEFAULT_MODEL_NAME,
) -> BaseChatModel:
    """Create the configured chat model for the requested provider."""
    if provider != "gemini":
        logger.error("unsupported model provider=%s", provider)
        raise ValueError(f"Unsupported model provider: {provider}")

    if model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")

    api_key = get_gemini_api_key()
    logger.info("chat model created provider=%s model=%s", provider, model_name)

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
    )
