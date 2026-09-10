"""Load and validate environment configuration for the application."""

import os

from dotenv import load_dotenv

# Load values from .env once when this module is first imported.
load_dotenv()


def get_database_url() -> str:
    """Return the PostgreSQL connection string."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is required.")

    return database_url


def get_gemini_api_key() -> str:
    """Return the Gemini API key without displaying its value."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required.")

    return api_key
