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


# Defaults used when the pool size variables are not set.
DEFAULT_DB_POOL_MIN_SIZE = 1
DEFAULT_DB_POOL_MAX_SIZE = 10


def _read_positive_int_env(name: str, default: int) -> int:
    """Read a whole number from the environment, falling back to a default."""
    raw_value = os.getenv(name)

    # An unset or blank variable means "use the default".
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a whole number.") from error

    if value < 1:
        raise RuntimeError(f"{name} must be at least 1.")

    return value


def get_database_pool_limits() -> tuple[int, int]:
    """Return (min_size, max_size) for the PostgreSQL connection pool."""
    min_size = _read_positive_int_env(
        "DB_POOL_MIN_SIZE",
        DEFAULT_DB_POOL_MIN_SIZE,
    )
    max_size = _read_positive_int_env(
        "DB_POOL_MAX_SIZE",
        DEFAULT_DB_POOL_MAX_SIZE,
    )

    if max_size < min_size:
        raise RuntimeError(
            "DB_POOL_MAX_SIZE must be greater than or equal to DB_POOL_MIN_SIZE."
        )

    return min_size, max_size
