import os
from unittest.mock import patch

import pytest

from config import get_database_pool_limits, get_database_url, get_gemini_api_key


def test_get_database_url_requires_environment_variable() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(RuntimeError, match="DATABASE_URL is required"),
    ):
        get_database_url()


def test_get_gemini_api_key_requires_environment_variable() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(RuntimeError, match="GEMINI_API_KEY is required"),
    ):
        get_gemini_api_key()


def test_database_pool_limits_default_when_unset() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert get_database_pool_limits() == (1, 10)


def test_database_pool_limits_read_environment_variables() -> None:
    env = {"DB_POOL_MIN_SIZE": "2", "DB_POOL_MAX_SIZE": "20"}

    with patch.dict(os.environ, env, clear=True):
        assert get_database_pool_limits() == (2, 20)


def test_database_pool_limits_treat_blank_value_as_default() -> None:
    env = {"DB_POOL_MIN_SIZE": "  ", "DB_POOL_MAX_SIZE": "4"}

    with patch.dict(os.environ, env, clear=True):
        assert get_database_pool_limits() == (1, 4)


@pytest.mark.parametrize("value", ["abc", "2.5"])
def test_database_pool_limits_reject_non_integer(value: str) -> None:
    env = {"DB_POOL_MAX_SIZE": value}

    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="DB_POOL_MAX_SIZE must be a whole number"),
    ):
        get_database_pool_limits()


@pytest.mark.parametrize("value", ["0", "-3"])
def test_database_pool_limits_reject_values_below_one(value: str) -> None:
    env = {"DB_POOL_MIN_SIZE": value}

    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="DB_POOL_MIN_SIZE must be at least 1"),
    ):
        get_database_pool_limits()


def test_database_pool_limits_reject_max_below_min() -> None:
    env = {"DB_POOL_MIN_SIZE": "5", "DB_POOL_MAX_SIZE": "3"}

    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="greater than or equal to DB_POOL_MIN_SIZE"),
    ):
        get_database_pool_limits()
