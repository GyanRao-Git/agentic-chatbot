import os
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api import app
from config import DEFAULT_MODEL_NAME
from model_factory import create_chat_model


# Checks that a model name outside the allowed list gets a 422 error before any work runs.
def test_chat_rejects_unknown_model() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")

    response = client.post(
        f"/conversations/{conversation_id}/messages",
        json={"message": "Hello", "model": "not-a-real-model"},
    )

    assert response.status_code == 422


# Checks that the model named in the request is the graph the agent uses.
def test_chat_uses_requested_model() -> None:
    client = TestClient(app)
    conversation_id = UUID("019535d9-3df7-79fb-b466-fa907fa17f9e")
    default_chatbot = MagicMock()
    other_chatbot = MagicMock()
    app.state.database_pool = MagicMock()
    app.state.chatbots = {
        DEFAULT_MODEL_NAME: default_chatbot,
        "gemini-2.5-flash": other_chatbot,
    }

    with patch(
        "api.call_agent",
        return_value={"conversation_id": str(conversation_id), "answer": "ok"},
    ) as send_mock:
        client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "Hello", "model": "gemini-2.5-flash"},
        )

    assert send_mock.call_args.kwargs["chatbot"] is other_chatbot
    assert send_mock.call_args.kwargs["model_name"] == "gemini-2.5-flash"


# Checks that create_chat_model refuses a model name that is not in the allowed list.
def test_create_chat_model_rejects_unknown_name() -> None:
    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
        pytest.raises(ValueError, match="Unsupported model"),
    ):
        create_chat_model(provider="gemini", model_name="not-a-real-model")
