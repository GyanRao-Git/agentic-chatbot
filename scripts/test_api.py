"""Manually test the running FastAPI server."""

import requests

API_URL = "http://127.0.0.1:8000"


# Check whether the API server is running.
health_response = requests.get(f"{API_URL}/health", timeout=10)
health_response.raise_for_status()
print("Health:", health_response.json())


# Create one conversation and save its UUID.
conversation_response = requests.post(
    f"{API_URL}/conversations",
    timeout=10,
)
conversation_response.raise_for_status()
conversation_id = conversation_response.json()["conversation_id"]
print("Conversation ID:", conversation_id)


# Both messages use the same conversation ID so checkpoint memory is reused.
first_response = requests.post(
    f"{API_URL}/conversations/{conversation_id}/messages",
    json={"message": "My name is Gyan"},
    timeout=60,
)
first_response.raise_for_status()
print("First answer:", first_response.json()["answer"])


second_response = requests.post(
    f"{API_URL}/conversations/{conversation_id}/messages",
    json={"message": "What is my name?"},
    timeout=60,
)
second_response.raise_for_status()
print("Memory answer:", second_response.json()["answer"])
