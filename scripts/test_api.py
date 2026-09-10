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
    json={"title": "API test conversation"},
    timeout=10,
)
conversation_response.raise_for_status()
conversation_id = conversation_response.json()["conversation_id"]
print("Conversation ID:", conversation_id)


# Check that the new conversation appears in the list.
conversations_response = requests.get(f"{API_URL}/conversations", timeout=10)
conversations_response.raise_for_status()
print("Conversations:", conversations_response.json())


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


# Ask a live-time question so Gemini must use the new clock tool.
clock_response = requests.post(
    f"{API_URL}/conversations/{conversation_id}/messages",
    json={"message": "What time is it in Asia/Kolkata right now?"},
    timeout=60,
)
clock_response.raise_for_status()
print("Clock answer:", clock_response.json()["answer"])


# Fetch the messages that LangGraph saved for this conversation.
messages_response = requests.get(
    f"{API_URL}/conversations/{conversation_id}/messages",
    timeout=10,
)
messages_response.raise_for_status()
print("Saved messages:", messages_response.json()["messages"])


# Debug mode also returns the tool result saved inside the checkpoint.
debug_messages_response = requests.get(
    f"{API_URL}/conversations/{conversation_id}/messages",
    params={"debug": True},
    timeout=10,
)
debug_messages_response.raise_for_status()
print("Debug messages:", debug_messages_response.json()["messages"])


# Rename the conversation.
rename_response = requests.patch(
    f"{API_URL}/conversations/{conversation_id}",
    json={"title": "Renamed API test"},
    timeout=10,
)
rename_response.raise_for_status()
print("Renamed conversation:", rename_response.json())


# Delete both the conversation row and its LangGraph history.
delete_response = requests.delete(
    f"{API_URL}/conversations/{conversation_id}",
    timeout=10,
)
delete_response.raise_for_status()
print("Delete status:", delete_response.status_code)
