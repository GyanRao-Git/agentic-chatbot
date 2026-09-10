"""Request and response models used by the FastAPI server."""

from uuid import UUID

from pydantic import BaseModel, field_validator


class ConversationResponse(BaseModel):
    """Response returned after creating a conversation."""

    conversation_id: UUID


class ConversationMessage(BaseModel):
    """One saved message returned to a frontend."""

    role: str
    content: str


class ConversationMessagesResponse(BaseModel):
    """Saved messages belonging to one conversation."""

    conversation_id: UUID
    messages: list[ConversationMessage]


class MessageRequest(BaseModel):
    """JSON body sent by a frontend when it sends a message."""

    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message must not be empty.")
        return message


class MessageResponse(BaseModel):
    """JSON response returned to the frontend."""

    conversation_id: str
    answer: str
