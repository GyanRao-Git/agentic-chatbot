"""Request and response models used by the FastAPI server."""

from uuid import UUID

from pydantic import BaseModel, field_validator

from config import ALLOWED_MODEL_NAMES


class ConversationResponse(BaseModel):
    """Conversation data returned to a frontend."""

    conversation_id: UUID
    title: str | None = None


class ConversationTitleRequest(BaseModel):
    """Optional title supplied when creating or renaming a conversation."""

    title: str | None = None

    @field_validator("title")
    @classmethod
    def title_must_be_valid(cls, value: str | None) -> str | None:
        if value is None:
            return None

        title = value.strip()
        if not title:
            raise ValueError("Title must not be blank.")
        if len(title) > 100:
            raise ValueError("Title must not be longer than 100 characters.")

        return title


class ConversationMessage(BaseModel):
    """One saved message returned to a frontend."""

    role: str
    content: str


class ConversationMessagesResponse(BaseModel):
    """Saved messages belonging to one conversation."""

    conversation_id: UUID
    messages: list[ConversationMessage]
    next_before: int | None = None


class MessageRequest(BaseModel):
    """JSON body sent by a frontend when it sends a message."""

    message: str
    # Optional. When missing, the server uses the default model.
    model: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Message must not be empty.")
        return message

    @field_validator("model")
    @classmethod
    def model_must_be_allowed(cls, value: str | None) -> str | None:
        if value is not None and value not in ALLOWED_MODEL_NAMES:
            raise ValueError(f"Model must be one of: {', '.join(ALLOWED_MODEL_NAMES)}")
        return value


class MessageResponse(BaseModel):
    """JSON response returned to the frontend."""

    conversation_id: str
    answer: str
