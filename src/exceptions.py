"""Application-specific errors that API routes can translate safely."""


class AgentExecutionError(RuntimeError):
    """Raised when the chatbot cannot complete a model or graph request."""

    def __init__(self) -> None:
        super().__init__("The agent could not complete the request. Please try again.")
