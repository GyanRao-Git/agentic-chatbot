"""Create the application and LangGraph database tables."""

import os

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver


def setup_database(database_url: str) -> None:
    """Create every table required by the chatbot."""

    # Our table stores the IDs used by the API and as LangGraph thread IDs.
    with psycopg.connect(
        database_url,
        autocommit=True,
        connect_timeout=5,
    ) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT uuidv7(),
                title VARCHAR(100)
            )
        """)

    # LangGraph creates and updates its own checkpoint tables.
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()


def main() -> None:
    """Load local configuration and prepare the database."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is required to set up the database.")

    setup_database(database_url)
    print("Database setup completed.")


if __name__ == "__main__":
    main()
