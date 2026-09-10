# This file handles SQL operations for records in our conversations table.
# LangGraph checkpoint messages are handled separately in conversation_history.py.

from uuid import UUID

from psycopg import Connection
from psycopg.rows import DictRow


def create_conversation(
    connection: Connection[DictRow],
    title: str | None = None,
) -> UUID:
    """
        Creates a row in Conversations Table and returnes its conversation ID
    """
    row = connection.execute("""
        INSERT INTO conversations (title)
        VALUES (%s)
        RETURNING id
    """, (title,)).fetchone()

    if row is None:
        raise RuntimeError("Could not create a row in Conversations Table \n")

    return UUID(str(row["id"]))


def list_conversations(
    connection: Connection[DictRow],
) -> list[tuple[UUID, str | None]]:
    """Return conversation IDs and titles, with the newest first."""
    rows = connection.execute("""
        SELECT id, title
        FROM conversations
        ORDER BY id DESC
    """).fetchall()

    conversations: list[tuple[UUID, str | None]] = []

    for row in rows:
        conversation = (UUID(str(row["id"])), row["title"])
        conversations.append(conversation)

    return conversations


def update_conversation_title(
    connection: Connection[DictRow],
    conversation_id: UUID,
    title: str | None,
) -> None:
    """Update or clear the title of one conversation."""
    connection.execute("""
        UPDATE conversations
        SET title = %s
        WHERE id = %s
    """, (title, conversation_id))


def delete_conversation(
    connection: Connection[DictRow],
    conversation_id: UUID,
) -> None:
    """Delete one row from the conversations table."""
    connection.execute(
        "DELETE FROM conversations WHERE id = %s",
        (conversation_id,),
    )


def conversation_exists(
    connection: Connection[DictRow],
    conversation_id: UUID,
) -> bool:

    row = connection.execute("""
        SELECT EXISTS ( 
            SELECT 1 FROM conversations WHERE id = %s
        ) AS conversation_exists
    """, 
    (conversation_id,)
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Could not check conversation_id: {conversation_id} in postgreSQL \n")

    return bool(row["conversation_exists"])

