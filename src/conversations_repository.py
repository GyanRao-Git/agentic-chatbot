from uuid import UUID

from psycopg import Connection


def create_conversation(connection: Connection) -> UUID:
    """
        Creates a row in Conversations Table and returnes its conversation ID
    """
    row = connection.execute(
        "INSERT INTO conversations DEFAULT VALUES RETURNING id"
    ).fetchone()

    if row is None:
        raise RuntimeError("Could not create a row in Conversations Table \n")

    return UUID(str(row[0]))

def conversation_exists(
    connection: Connection,
    conversation_id: UUID,
) -> bool:

    row = connection.execute("""
        SELECT EXISTS ( 
            SELECT 1 FROM conversations WHERE id = %s )
    """, 
    (conversation_id,)
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Could not check conversation_id: {conversation_id} in postgreSQL \n")

    return bool(row[0])

