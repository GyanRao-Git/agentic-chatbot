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

