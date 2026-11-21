from unittest.mock import MagicMock, patch

from setup_database import setup_database


def test_setup_database_creates_application_and_checkpoint_tables() -> None:
    database_url = "postgresql://test"

    connection = MagicMock()
    connection_context = MagicMock()
    connection_context.__enter__.return_value = connection

    checkpointer = MagicMock()
    checkpointer_context = MagicMock()
    checkpointer_context.__enter__.return_value = checkpointer

    with (
        patch(
            "setup_database.psycopg.connect",
            return_value=connection_context,
        ) as connect_mock,
        patch(
            "setup_database.PostgresSaver.from_conn_string",
            return_value=checkpointer_context,
        ) as checkpointer_mock,
    ):
        setup_database(database_url)

    connect_mock.assert_called_once_with(
        database_url,
        autocommit=True,
        connect_timeout=5,
    )
    create_table_query = connection.execute.call_args_list[0].args[0]
    assert "CREATE TABLE IF NOT EXISTS conversations" in create_table_query
    assert "DEFAULT uuidv7()" in create_table_query
    assert "title VARCHAR(100)" in create_table_query
    checkpointer_mock.assert_called_once_with(database_url)
    checkpointer.setup.assert_called_once_with()
