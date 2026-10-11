from datetime import datetime, timezone

from .conversation_store import create_conversation_table
from .database import get_connection


def create_message_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    connection.commit()


def save_message(conversation_id, role, content):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        create_conversation_table(connection)
        create_message_table(connection)
        cursor = connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content.strip(), created_at),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (created_at, conversation_id),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return the new message id.")
        return cursor.lastrowid


def get_message_with_id(message_id):
    with get_connection() as connection:
        create_conversation_table(connection)
        create_message_table(connection)
        cursor = connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
    return cursor


def get_messages_for_conversation(conversation_id, limit=50):
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        create_conversation_table(connection)
        create_message_table(connection)
        return connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()


def delete_message(message_id):
    with get_connection() as connection:
        create_conversation_table(connection)
        create_message_table(connection)
        cursor = connection.execute(
            """
            DELETE FROM messages
            WHERE id = ?
            """,
            (message_id,),
        )
    if cursor.rowcount == 0:
        raise ValueError(f"Message with id {message_id} was not found.")
    return cursor.rowcount
