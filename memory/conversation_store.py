from datetime import datetime, timezone

from .database import get_connection


def create_conversation_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def save_conversation(title):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        create_conversation_table(connection)
        cursor = connection.execute(
            """
            INSERT INTO conversations
                (title, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (title.strip(), created_at, created_at),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return the new conversation id.")
        return cursor.lastrowid


def get_conversation_with_id(conversation_id):
    with get_connection() as connection:
        create_conversation_table(connection)
        cursor = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return cursor


def get_conversations(limit=10):
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        create_conversation_table(connection)
        return connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def update_conversation(conversation_id, title=None):
    if title is None:
        raise ValueError("Title must be provided to update.")

    with get_connection() as connection:
        create_conversation_table(connection)
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title.strip(), datetime.now(timezone.utc).isoformat(), conversation_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Conversation with id {conversation_id} was not found.")

    return get_conversation_with_id(conversation_id)


def delete_conversation(conversation_id):
    with get_connection() as connection:
        create_conversation_table(connection)
        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )
    if cursor.rowcount == 0:
        raise ValueError(f"Conversation with id {conversation_id} was not found.")
    return cursor.rowcount
