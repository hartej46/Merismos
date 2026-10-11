from datetime import datetime, timezone

from .database import get_connection


def create_memory_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            importance REAL NOT NULL DEFAULT 0.5,
            source_message_id INTEGER,
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL,
            expires_at TEXT,
            status TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    connection.commit()


def save_memory(
    content,
    memory_type="general",
    importance=0.5,
    confidence=0.5,
    source_message_id=None,
    expires_at=None,
):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            """
            INSERT INTO memories
                (content, memory_type, importance, confidence,
                source_message_id, created_at, last_used_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content.strip(),
                memory_type,
                importance,
                confidence,
                source_message_id,
                created_at,
                created_at,
                expires_at,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return the new memory id.")
        return cursor.lastrowid


def get_active_memories(limit = 10):
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        create_memory_table(connection)
        return connection.execute(
            """
            SELECT id, content, memory_type, confidence, importance,
                source_message_id, created_at, last_used_at, expires_at
            FROM memories
            WHERE status = 'active'
                AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (datetime.now(timezone.utc).isoformat(), limit),
        ).fetchall()


def get_memory_with_id(memory_id):
    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            """
                SELECT id, content, memory_type, confidence, importance,
                        source_message_id, created_at, last_used_at, expires_at, status
                FROM memories
                WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
    return cursor

def search_memories(query, limit = 10):
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            """
                SELECT id, content, memory_type, confidence, importance,
                        source_message_id, created_at, last_used_at, expires_at, status
                FROM memories
                WHERE content LIKE ?
                    AND status = 'active'
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """,
            (f"%{query.strip()}%", limit)
        ).fetchall()

    return cursor


def update_memory(memory_id,content=None,memory_type=None,importance=None,confidence=None):
    updates = []
    values = []

    if content is not None:
        updates.append("content = ?")
        values.append(content.strip())
    if memory_type is not None:
        updates.append("memory_type = ?")
        values.append(memory_type)
    if importance is not None:
        updates.append("importance = ?")
        values.append(importance)
    if confidence is not None:
        updates.append("confidence = ?")
        values.append(confidence)

    if not updates:
        raise ValueError("At least one field must be provided to update.")

    values.append(memory_id)
    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            f"""
            UPDATE memories
            SET {", ".join(updates)}
            WHERE id = ?
            """,
            values,
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Memory with id {memory_id} was not found.")

    return get_memory_with_id(memory_id)


def archive_memory(memory_id):
    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            """
                UPDATE memories
                SET status = 'archived'
                WHERE id = ? AND status = 'active'
            """,
            (memory_id,),
        )

    if cursor.rowcount == 0:
        raise ValueError(f"Active memory with id {memory_id} was not found.")

    return get_memory_with_id(memory_id)


def delete_memory(memory_id):
    with get_connection() as connection:
        create_memory_table(connection)
        cursor = connection.execute(
            """
                DELETE FROM memories
                WHERE id = ?
            """,
            (memory_id,),
        )

    if cursor.rowcount == 0:
        raise ValueError(f"Memory with id {memory_id} was not found.")

    return cursor.rowcount
