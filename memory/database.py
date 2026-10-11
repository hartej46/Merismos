import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).with_name("memory_store.sqlite3")
CONNECTION_TIMEOUT_SECONDS = 5


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
