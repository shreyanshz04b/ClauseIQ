import sqlite3
from config import DB_PATH


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT,
        keywords TEXT,
        page INTEGER DEFAULT 0,
        source TEXT DEFAULT 'uploaded'
    )
    """)

    conn.commit()
    conn.close()


def migrate_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(docs)")
    columns = [c[1] for c in cur.fetchall()]

    if "page" not in columns:
        cur.execute("ALTER TABLE docs ADD COLUMN page INTEGER DEFAULT 0")

    if "source" not in columns:
        cur.execute("ALTER TABLE docs ADD COLUMN source TEXT DEFAULT 'uploaded'")

    conn.commit()
    conn.close()