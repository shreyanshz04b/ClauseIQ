import sqlite3

conn = sqlite3.connect("legal.db")
cur = conn.cursor()

try:
    cur.execute("ALTER TABLE docs ADD COLUMN page INTEGER DEFAULT 0")
    print("Added column: page")
except Exception as e:
    print("page exists or error:", e)

try:
    cur.execute("ALTER TABLE docs ADD COLUMN source TEXT DEFAULT 'uploaded'")
    print("Added column: source")
except Exception as e:
    print("source exists or error:", e)

conn.commit()
conn.close()

print("DB FIX DONE")