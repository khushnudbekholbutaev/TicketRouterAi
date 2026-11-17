"""Ilovaning yagona SQLite bazasi: foydalanuvchilar, sessiyalar, login urinishlari, so'rovlar tarixi va feedback.
Fayl: APP_DB (standart: var/app.db). Docker'da `history` volume'ida turadi — rebuild'da saqlanadi.
Bir nechta uvicorn worker bir vaqtda yozishi uchun WAL rejimi va busy_timeout ishlatiladi.
Sxema versiyalanadi (PRAGMA user_version): yangi o'zgarish = MIGRATIONS ro'yxatiga yangi element."""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import ROOT

MIGRATIONS = [
    # v1: asosiy jadvallar
    """
    CREATE TABLE IF NOT EXISTS users (
        username      TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        is_active     INTEGER NOT NULL DEFAULT 1,
        created_at    REAL NOT NULL,
        last_login_at REAL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id          TEXT PRIMARY KEY,
        username    TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
        created_at  REAL NOT NULL,
        expires_at  REAL NOT NULL,
        revoked_at  REAL,
        ip          TEXT,
        user_agent  TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(username);
    CREATE TABLE IF NOT EXISTS login_attempts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts        REAL NOT NULL,
        ip        TEXT NOT NULL,
        username  TEXT NOT NULL,
        success   INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_attempts_key ON login_attempts(ip, username, ts);
    CREATE TABLE IF NOT EXISTS predictions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  REAL NOT NULL,
        username    TEXT NOT NULL,
        request_id  TEXT NOT NULL UNIQUE,
        source      TEXT NOT NULL,          -- 'single' | 'batch'
        subject     TEXT NOT NULL,
        body        TEXT NOT NULL,
        language    TEXT NOT NULL,
        queue       TEXT NOT NULL,
        confidence  REAL NOT NULL,
        routed      INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        result_json TEXT NOT NULL,          -- to'liq javob (top3, reasons, threshold ...)
        feedback_queue TEXT,                -- operator belgilagan to'g'ri navbat (qayta o'qitish uchun)
        feedback_by    TEXT,
        feedback_at    REAL
    );
    CREATE INDEX IF NOT EXISTS ix_pred_user_time ON predictions(username, created_at DESC);
    CREATE INDEX IF NOT EXISTS ix_pred_feedback ON predictions(feedback_queue) WHERE feedback_queue IS NOT NULL;
    """,
]


def db_path() -> Path:
    return Path(os.environ.get('APP_DB', ROOT / 'var' / 'app.db'))


@contextmanager
def connect():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    try:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA busy_timeout=10000')
        con.execute('PRAGMA foreign_keys=ON')
        con.row_factory = sqlite3.Row
        yield con
        con.commit()
    finally:
        con.close()


def migrate():
    with connect() as con:
        con.execute('BEGIN IMMEDIATE')  # ikki worker bir vaqtda migratsiya qilmasin
        v = con.execute('PRAGMA user_version').fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[v:], start=v + 1):
            for stmt in [s for s in sql.split(';') if s.strip()]:
                con.execute(stmt)
            con.execute(f'PRAGMA user_version={i}')
        return con.execute('PRAGMA user_version').fetchone()[0]
