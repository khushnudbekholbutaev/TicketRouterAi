"""So'rovlar tarixi (SQLite): har bir /predict va /predict/batch natijasi foydalanuvchiga bog'lab saqlanadi.
UI tarixni shu yerdan oladi — barcha brauzer/qurilmalarda bir xil. Monitoring statistikasi ham shu bazadan
hisoblanadi (uvicorn workerlari soniga bog'liq emas).
Fayl: HISTORY_DB (standart: var/history.db). Bir nechta worker uchun WAL rejimi."""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from src.config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    username TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'single' | 'batch'
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    language TEXT NOT NULL,
    queue TEXT NOT NULL,
    confidence REAL NOT NULL,
    routed INTEGER NOT NULL,
    result_json TEXT NOT NULL          -- to'liq javob (top3, reasons, threshold ...)
);
CREATE INDEX IF NOT EXISTS ix_pred_user_time ON predictions(username, created_at DESC);
"""


def db_path() -> Path:
    return Path(os.environ.get('HISTORY_DB', ROOT / 'var' / 'history.db'))


@contextmanager
def connect():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    try:
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA busy_timeout=10000')
        con.row_factory = sqlite3.Row
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def save(username: str, source: str, items):
    """items: [(Ticket, Prediction), ...]"""
    now = time.time()
    rows = [(now, username, p.request_id, source, t.subject, t.body, p.language, p.queue, p.confidence,
             int(p.route_to_operator), json.dumps(p.model_dump(), ensure_ascii=False)) for t, p in items]
    with connect() as con:
        con.executemany('INSERT INTO predictions (created_at, username, request_id, source, subject, body, language,'
                        ' queue, confidence, routed, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)


def list_for(username: str, limit=20, offset=0):
    with connect() as con:
        total = con.execute('SELECT COUNT(*) FROM predictions WHERE username=?', (username,)).fetchone()[0]
        rows = con.execute('SELECT * FROM predictions WHERE username=? ORDER BY created_at DESC, id DESC '
                           'LIMIT ? OFFSET ?', (username, limit, offset)).fetchall()
    items = [{'id': r['id'], 'created_at': r['created_at'], 'source': r['source'], 'subject': r['subject'],
              'body': r['body'], 'result': json.loads(r['result_json'])} for r in rows]
    return {'total': total, 'items': items}


def clear_for(username: str) -> int:
    with connect() as con:
        return con.execute('DELETE FROM predictions WHERE username=?', (username,)).rowcount


def stats(username: str | None = None):
    where, args = ('WHERE username=?', (username,)) if username else ('', ())
    with connect() as con:
        tot = con.execute(f'SELECT COUNT(*) n, COALESCE(SUM(routed),0) r, COALESCE(AVG(confidence),0) c '
                          f'FROM predictions {where}', args).fetchone()
        by_q = con.execute(f'SELECT queue k, COUNT(*) n FROM predictions {where} GROUP BY queue ORDER BY n DESC',
                           args).fetchall()
        by_l = con.execute(f'SELECT language k, COUNT(*) n FROM predictions {where} GROUP BY language ORDER BY n DESC',
                           args).fetchall()
        users = con.execute(f'SELECT COUNT(DISTINCT username) FROM predictions {where}', args).fetchone()[0]
    return {'total': tot['n'], 'routed': tot['r'], 'avg_confidence': round(tot['c'], 4), 'users': users,
            'by_queue': {r['k']: r['n'] for r in by_q}, 'by_language': {r['k']: r['n'] for r in by_l}}
