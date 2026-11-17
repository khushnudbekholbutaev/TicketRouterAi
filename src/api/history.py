"""So'rovlar tarixi va operator feedback'i (SQLite, src/api/db.py).
Har bir /predict va /predict/batch natijasi foydalanuvchiga bog'lab saqlanadi; UI tarixni shu yerdan oladi —
barcha brauzer/qurilmalarda bir xil. Monitoring statistikasi ham shu bazadan hisoblanadi.

Operator "to'g'ri navbat"ni belgilasa (feedback), u qayta o'qitish uchun haqiqiy belgilangan ma'lumotga aylanadi:
    python -m src.api.history export-feedback [fayl.csv]     # subject, body, language, queue (to'g'ri), ...
"""
import csv
import json
import sys
import time

from src.api.db import connect


def save(username: str, source: str, items):
    """items: [(Ticket, Prediction), ...]"""
    now = time.time()
    rows = [(now, username, p.request_id, source, t.subject, t.body, p.language, p.queue, p.confidence,
             int(p.route_to_operator), p.model_version, json.dumps(p.model_dump(), ensure_ascii=False))
            for t, p in items]
    with connect() as con:
        con.executemany('INSERT INTO predictions (created_at, username, request_id, source, subject, body, language,'
                        ' queue, confidence, routed, model_version, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        rows)


def _item(r):
    return {'id': r['id'], 'created_at': r['created_at'], 'source': r['source'], 'subject': r['subject'],
            'body': r['body'], 'result': json.loads(r['result_json']),
            'feedback': ({'queue': r['feedback_queue'], 'by': r['feedback_by'], 'at': r['feedback_at']}
                         if r['feedback_queue'] else None)}


def list_for(username: str, limit=20, offset=0):
    with connect() as con:
        total = con.execute('SELECT COUNT(*) FROM predictions WHERE username=?', (username,)).fetchone()[0]
        rows = con.execute('SELECT * FROM predictions WHERE username=? ORDER BY created_at DESC, id DESC '
                           'LIMIT ? OFFSET ?', (username, limit, offset)).fetchall()
    return {'total': total, 'items': [_item(r) for r in rows]}


def clear_for(username: str) -> int:
    with connect() as con:
        return con.execute('DELETE FROM predictions WHERE username=?', (username,)).rowcount


def set_feedback(username: str, request_id: str, queue: str | None) -> dict | None:
    """Faqat o'z so'roviga feedback berish mumkin. queue=None — feedback'ni olib tashlash."""
    with connect() as con:
        n = con.execute('UPDATE predictions SET feedback_queue=?, feedback_by=?, feedback_at=? '
                        'WHERE request_id=? AND username=?',
                        (queue, username if queue else None, time.time() if queue else None,
                         request_id, username)).rowcount
        if not n:
            return None
        return _item(con.execute('SELECT * FROM predictions WHERE request_id=?', (request_id,)).fetchone())


def stats(username: str | None = None):
    where, args = ('WHERE username=?', (username,)) if username else ('', ())
    with connect() as con:
        tot = con.execute(f'SELECT COUNT(*) n, COALESCE(SUM(routed),0) r, COALESCE(AVG(confidence),0) c, '
                          f'COUNT(feedback_queue) f, COALESCE(SUM(feedback_queue = queue),0) fok '
                          f'FROM predictions {where}', args).fetchone()
        by_q = con.execute(f'SELECT queue k, COUNT(*) n FROM predictions {where} GROUP BY queue ORDER BY n DESC',
                           args).fetchall()
        by_l = con.execute(f'SELECT language k, COUNT(*) n FROM predictions {where} GROUP BY language ORDER BY n DESC',
                           args).fetchall()
        users = con.execute(f'SELECT COUNT(DISTINCT username) FROM predictions {where}', args).fetchone()[0]
    return {'total': tot['n'], 'routed': tot['r'], 'avg_confidence': round(tot['c'], 4), 'users': users,
            'feedback': tot['f'], 'feedback_correct': tot['fok'],  # model to'g'ri topgan feedback'lar soni
            'by_queue': {r['k']: r['n'] for r in by_q}, 'by_language': {r['k']: r['n'] for r in by_l}}


def export_feedback(path: str):
    cols = ['subject', 'body', 'language', 'feedback_queue', 'queue', 'confidence', 'model_version', 'created_at']
    with connect() as con:
        rows = con.execute(f'SELECT {", ".join(cols)} FROM predictions WHERE feedback_queue IS NOT NULL '
                           'ORDER BY created_at').fetchall()
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'body', 'language', 'queue', 'predicted_queue', 'confidence', 'model_version',
                    'created_at'])
        w.writerows([tuple(r) for r in rows])
    return len(rows)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'export-feedback':
        out = sys.argv[2] if len(sys.argv) > 2 else 'feedback.csv'
        print(f'{export_feedback(out)} ta qator -> {out}')
    else:
        print(__doc__)
