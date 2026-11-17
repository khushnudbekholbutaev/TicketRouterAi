"""UI uchun login (SQLite, src/api/db.py): foydalanuvchilar (PBKDF2-SHA256 xesh), server tomonda saqlanadigan
sessiyalar (cookie'da HMAC bilan imzolangan sessiya ID) va login urinishlari jurnali (parol terishga qarshi blok).

Boshqaruv (Docker'da: docker compose exec api python -m src.api.auth ...; parol yashirin so'raladi):
    add <login>          yangi foydalanuvchi yoki parolni almashtirish (barcha sessiyalari yopiladi)
    remove <login>       o'chirish (sessiyalari ham)
    disable|enable <login>
    logout <login>       barcha sessiyalarini yopish
    list                 foydalanuvchilar, oxirgi kirish, faol sessiyalar
    attempts [n]         oxirgi n ta login urinishi
Eski auth/users.json (AUTH_USERS_FILE) bo'lsa va bazada foydalanuvchi bo'lmasa — birinchi ishga tushishda ko'chiriladi.
nginx har bir /api so'rovidan oldin GET /auth/check ga murojaat qiladi (auth_request)."""
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.api.db import connect
from src.config import ROOT

COOKIE = 'tr_session'
ITER = 260_000
TTL = int(os.environ.get('AUTH_SESSION_HOURS', '12')) * 3600
FAIL_LIMIT, FAIL_WINDOW = 10, 300  # 5 daqiqada 10 ta noto'g'ri urinish -> vaqtinchalik blok (IP + login bo'yicha)
# Barcha uvicorn workerlarida bir xil bo'lishi shart; .env da AUTH_SECRET bo'lmasa API_KEY dan hosil qilinadi
_SECRET = os.environ.get('AUTH_SECRET') or hashlib.sha256(
    ('tr-auth:' + os.environ.get('API_KEY', 'change-me-dev-key')).encode()).hexdigest()

router = APIRouter(prefix='/auth', tags=['auth'])


# ---------- parollar ----------
def hash_password(pw: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, ITER)
    return f'pbkdf2_sha256${ITER}${salt.hex()}${dk.hex()}'


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split('$')
        dk = hashlib.pbkdf2_hmac('sha256', pw.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), h)
    except (ValueError, TypeError):
        return False


_DUMMY = hash_password('dummy-password-for-timing')  # yo'q login uchun ham bir xil vaqt sarflansin


# ---------- foydalanuvchilar ----------
def norm(name: str) -> str:
    return name.strip().lower()


def set_user(name: str, password: str):
    name = norm(name)
    with connect() as con:
        con.execute('INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?) '
                    'ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, is_active=1',
                    (name, hash_password(password), time.time()))
        con.execute('UPDATE sessions SET revoked_at=? WHERE username=? AND revoked_at IS NULL', (time.time(), name))


def delete_user(name: str) -> bool:
    with connect() as con:
        return con.execute('DELETE FROM users WHERE username=?', (norm(name),)).rowcount > 0


def set_active(name: str, active: bool) -> bool:
    name = norm(name)
    with connect() as con:
        n = con.execute('UPDATE users SET is_active=? WHERE username=?', (int(active), name)).rowcount
        if not active:
            con.execute('UPDATE sessions SET revoked_at=? WHERE username=? AND revoked_at IS NULL', (time.time(), name))
        return n > 0


def revoke_all(name: str) -> int:
    with connect() as con:
        return con.execute('UPDATE sessions SET revoked_at=? WHERE username=? AND revoked_at IS NULL',
                           (time.time(), norm(name))).rowcount


def list_users():
    with connect() as con:
        return [dict(r) for r in con.execute(
            'SELECT u.username, u.is_active, u.created_at, u.last_login_at, '
            '(SELECT COUNT(*) FROM sessions s WHERE s.username=u.username AND s.revoked_at IS NULL '
            ' AND s.expires_at > ?) AS active_sessions FROM users u ORDER BY u.username', (time.time(),))]


def import_users_file():
    """Bir martalik migratsiya: auth/users.json -> users jadvali (faqat bazada foydalanuvchi bo'lmasa)."""
    p = Path(os.environ.get('AUTH_USERS_FILE', ROOT / 'auth' / 'users.json'))
    try:
        users = json.loads(p.read_text() or '{}')
    except (FileNotFoundError, ValueError):
        return 0
    with connect() as con:
        if con.execute('SELECT COUNT(*) FROM users').fetchone()[0]:
            return 0
        con.executemany('INSERT OR IGNORE INTO users (username, password_hash, created_at) VALUES (?,?,?)',
                        [(norm(u), h, time.time()) for u, h in users.items()])
    return len(users)


# ---------- sessiyalar ----------
def _sign(payload: str) -> str:
    return hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def create_session(user: str, ip: str = '', ua: str = '') -> str:
    sid, now = secrets.token_urlsafe(24), time.time()
    with connect() as con:
        con.execute('INSERT INTO sessions (id, username, created_at, expires_at, ip, user_agent) VALUES (?,?,?,?,?,?)',
                    (sid, norm(user), now, now + TTL, ip, ua[:200]))
        con.execute('UPDATE users SET last_login_at=? WHERE username=?', (now, norm(user)))
        con.execute('DELETE FROM sessions WHERE expires_at < ?', (now - 7 * 86400,))  # eski yozuvlarni tozalash
    payload = base64.urlsafe_b64encode(json.dumps({'sid': sid}).encode()).decode()
    return payload + '.' + _sign(payload)


def read_token(token: str | None) -> str | None:
    """Imzo to'g'ri + sessiya bazada faol + foydalanuvchi faol bo'lsa — login qaytadi."""
    if not token or '.' not in token:
        return None
    payload, sig = token.rsplit('.', 1)
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        sid = json.loads(base64.urlsafe_b64decode(payload.encode()))['sid']
    except (ValueError, KeyError, TypeError):
        return None
    with connect() as con:
        r = con.execute('SELECT s.username FROM sessions s JOIN users u ON u.username=s.username '
                        'WHERE s.id=? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.is_active=1',
                        (sid, time.time())).fetchone()
    return r['username'] if r else None


def revoke_token(token: str | None):
    try:
        sid = json.loads(base64.urlsafe_b64decode(token.rsplit('.', 1)[0].encode()))['sid']
    except Exception:
        return
    with connect() as con:
        con.execute('UPDATE sessions SET revoked_at=? WHERE id=? AND revoked_at IS NULL', (time.time(), sid))


# ---------- login urinishlari ----------
def _client(request: Request) -> str:
    return request.headers.get('x-real-ip') or (request.client.host if request.client else '?')


def _too_many(ip: str, name: str) -> bool:
    with connect() as con:
        n = con.execute('SELECT COUNT(*) FROM login_attempts WHERE ip=? AND username=? AND success=0 AND ts > '
                        'COALESCE((SELECT MAX(ts) FROM login_attempts WHERE ip=? AND username=? AND success=1), 0) '
                        'AND ts > ?', (ip, name, ip, name, time.time() - FAIL_WINDOW)).fetchone()[0]
    return n >= FAIL_LIMIT


def _log_attempt(ip: str, name: str, ok: bool):
    with connect() as con:
        con.execute('INSERT INTO login_attempts (ts, ip, username, success) VALUES (?,?,?,?)',
                    (time.time(), ip, name[:64], int(ok)))
        con.execute('DELETE FROM login_attempts WHERE ts < ?', (time.time() - 30 * 86400,))  # 30 kundan eskisi


# ---------- endpointlar ----------
class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post('/login')
def login(body: Login, request: Request, response: Response):
    ip, name = _client(request), norm(body.username)
    if _too_many(ip, name):  # blok IP + login bo'yicha: proksi ortidagi boshqa foydalanuvchilarga ta'sir qilmaydi
        raise HTTPException(429, 'Juda ko\'p noto\'g\'ri urinish. 5 daqiqadan so\'ng qayta urinib ko\'ring.')
    with connect() as con:
        r = con.execute('SELECT password_hash, is_active FROM users WHERE username=?', (name,)).fetchone()
    ok = verify_password(body.password, r['password_hash'] if r else _DUMMY) and bool(r) and bool(r['is_active'])
    _log_attempt(ip, name, ok)
    if not ok:
        raise HTTPException(401, 'Login yoki parol noto\'g\'ri')
    token = create_session(name, ip, request.headers.get('user-agent', ''))
    response.set_cookie(COOKIE, token, max_age=TTL, httponly=True, samesite='lax',
                        secure=os.environ.get('AUTH_COOKIE_SECURE', '0') == '1', path='/')
    return {'user': name}


@router.post('/logout')
def logout(request: Request, response: Response):
    revoke_token(request.cookies.get(COOKIE))  # sessiya serverda yopiladi — cookie nusxasi ham ishlamay qoladi
    response.delete_cookie(COOKIE, path='/')
    return {'ok': True}


@router.get('/check')
def check(request: Request, response: Response):
    user = read_token(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, 'Avval tizimga kiring')
    response.headers['X-User'] = user
    return {'user': user}


# ---------- CLI ----------
def _fmt(t):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else '—'


def cli(argv: list[str]):
    from src.api.db import migrate
    migrate()
    import_users_file()
    cmd = argv[0] if argv else ''
    one = {'add', 'remove', 'disable', 'enable', 'logout'}
    if cmd not in one | {'list', 'attempts'} or (cmd in one and len(argv) != 2):
        print(__doc__)
        sys.exit(2)
    if cmd == 'list':
        rows = list_users()
        for u in rows:
            print(f"{u['username']:<20} {'faol' if u['is_active'] else 'o‘chirilgan':<11} "
                  f"oxirgi kirish: {_fmt(u['last_login_at']):<17} sessiyalar: {u['active_sessions']}")
        print('' if rows else '(foydalanuvchi yo\'q)', end='')
        return
    if cmd == 'attempts':
        n = int(argv[1]) if len(argv) > 1 else 20
        with connect() as con:
            for r in con.execute('SELECT * FROM login_attempts ORDER BY ts DESC LIMIT ?', (n,)):
                print(_fmt(r['ts']), r['ip'], r['username'], 'OK' if r['success'] else 'XATO')
        return
    name = norm(argv[1])
    if cmd == 'remove':
        print('o\'chirildi' if delete_user(name) else 'topilmadi', name)
    elif cmd in ('disable', 'enable'):
        print('saqlandi' if set_active(name, cmd == 'enable') else 'topilmadi', name)
    elif cmd == 'logout':
        print(f'{revoke_all(name)} ta sessiya yopildi')
    else:
        pw = getpass.getpass(f'{name} uchun parol: ')
        if len(pw) < 8:
            sys.exit('Parol kamida 8 belgi bo\'lishi kerak')
        if getpass.getpass('Parolni takrorlang: ') != pw:
            sys.exit('Parollar mos emas')
        set_user(name, pw)
        print(f'saqlandi: {name}')


if __name__ == '__main__':
    cli(sys.argv[1:])
