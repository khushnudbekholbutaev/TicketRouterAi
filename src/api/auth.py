"""UI uchun login: foydalanuvchilar fayli (PBKDF2-SHA256 xesh), HMAC bilan imzolangan sessiya cookie.

Foydalanuvchi qo'shish / o'chirish (parol terminalda yashirin so'raladi):
    python -m src.api.auth add <login>
    python -m src.api.auth remove <login>
    python -m src.api.auth list
Fayl: AUTH_USERS_FILE (standart: auth/users.json). Fayl har o'zgarganda qayta o'qiladi — restart kerak emas.
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

from src.config import ROOT

COOKIE = 'tr_session'
ITER = 260_000
TTL = int(os.environ.get('AUTH_SESSION_HOURS', '12')) * 3600
# Barcha uvicorn workerlarida bir xil bo'lishi shart; .env da AUTH_SECRET bo'lmasa API_KEY dan hosil qilinadi
_SECRET = os.environ.get('AUTH_SECRET') or hashlib.sha256(
    ('tr-auth:' + os.environ.get('API_KEY', 'change-me-dev-key')).encode()).hexdigest()
_cache = {'mtime': None, 'users': {}}
_fails: dict[str, list[float]] = {}

router = APIRouter(prefix='/auth', tags=['auth'])


def users_file() -> Path:
    return Path(os.environ.get('AUTH_USERS_FILE', ROOT / 'auth' / 'users.json'))


def load_users() -> dict:
    p = users_file()
    try:
        st = p.stat()
    except FileNotFoundError:
        return {}
    # _save() faylni atomar almashtiradi (yangi inode), shuning uchun inode ham kalitga kiradi:
    # bir zumda ketma-ket ikki yozuvda mtime bir xil bo'lishi mumkin
    m = (st.st_ino, st.st_mtime_ns, st.st_size)
    if m != _cache['mtime']:
        _cache.update(mtime=m, users=json.loads(p.read_text() or '{}'))
    return _cache['users']


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


def _sign(payload: str) -> str:
    return hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(user: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({'u': user, 'exp': int(time.time()) + TTL}).encode()).decode()
    return payload + '.' + _sign(payload)


def read_token(token: str | None) -> str | None:
    if not token or '.' not in token:
        return None
    payload, sig = token.rsplit('.', 1)
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode()))
    except ValueError:
        return None
    if data.get('exp', 0) < time.time() or data.get('u') not in load_users():
        return None  # muddati o'tgan yoki foydalanuvchi o'chirilgan
    return data['u']


def _client(request: Request) -> str:
    return request.headers.get('x-real-ip') or (request.client.host if request.client else '?')


def _too_many(ip: str, limit=10, window=300) -> bool:
    now = time.time()
    _fails[ip] = [t for t in _fails.get(ip, []) if now - t < window]
    return len(_fails[ip]) >= limit


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post('/login')
def login(body: Login, request: Request, response: Response):
    ip = _client(request)
    if _too_many(ip):
        raise HTTPException(429, 'Juda ko\'p noto\'g\'ri urinish. 5 daqiqadan so\'ng qayta urinib ko\'ring.')
    stored = load_users().get(body.username)
    if not stored or not verify_password(body.password, stored):
        _fails.setdefault(ip, []).append(time.time())
        raise HTTPException(401, 'Login yoki parol noto\'g\'ri')
    _fails.pop(ip, None)
    response.set_cookie(COOKIE, make_token(body.username), max_age=TTL, httponly=True, samesite='strict',
                        secure=os.environ.get('AUTH_COOKIE_SECURE', '0') == '1', path='/')
    return {'user': body.username}


@router.post('/logout')
def logout(response: Response):
    response.delete_cookie(COOKIE, path='/')
    return {'ok': True}


@router.get('/check')
def check(request: Request, response: Response):
    user = read_token(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, 'Avval tizimga kiring')
    response.headers['X-User'] = user
    return {'user': user}


def _save(users: dict):
    p = users_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(users, indent=2))
    os.chmod(tmp, 0o644)
    tmp.replace(p)


def cli(argv: list[str]):
    if len(argv) < 1 or argv[0] not in ('add', 'remove', 'list') or (argv[0] != 'list' and len(argv) != 2):
        print(__doc__)
        sys.exit(2)
    users = dict(load_users())
    cmd = argv[0]
    if cmd == 'list':
        print('\n'.join(sorted(users)) or '(foydalanuvchi yo\'q)')
        return
    name = argv[1]
    if cmd == 'remove':
        users.pop(name, None)
        _save(users)
        print(f'o\'chirildi: {name}')
        return
    pw = getpass.getpass(f'{name} uchun parol: ')
    if len(pw) < 8:
        sys.exit('Parol kamida 8 belgi bo\'lishi kerak')
    if getpass.getpass('Parolni takrorlang: ') != pw:
        sys.exit('Parollar mos emas')
    users[name] = hash_password(pw)
    _save(users)
    print(f'saqlandi: {name} -> {users_file()}')


if __name__ == '__main__':
    cli(sys.argv[1:])
