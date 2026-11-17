import os

os.environ['API_KEY'] = 'test-key'
os.environ['RATE_LIMIT'] = '1000/minute'
import tempfile  # noqa: E402
os.environ['AUTH_USERS_FILE'] = os.path.join(tempfile.mkdtemp(), 'users.json')
os.environ['APP_DB'] = os.path.join(tempfile.mkdtemp(), 'app.db')

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api import auth  # noqa: E402
from src.api.main import app, detect_language, mask_pii  # noqa: E402

H = {'X-API-Key': 'test-key'}
T = {'subject': 'Refund', 'body': 'I was charged twice for my subscription, please refund the payment.'}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_predict_ok_contract(client):
    r = client.post('/predict', json=T, headers=H)
    assert r.status_code == 200
    j = r.json()
    assert {'request_id', 'queue', 'confidence', 'top3', 'route_to_operator', 'reasons', 'model_version'} <= set(j)
    assert len(j['top3']) == 3 and 0 <= j['confidence'] <= 1


def test_predict_requires_api_key(client):
    assert client.post('/predict', json=T).status_code == 401
    assert client.post('/predict', json=T, headers={'X-API-Key': 'wrong'}).status_code == 401


def test_predict_validation_short_body(client):
    assert client.post('/predict', json={'body': 'hi'}, headers=H).status_code == 422


def test_predict_validation_too_long(client):
    assert client.post('/predict', json={'body': 'a' * 5001}, headers=H).status_code == 422


def test_predict_bad_language(client):
    assert client.post('/predict', json={**T, 'language': 'fr'}, headers=H).status_code == 422


def test_batch(client):
    r = client.post('/predict/batch', json={'tickets': [T, {**T, 'body': 'VPN ulanmayapti, iltimos yordam bering.'}]},
                    headers=H)
    assert r.status_code == 200 and len(r.json()) == 2
    assert r.json()[1]['language'] == 'uz'


def test_batch_limit(client):
    assert client.post('/predict/batch', json={'tickets': [T] * 101}, headers=H).status_code == 422


def test_metrics_and_info(client):
    assert 'predictions_total' in client.get('/metrics').text
    info = client.get('/model/info').json()
    assert len(info['classes']) == 10


def test_pii_masking_and_language():
    s = mask_pii('mail me at a.b@c.uz or +998 90 123 45 67')
    assert '<EMAIL>' in s and '<PHONE>' in s and '@' not in s
    assert detect_language('Die Rechnung ist falsch, bitte prüfen Sie das') == 'de'
    assert detect_language("Internet ishlamayapti, iltimos yordam bering") == 'uz'
    assert detect_language('С моей карты дважды списали деньги, верните пожалуйста') == 'ru'
    assert detect_language('Интернет ишламаяпти, илтимос ёрдам беринг, ҳозироқ') != 'ru'  # o'zbek kirill


def test_predict_russian(client):
    r = client.post('/predict', json={'subject': 'Оплата', 'body': 'С моей карты дважды списали деньги, верните пожалуйста.'},
                    headers=H)
    assert r.status_code == 200 and r.json()['language'] == 'ru'


TEST_PW = 'unit-test-' + os.urandom(4).hex()


def test_auth_flow(client):
    auth.set_user('tester', TEST_PW)
    client.cookies.clear()
    assert client.get('/auth/check').status_code == 401
    assert client.post('/auth/login', json={'username': 'tester', 'password': 'wrong-password'}).status_code == 401
    assert client.post('/auth/login', json={'username': 'nobody', 'password': TEST_PW}).status_code == 401
    r = client.post('/auth/login', json={'username': 'tester', 'password': TEST_PW})
    assert r.status_code == 200 and auth.COOKIE in r.cookies
    tok = r.cookies[auth.COOKIE]
    c = client.get('/auth/check')
    assert c.status_code == 200 and c.json()['user'] == 'tester'
    client.post('/auth/logout')
    client.cookies.clear()
    assert client.get('/auth/check').status_code == 401
    # logout sessiyani serverda yopadi: eski cookie nusxasi ham ishlamaydi
    assert client.get('/auth/check', cookies={auth.COOKIE: tok}).status_code == 401
    client.cookies.clear()


def test_session_tamper_disable_remove_and_password_change():
    auth.set_user('tester', TEST_PW)
    tok = auth.create_session('tester')
    assert auth.read_token(tok) == 'tester'
    assert auth.read_token(tok[:-1] + ('0' if tok[-1] != '0' else '1')) is None
    assert auth.set_active('tester', False) and auth.read_token(tok) is None   # o'chirilgan -> sessiya yopiladi
    auth.set_active('tester', True)
    tok = auth.create_session('tester')
    auth.set_user('tester', TEST_PW + 'x')                                       # parol almashsa -> sessiyalar yopiladi
    assert auth.read_token(tok) is None
    tok = auth.create_session('tester')
    assert auth.delete_user('tester') and auth.read_token(tok) is None
    u = {x['username']: x for x in auth.list_users()}
    assert 'tester' not in u


def test_login_bruteforce_limit_is_shared_and_per_user(client):
    for _ in range(10):
        client.post('/auth/login', json={'username': 'x', 'password': 'y'}, headers={'X-Real-IP': '10.9.9.9'})
    assert client.post('/auth/login', json={'username': 'x', 'password': 'y'},
                       headers={'X-Real-IP': '10.9.9.9'}).status_code == 429
    assert auth._too_many('10.9.9.9', 'x')  # bazada — barcha workerlar uchun umumiy
    # boshqa login (o'sha IP / proksi ortida) bloklanmaydi
    auth.set_user('tester', TEST_PW)
    assert client.post('/auth/login', json={'username': 'tester', 'password': TEST_PW},
                       headers={'X-Real-IP': '10.9.9.9'}).status_code == 200
    client.cookies.clear()


def test_login_username_case_insensitive_and_last_login(client):
    auth.set_user('tester', TEST_PW)
    r = client.post('/auth/login', json={'username': ' Tester ', 'password': TEST_PW})
    assert r.status_code == 200 and r.json()['user'] == 'tester'
    assert {x['username']: x for x in auth.list_users()}['tester']['last_login_at']
    client.cookies.clear()


def test_import_users_file(tmp_path, monkeypatch):
    import json
    from src.api import db
    monkeypatch.setenv('APP_DB', str(tmp_path / 'fresh.db'))
    f = tmp_path / 'users.json'
    f.write_text(json.dumps({'Legacy': auth.hash_password(TEST_PW)}))
    monkeypatch.setenv('AUTH_USERS_FILE', str(f))
    db.migrate()
    assert auth.import_users_file() == 1 and auth.import_users_file() == 0   # faqat bir marta
    assert auth.read_token(auth.create_session('legacy')) == 'legacy'


def test_history_is_per_user(client):
    A, B = {**H, 'X-User': 'alice'}, {**H, 'X-User': 'bob'}
    client.delete('/history', headers=A)
    client.delete('/history', headers=B)
    assert client.post('/predict', json=T, headers=A).status_code == 200
    assert client.post('/predict/batch', json={'tickets': [T, T]}, headers=A).status_code == 200
    assert client.post('/predict', json={**T, 'subject': 'bob'}, headers=B).status_code == 200
    ha, hb = client.get('/history', headers=A).json(), client.get('/history', headers=B).json()
    assert ha['user'] == 'alice' and ha['total'] == 3 and hb['total'] == 1
    assert {i['source'] for i in ha['items']} == {'single', 'batch'}
    assert hb['items'][0]['subject'] == 'bob' and hb['items'][0]['result']['queue']
    assert ha['items'][0]['created_at'] >= ha['items'][-1]['created_at']  # yangisi birinchi
    # bir foydalanuvchi tozalasa, boshqasiniki qoladi
    assert client.delete('/history', headers=A).json()['deleted'] == 3
    assert client.get('/history', headers=A).json()['total'] == 0
    assert client.get('/history', headers=B).json()['total'] == 1


def test_history_requires_api_key_and_paging(client):
    assert client.get('/history').status_code == 401
    U = {**H, 'X-User': 'pager'}
    client.delete('/history', headers=U)
    client.post('/predict/batch', json={'tickets': [T] * 5}, headers=U)
    r = client.get('/history?limit=2&offset=1', headers=U).json()
    assert r['total'] == 5 and len(r['items']) == 2
    assert client.get('/history?limit=0', headers=U).status_code == 422


def test_stats_from_db(client):
    U = {**H, 'X-User': 'statuser'}
    client.delete('/history', headers=U)
    client.post('/predict/batch', json={'tickets': [T, T]}, headers=U)
    mine = client.get('/stats?mine=true', headers=U).json()
    assert mine['total'] == 2 and sum(mine['by_queue'].values()) == 2
    allst = client.get('/stats', headers=U).json()
    assert allst['total'] >= 2 and allst['users'] >= 1


def test_feedback(client):
    A, B = {**H, 'X-User': 'fb-a'}, {**H, 'X-User': 'fb-b'}
    rid = client.post('/predict', json=T, headers=A).json()['request_id']
    assert client.post(f'/history/{rid}/feedback', json={'queue': 'Nope'}, headers=A).status_code == 422
    assert client.post(f'/history/{rid}/feedback', json={'queue': 'IT Support'}, headers=B).status_code == 404
    r = client.post(f'/history/{rid}/feedback', json={'queue': 'IT Support'}, headers=A)
    assert r.status_code == 200 and r.json()['feedback']['queue'] == 'IT Support'
    assert client.get('/history', headers=A).json()['items'][0]['feedback']['queue'] == 'IT Support'
    assert client.get('/stats?mine=true', headers=A).json()['feedback'] == 1
    assert client.post(f'/history/{rid}/feedback', json={'queue': None}, headers=A).json()['feedback'] is None
