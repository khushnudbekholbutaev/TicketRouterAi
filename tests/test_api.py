import os

os.environ['API_KEY'] = 'test-key'
os.environ['RATE_LIMIT'] = '1000/minute'
import tempfile  # noqa: E402
os.environ['AUTH_USERS_FILE'] = os.path.join(tempfile.mkdtemp(), 'users.json')

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
    auth._save({'tester': auth.hash_password(TEST_PW)})
    assert client.get('/auth/check').status_code == 401
    assert client.post('/auth/login', json={'username': 'tester', 'password': 'wrong-password'}).status_code == 401
    assert client.post('/auth/login', json={'username': 'nobody', 'password': TEST_PW}).status_code == 401
    r = client.post('/auth/login', json={'username': 'tester', 'password': TEST_PW})
    assert r.status_code == 200 and auth.COOKIE in r.cookies
    c = client.get('/auth/check')
    assert c.status_code == 200 and c.json()['user'] == 'tester'
    client.post('/auth/logout')
    client.cookies.clear()
    assert client.get('/auth/check').status_code == 401


def test_auth_token_tamper_and_removed_user():
    auth._save({'tester': auth.hash_password(TEST_PW)})
    tok = auth.make_token('tester')
    assert auth.read_token(tok) == 'tester'
    assert auth.read_token(tok[:-1] + ('0' if tok[-1] != '0' else '1')) is None
    auth._save({})
    assert auth.read_token(tok) is None


def test_login_bruteforce_limit(client):
    auth._fails.clear()
    for _ in range(10):
        client.post('/auth/login', json={'username': 'x', 'password': 'y'}, headers={'X-Real-IP': '10.9.9.9'})
    assert client.post('/auth/login', json={'username': 'x', 'password': 'y'},
                       headers={'X-Real-IP': '10.9.9.9'}).status_code == 429
    auth._fails.clear()
