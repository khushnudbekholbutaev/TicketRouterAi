import os

os.environ['API_KEY'] = 'test-key'
os.environ['RATE_LIMIT'] = '1000/minute'

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

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
