"""Bosqich 7: FastAPI xizmati — murojaatni toifalash va yo'naltirish.
Endpointlar: POST /predict, POST /predict/batch, GET /health, GET /metrics, GET /model/info,
POST /auth/login, POST /auth/logout, GET /auth/check (UI login),
GET/DELETE /history (foydalanuvchi tarixi, SQLite), GET /stats (umumiy statistika bazadan).
Xavfsizlik: X-API-Key, rate limiting (slowapi), kiritma hajmi cheklovi, loglarda PII maskalash."""
import hmac
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api import history
from src.api.auth import router as auth_router
from src.config import MODELS
from src.models.explain import Explainer

MODEL_VERSION = os.environ.get('MODEL_VERSION', '1.0.0')
API_KEY = os.environ.get('API_KEY', 'change-me-dev-key')
RATE = os.environ.get('RATE_LIMIT', '60/minute')

log = logging.getLogger('ticket_router')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

REQS = Counter('api_requests_total', 'So\'rovlar', ['endpoint', 'status'])
LAT = Histogram('api_latency_seconds', 'Kechikish', ['endpoint'],
                buckets=(.005, .01, .025, .05, .1, .2, .5, 1, 2))
PRED = Counter('predictions_total', 'Bashoratlar', ['queue', 'routed', 'language'])
CONF = Histogram('prediction_confidence', 'Bashorat ishonchi', buckets=[i / 10 for i in range(1, 11)])
TEXT_LEN = Histogram('ticket_text_length', 'Matn uzunligi (belgi)', buckets=(50, 100, 200, 400, 800, 1600, 5000))

PII = [(re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '<EMAIL>'),
       (re.compile(r'\+?\d[\d\s()-]{7,}\d'), '<PHONE>'),
       (re.compile(r'\b(?:\d[ -]?){13,19}\b'), '<CARD>')]


def mask_pii(s: str) -> str:
    for rx, rep in PII:
        s = rx.sub(rep, s)
    return s


UZ = re.compile(r"(\b(va|uchun|bilan|iltimos|kerak|yordam|murojaat|qilish|bo'yicha|emas)\b|[ўғқҳ]|o'|g')", re.I)
DE = re.compile(r'(\b(und|nicht|ist|bitte|wir|unser|ich|der|die|das|mit)\b|[äöüß])', re.I)
CYR = re.compile(r'[а-яё]', re.I)
UZ_CYR = re.compile(r'[ўғқҳ]', re.I)


def detect_language(t: str) -> str:
    letters = sum(c.isalpha() for c in t) or 1
    if len(CYR.findall(t)) / letters > 0.5 and not UZ_CYR.search(t):  # kirill, lekin o'zbekcha harflarsiz
        return 'ru'
    uz, de = len(UZ.findall(t)), len(DE.findall(t))
    if uz >= 2 and uz >= de:
        return 'uz'
    return 'de' if de >= 3 else 'en'


class Ticket(BaseModel):
    subject: str = Field('', max_length=300)
    body: str = Field(min_length=10, max_length=5000)
    language: Optional[Literal['en', 'de', 'uz', 'ru']] = None
    customer_type: Optional[Literal['jismoniy', 'yuridik']] = None


class QueueProb(BaseModel):
    queue: str
    probability: float


class Reason(BaseModel):
    feature: str
    contribution: float


class Prediction(BaseModel):
    request_id: str
    queue: str
    confidence: float
    top3: list[QueueProb]
    route_to_operator: bool
    threshold: float
    language: str
    reasons: list[Reason]
    model_version: str


class Batch(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=100)


STATE = {}


def load_artifacts():
    pipe = joblib.load(os.environ.get('MODEL_PATH', MODELS / 'model.joblib'))
    thr = json.load(open(MODELS / 'threshold.json'))['route_threshold']
    gpath = MODELS / 'group_thresholds.json'
    gthr = json.load(open(gpath))['per_language_threshold'] if gpath.exists() else {}
    STATE.update(pipe=pipe, expl=Explainer(pipe), thr=thr, gthr=gthr, loaded=time.time())


@asynccontextmanager
async def lifespan(app):
    load_artifacts()
    history.init_db()
    yield


def client_ip(request: Request) -> str:
    # nginx ortida hamma so'rov nginx IP'sidan keladi — haqiqiy IP X-Real-IP sarlavhasida
    return request.headers.get('x-real-ip') or get_remote_address(request)


limiter = Limiter(key_func=client_ip)
app = FastAPI(title='TicketRouter AI', version=MODEL_VERSION, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(auth_router)


def check_api_key(x_api_key: str = Header(default='')):
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail='API-kalit noto\'g\'ri yoki yo\'q')


def current_user(x_user: str = Header(default='')) -> str:
    # nginx /auth/check natijasidagi login'ni X-User sifatida uzatadi (mijoz yuborgani nginx'da almashtiriladi).
    # To'g'ridan-to'g'ri API-kalit bilan kelgan so'rovlar 'api' foydalanuvchisiga yoziladi.
    return x_user.strip().lower()[:64] or 'api'


def _record(user: str, source: str, items, preds):
    try:
        history.save(user, source, list(zip(items, preds)))
    except Exception:  # tarix yozilmasa ham bashorat qaytarilsin
        log.exception('history save failed')


def _predict(items: list[Ticket]) -> list[Prediction]:
    texts = [(t.subject + ' . ' + t.body).lower() for t in items]
    res, _ = STATE['expl'].explain(texts)
    out = []
    for t, r in zip(items, res):
        lang = t.language or detect_language(t.subject + ' ' + t.body)
        thr = float(STATE['gthr'].get(lang, STATE['thr'])) if os.environ.get('GROUP_THRESHOLDS', '1') == '1' else STATE['thr']
        routed = r['confidence'] < thr
        rid = uuid.uuid4().hex[:12]
        PRED.labels(queue=r['queue'], routed=str(routed), language=lang).inc()
        CONF.observe(r['confidence'])
        TEXT_LEN.observe(len(t.body))
        log.info('request_id=%s lang=%s queue=%s conf=%.3f routed=%s text="%s"', rid, lang, r['queue'],
                 r['confidence'], routed, mask_pii(t.body[:120]))
        out.append(Prediction(request_id=rid, queue=r['queue'], confidence=round(r['confidence'], 4),
                              top3=r['top3'], route_to_operator=routed, threshold=thr, language=lang,
                              reasons=r['reasons'], model_version=MODEL_VERSION))
    return out


@app.middleware('http')
async def metrics_mw(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ep = request.url.path
    LAT.labels(endpoint=ep).observe(time.perf_counter() - t0)
    REQS.labels(endpoint=ep, status=str(resp.status_code)).inc()
    return resp


@app.post('/predict', response_model=Prediction)
@limiter.limit(RATE)
def predict(request: Request, ticket: Ticket, _=Depends(check_api_key), user: str = Depends(current_user)):
    preds = _predict([ticket])
    _record(user, 'single', [ticket], preds)
    return preds[0]


@app.post('/predict/batch', response_model=list[Prediction])
@limiter.limit(RATE)
def predict_batch(request: Request, batch: Batch, _=Depends(check_api_key), user: str = Depends(current_user)):
    preds = _predict(batch.tickets)
    _record(user, 'batch', batch.tickets, preds)
    return preds


@app.get('/history')
def get_history(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
                _=Depends(check_api_key), user: str = Depends(current_user)):
    return {'user': user, **history.list_for(user, limit, offset)}


@app.delete('/history')
def delete_history(_=Depends(check_api_key), user: str = Depends(current_user)):
    return {'user': user, 'deleted': history.clear_for(user)}


@app.get('/stats')
def get_stats(mine: bool = False, _=Depends(check_api_key), user: str = Depends(current_user)):
    return history.stats(user if mine else None)


@app.get('/health')
def health():
    ok = 'pipe' in STATE
    return {'status': 'ok' if ok else 'loading', 'model_version': MODEL_VERSION}


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get('/model/info')
def model_info():
    p = STATE['pipe']
    return {'model_version': MODEL_VERSION, 'classes': list(np.asarray(p.classes_)),
            'route_threshold': STATE['thr'], 'per_language_threshold': STATE['gthr'],
            'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(STATE['loaded'])),
            'algorithm': 'TF-IDF (word 1-2 + char 2-5) + LogisticRegression'}
