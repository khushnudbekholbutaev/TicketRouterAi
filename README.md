# TicketRouter AI

Qo'llab-quvvatlash murojaatlarini avtomatik toifalash va yo'naltirish: ma'lumot → model → API → monitoring → adolatlilik.

## O'rnatish va to'liq takrorlash
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all        # data -> train -> evaluate -> fairness -> drift -> test  (yoki: dvc repro)
make serve      # http://127.0.0.1:8000/docs
make load       # yuk testi (boshqa terminalda)
cd docker && docker compose up -d   # api + ui + prometheus + grafana + mlflow
make user NAME=admin                 # UI foydalanuvchisi (parol yashirin so'raladi)
```

## Web UI
`http://<server>:8080` — login sahifasi, bitta va ommaviy (CSV) toifalash, model ma'lumoti, jonli metrikalar.
nginx `/api/*` so'rovlarini faqat tizimga kirgan foydalanuvchilar uchun o'tkazadi va API-kalitni o'zi qo'shadi.
Foydalanuvchilar `auth/users.json` da (PBKDF2 xesh, gitga kirmaydi); `.env` ga `AUTH_SECRET` qo'shish tavsiya etiladi.

## API misoli
```bash
curl -X POST localhost:8000/predict -H 'X-API-Key: change-me-dev-key' -H 'Content-Type: application/json' \
  -d '{"subject":"Hisob","body":"Kartamdan pul ikki marta yechib olindi, qaytarib bering"}'
```
Javob: `queue`, `confidence`, `top3`, `route_to_operator`, `threshold`, `language`, `reasons[]`, `request_id`, `model_version`.
Endpointlar: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /metrics`, `GET /model/info`,
`POST /auth/login`, `POST /auth/logout`, `GET /auth/check`.
Tillar: `en`, `de`, `uz`, `ru` (avtomatik aniqlanadi yoki `language` maydonida beriladi).

## Natijalar
Test to'plami: F1-macro 0.504, accuracy 0.556, top-3 accuracy 0.831, ECE 0.078.
Rus tili (sintetik test): accuracy 0.794. O'zbek va rus ma'lumotlari sintetik — real murojaatlarda sifat pastroq bo'lishi mumkin.
Hujjatlar: `docs/` (Data Card, Model Card, fairness, xavflar reyestri), hisobot: `reports/`.
