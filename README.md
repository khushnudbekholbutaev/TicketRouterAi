# TicketRouter AI — YKL, variant 23

Qo'llab-quvvatlash murojaatlarini avtomatik toifalash va yo'naltirish: ma'lumot → model → API → monitoring → adolatlilik.

## O'rnatish va to'liq takrorlash
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all        # data -> train -> evaluate -> fairness -> drift -> test  (yoki: dvc repro)
make serve      # http://127.0.0.1:8000/docs
make load       # yuk testi (boshqa terminalda)
cd docker && docker compose up -d   # api + prometheus + grafana + mlflow
```

## API misoli
```bash
curl -X POST localhost:8000/predict -H 'X-API-Key: change-me-dev-key' -H 'Content-Type: application/json' \
  -d '{"subject":"Hisob","body":"Kartamdan pul ikki marta yechib olindi, qaytarib bering"}'
```
Javob: `queue`, `confidence`, `top3`, `route_to_operator`, `threshold`, `language`, `reasons[]`, `request_id`, `model_version`.
Endpointlar: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /metrics`, `GET /model/info`.

## Natijalar (test)
F1-macro 0.509, top-3 accuracy 0.817, ECE 0.047; p95 latency 129 ms.
Hujjatlar: `docs/` (Data Card, Model Card, fairness, xavflar reyestri), hisobot: `reports/`.
