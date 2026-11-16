PY=.venv/bin/python
.PHONY: all data train evaluate fairness drift test serve load report lint
all: data train evaluate fairness drift test
data:      ; $(PY) -m src.data.ingest && $(PY) -m src.data.split
train:     ; $(PY) -m src.models.train
evaluate:  ; $(PY) -m src.models.evaluate
fairness:  ; $(PY) -m src.models.fairness
drift:     ; $(PY) -m src.monitoring.drift
test:      ; $(PY) -m pytest -q tests
serve:     ; .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
load:      ; $(PY) tests/loadtest.py 1000 20
report:    ; $(PY) make_report.py
lint:      ; ruff check src tests
