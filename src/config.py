"""Loyiha konfiguratsiyasi: yo'llar, seed, toifalar."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW_CSV = DATA / 'raw' / 'tickets_multilang_20k.csv'
INTERIM = DATA / 'processed' / 'tickets_all.parquet'
PROCESSED = DATA / 'processed'
REFERENCE = DATA / 'reference'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'
FIG = REPORTS / 'figures'
SEED = 23

QUEUES = ['Technical Support', 'Product Support', 'Customer Service', 'IT Support', 'Billing and Payments',
          'Returns and Exchanges', 'Service Outages and Maintenance', 'Sales and Pre-Sales', 'Human Resources',
          'General Inquiry']
LANGS = ['en', 'de', 'uz']
CUSTOMER_TYPES = ['jismoniy', 'yuridik']
# Ishonchsiz bashoratni operatorga yuborish chegarasi (evaluate.py da tanlanadi va threshold.json ga yoziladi)
DEFAULT_ROUTE_THRESHOLD = 0.35
