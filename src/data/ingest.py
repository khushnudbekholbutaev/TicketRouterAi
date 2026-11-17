"""Bosqich 1: ma'lumotlarni yuklash. Manba, yuklash sanasi va SHA-256 xesh data/raw/provenance.json ga yoziladi.
Mijoz turi (customer_type) asosiy to'plamda yo'q -> matndagi tashkilot belgilaridan PROKSI sifatida chiqariladi."""
import datetime as dt
import hashlib
import json
import re

import pandas as pd

from src.config import INTERIM, RAW_CSV, SEED
from src.data.ru_synth import generate as generate_ru
from src.data.uz_synth import generate

SOURCE_URL = ('https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/'
              'dataset-tickets-multi-lang-4-20k.csv')
ORG = re.compile(r'\b(our (company|business|organization|organisation|team|firm|clients|enterprise|store|agency)|'
                 r'company|corporate|enterprise|organization|organisation|business|firm|'
                 r'unser(e|em|en)? (unternehmen|firma|team|kunden|organisation)|unternehmen|firma|organisation|betrieb)\b',
                 re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    raw = pd.read_csv(RAW_CSV)
    df = raw[['subject', 'body', 'queue', 'language', 'type', 'priority']].copy()
    df['subject'] = df['subject'].fillna('')
    df['script'] = 'latn'
    df['customer_type'] = (df['subject'] + ' ' + df['body']).str.contains(ORG, regex=True).map({True: 'yuridik', False: 'jismoniy'})
    df['source'] = 'hf_tickets_20k'
    uz = generate(n=2000, seed=SEED)
    ru = generate_ru(n=2000, seed=SEED + 6)
    all_df = pd.concat([df, uz, ru], ignore_index=True)
    all_df.insert(0, 'ticket_id', [f'T{i:06d}' for i in range(len(all_df))])
    INTERIM.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(INTERIM, index=False)
    prov = {'source_url': SOURCE_URL, 'license': 'CC BY-NC 4.0', 'downloaded_utc': dt.datetime.fromtimestamp(
        RAW_CSV.stat().st_mtime, dt.UTC).isoformat(), 'raw_sha256': sha256(RAW_CSV), 'raw_rows': len(raw),
        'uz_synthetic_rows': len(uz), 'uz_generator': 'src/data/uz_synth.py (seed=%d)' % SEED,
        'ru_synthetic_rows': len(ru), 'ru_generator': 'src/data/ru_synth.py (seed=%d)' % (SEED + 6),
        'interim_sha256': sha256(INTERIM), 'rows_total': len(all_df)}
    json.dump(prov, open(RAW_CSV.parent / 'provenance.json', 'w'), indent=2)
    print(json.dumps(prov, indent=2))
    print(all_df.groupby(['language', 'customer_type']).size())


if __name__ == '__main__':
    main()
