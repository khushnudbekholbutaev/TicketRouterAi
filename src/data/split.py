"""Validatsiya, dublikatlarni olib tashlash va stratifikatsiyalangan bo'lish 70/15/15 (queue × language)."""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM, PROCESSED, REFERENCE, SEED
from src.data.validate import validate


def main():
    df = pd.read_parquet(INTERIM)
    n0 = len(df)
    df = df.drop_duplicates(['subject', 'body']).reset_index(drop=True)
    n1 = len(df)
    bad = df['body'].isna() | (df['body'].fillna('').str.strip().str.len() < 10)  # bo'sh / juda qisqa matn
    df = df[~bad].reset_index(drop=True)
    df = validate(df)
    strat = df['queue'] + '|' + df['language']
    tr, tmp = train_test_split(df, test_size=0.30, stratify=strat, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp['queue'] + '|' + tmp['language'], random_state=SEED)
    for name, part in (('train', tr), ('val', va), ('test', te)):
        part.to_parquet(PROCESSED / f'{name}.parquet', index=False)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    va.to_parquet(REFERENCE / 'reference.parquet', index=False)  # drift etaloni: modelga ko'rsatilmagan val to'plami
    info = {'rows_before_dedup': n0, 'rows_after_dedup': n1, 'dropped_short_or_empty': int(bad.sum()),
            'rows_clean': len(df), 'train': len(tr), 'val': len(va), 'test': len(te),
            'random_state': SEED}
    json.dump(info, open(PROCESSED / 'split_info.json', 'w'), indent=2)
    print(info)


if __name__ == '__main__':
    main()
