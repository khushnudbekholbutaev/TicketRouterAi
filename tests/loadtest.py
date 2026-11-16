"""Yuk testi (locust o'rniga yengil asinxron httpx): N so'rov, C parallel foydalanuvchi; p50/p95/p99, RPS.
Ishlatish: uvicorn src.api.main:app --port 8000 &  python tests/loadtest.py 1000 20"""
import asyncio
import json
import sys
import time

import httpx
import numpy as np
import pandas as pd

URL = 'http://127.0.0.1:8000/predict'
H = {'X-API-Key': 'change-me-dev-key'}


async def main(n=1000, c=20):
    df = pd.read_parquet('data/processed/test.parquet').sample(n, replace=True, random_state=1)
    payload = [{'subject': s, 'body': b[:4900]} for s, b in zip(df['subject'], df['body'])]
    lat, codes = [], []
    sem = asyncio.Semaphore(c)
    async with httpx.AsyncClient(timeout=30) as cl:
        async def one(p):
            async with sem:
                t0 = time.perf_counter()
                r = await cl.post(URL, json=p, headers=H)
                lat.append((time.perf_counter() - t0) * 1e3)
                codes.append(r.status_code)
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in payload))
        total = time.perf_counter() - t0
    a = np.array(lat)
    res = {'requests': n, 'concurrency': c, 'rps': n / total, 'p50_ms': float(np.percentile(a, 50)),
           'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)), 'mean_ms': float(a.mean()),
           'errors': int(sum(x != 200 for x in codes)), 'status_codes': {str(k): codes.count(k) for k in set(codes)}}
    print(json.dumps(res, indent=1))
    json.dump(res, open('reports/loadtest.json', 'w'), indent=2)


if __name__ == '__main__':
    asyncio.run(main(*(int(x) for x in sys.argv[1:3])))
