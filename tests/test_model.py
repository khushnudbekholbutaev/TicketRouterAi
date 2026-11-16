import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS, PROCESSED, REPORTS
from src.models.train import text


@pytest.fixture(scope='module')
def pipe():
    return joblib.load(MODELS / 'model.joblib')


def proba_of(pipe, t, cls):
    p = pipe.predict_proba([t.lower()])[0]
    return p[list(pipe.classes_).index(cls)]


def test_beats_baseline():
    r = json.load(open(REPORTS / 'train_results.json'))['candidates']
    assert r['lr_word_char_optuna']['f1_macro'] > r['baseline_lr']['f1_macro']


def test_test_set_quality_floor(pipe):
    te = pd.read_parquet(PROCESSED / 'test.parquet')
    p = pipe.predict_proba(text(te))
    top3 = np.argsort(-p, 1)[:, :3]
    cls = np.asarray(pipe.classes_)
    y = te['queue'].to_numpy(dtype=object)
    assert np.mean([y[i] in cls[top3[i]] for i in range(len(y))]) >= 0.70


def test_invariance_to_pii(pipe):
    a = 'My invoice shows a double charge on my credit card, please refund. Contact: john@mail.com'
    b = 'My invoice shows a double charge on my credit card, please refund. Contact: aziza@inbox.uz'
    assert pipe.predict([a.lower()])[0] == pipe.predict([b.lower()])[0]


def test_directional_billing(pipe):
    base = 'I have a question about my account.'
    more = base + ' I was charged twice, the invoice amount is wrong, please refund the payment.'
    assert proba_of(pipe, more, 'Billing and Payments') > proba_of(pipe, base, 'Billing and Payments')


def test_probabilities_valid(pipe):
    p = pipe.predict_proba(['server is down since morning'])
    assert np.isclose(p.sum(), 1.0) and (p >= 0).all()
