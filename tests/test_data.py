import pandas as pd
import pandera.errors
import pytest

from src.config import PROCESSED, QUEUES
from src.data.uz_synth import generate, to_cyr
from src.data.validate import validate


@pytest.fixture(scope='module')
def parts():
    return {s: pd.read_parquet(PROCESSED / f'{s}.parquet') for s in ('train', 'val', 'test')}


def test_splits_disjoint(parts):
    ids = [set(p['ticket_id']) for p in parts.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_proportions(parts):
    n = sum(len(p) for p in parts.values())
    assert abs(len(parts['train']) / n - 0.70) < 0.01
    assert abs(len(parts['test']) / n - 0.15) < 0.01


def test_stratification_keeps_class_shares(parts):
    a = parts['train']['queue'].value_counts(normalize=True)
    b = parts['test']['queue'].value_counts(normalize=True)
    assert (a - b).abs().max() < 0.01


def test_schema_passes_on_processed(parts):
    validate(parts['test'])


def test_schema_rejects_bad_rows(parts):
    bad = parts['test'].head(20).copy()
    bad.loc[bad.index[0], 'queue'] = 'Unknown queue'
    bad.loc[bad.index[1], 'language'] = 'fr'
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(bad)


def test_uz_generator_reproducible_and_valid():
    a, b = generate(50, seed=1), generate(50, seed=1)
    assert a.equals(b)
    assert set(a['queue']) <= set(QUEUES)
    assert to_cyr("o'zbek") == 'ўзбек'
