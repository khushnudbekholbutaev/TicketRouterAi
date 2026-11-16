"""Ma'lumotlar sifati tekshiruvi (pandera): ≥ 5 qoida. Qoida buzilsa pipeline to'xtaydi."""
import pandera.pandas as pa
from pandera.pandas import Check, Column

from src.config import CUSTOMER_TYPES, LANGS, QUEUES

SCHEMA = pa.DataFrameSchema({
    'ticket_id': Column(str, unique=True),                                              # 1. noyob ID
    'subject': Column(str, nullable=False),
    'body': Column(str, [Check.str_length(10, 5000)]),                                  # 2. matn uzunligi
    'queue': Column(str, Check.isin(QUEUES)),                                           # 3. ma'lum toifalar
    'language': Column(str, Check.isin(LANGS)),                                         # 4. tillar ro'yxati
    'customer_type': Column(str, Check.isin(CUSTOMER_TYPES)),                           # 5. mijoz turi
    'priority': Column(str, Check.isin(['low', 'medium', 'high'])),                     # 6. ustuvorlik
    'type': Column(str, Check.isin(['Incident', 'Request', 'Problem', 'Change'])),      # 7. murojaat turi
    'script': Column(str, Check.isin(['latn', 'cyrl'])),
}, checks=[
    Check(lambda d: d.duplicated(['subject', 'body']).mean() <= 0.05, name='dublikatlar_ulushi<=5%'),  # 8
    Check(lambda d: d['queue'].value_counts(normalize=True).min() >= 0.005, name='har_toifa>=0.5%'),   # 9
], strict=False, coerce=True)


def validate(df):
    return SCHEMA.validate(df, lazy=True)
