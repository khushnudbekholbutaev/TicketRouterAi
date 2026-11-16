"""Bosqich 9: data/concept drift monitoringi va drift simulyatsiyasi.
Evidently DataDriftPreset ga ekvivalent statistikalar (PSI, KS) qo'lda hisoblanadi (yengil, qo'shimcha paketsiz).
Simulyatsiya: test to'plami 6 ta "kun"ga bo'linadi; 4–6-kunlarda drift kiritiladi:
  - til aralashmasi siljishi (o'zbek, kirill murojaatlari ko'payadi),
  - yangi mavzu (QR-to'lov / kripto-hamyon) — lug'atdan tashqari so'zlar (OOV) ulushi oshadi,
  - concept drift: 6-kunda biznes qoidasi o'zgaradi (Product va IT Support -> Technical Support ga birlashtiriladi)."""
import json

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.config import FIG, MODELS, PROCESSED, REFERENCE, REPORTS, SEED  # noqa: E402
from src.data.uz_synth import generate  # noqa: E402
from src.models.train import text  # noqa: E402

ALERTS = {'psi': 0.2, 'min_drifted_features': 3, 'route_rate_ratio': 1.5, 'f1_drop': 0.05}
NEW_TOPIC = ['QR-kod orqali to\'lov o\'tmadi, kripto-hamyon balansi yangilanmadi',
             'NFT marketplace wallet connect failed, blockchain transaction pending',
             'Die Krypto-Wallet zeigt den QR-Zahlungsstatus nicht an']


def psi(ref, cur, bins=None, eps=1e-4):
    if bins is None:  # sonli: reference kvantillari bo'yicha 10 ta bin
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    else:  # kategoriyali
        r = pd.Series(ref).value_counts(normalize=True).reindex(bins, fill_value=0).values
        c = pd.Series(cur).value_counts(normalize=True).reindex(bins, fill_value=0).values
    r, c = r + eps, c + eps
    return float(np.sum((c - r) * np.log(c / r)))


def profile(pipe, df, vocab):
    t = text(df)
    p = pipe.predict_proba(t)
    toks = [s.split() for s in t]
    oov = np.array([np.mean([w not in vocab for w in ws]) if ws else 0 for ws in toks])
    return {'text_len': np.array([len(s) for s in t]), 'oov_rate': oov, 'confidence': p.max(1),
            'pred_queue': np.asarray(pipe.classes_)[p.argmax(1)], 'language': df['language'].to_numpy(dtype=object),
            'customer_type': df['customer_type'].to_numpy(dtype=object), 'script': df['script'].to_numpy(dtype=object)}


def main():
    rng = np.random.default_rng(SEED)
    pipe = joblib.load(MODELS / 'model.joblib')
    thr = json.load(open(MODELS / 'threshold.json'))['route_threshold']
    vocab = set(pipe.named_steps['tfidf'].transformer_list[0][1].vocabulary_)
    vocab = {w for v in vocab for w in v.split()}
    ref = pd.read_parquet(REFERENCE / 'reference.parquet')
    te = pd.read_parquet(PROCESSED / 'test.parquet').sample(frac=1, random_state=SEED).reset_index(drop=True)
    R = profile(pipe, ref, vocab)
    ref_route = float((R['confidence'] < thr).mean())
    days = [te.iloc[ix] for ix in np.array_split(np.arange(len(te)), 6)]
    uz_extra = generate(1500, seed=99)
    uz_extra['ticket_id'] = [f'U{i:05d}' for i in range(len(uz_extra))]
    rows = []
    for d, day in enumerate(days, 1):
        day = day.copy()
        if d >= 4:  # til siljishi: o'zbek murojaatlari ulushi ~ 40% ga, kirill ko'payadi
            k = int(len(day) * (0.25 + 0.1 * (d - 4)))
            add = uz_extra.sample(k, random_state=d).copy()
            cyr = add.sample(frac=0.6, random_state=d).index
            add.loc[cyr, 'script'] = 'cyrl'
            day = pd.concat([day, add], ignore_index=True)
        if d >= 5:  # yangi mavzu: 20% murojaatlarga notanish lug'at qo'shiladi
            m = rng.random(len(day)) < 0.2
            day.loc[m, 'body'] = day.loc[m, 'body'] + ' ' + rng.choice(NEW_TOPIC, m.sum())
        y = day['queue'].to_numpy(dtype=object).copy()
        if d == 6:  # concept drift: biznes qoidasi — Product va IT Support endi Technical Support ga birlashtirildi
            y[np.isin(y, ['Product Support', 'IT Support'])] = 'Technical Support'
        C = profile(pipe, day, vocab)
        feats = {'text_len': psi(R['text_len'], C['text_len']), 'oov_rate': psi(R['oov_rate'], C['oov_rate']),
                 'confidence': psi(R['confidence'], C['confidence']),
                 'pred_queue': psi(R['pred_queue'], C['pred_queue'], bins=list(pipe.classes_)),
                 'language': psi(R['language'], C['language'], bins=['en', 'de', 'uz']),
                 'customer_type': psi(R['customer_type'], C['customer_type'], bins=['jismoniy', 'yuridik']),
                 'script': psi(R['script'], C['script'], bins=['latn', 'cyrl'])}
        ks = {k: float(ks_2samp(R[k], C[k]).pvalue) for k in ('text_len', 'oov_rate', 'confidence')}
        route = float((C['confidence'] < thr).mean())
        f1 = f1_score(y, C['pred_queue'], average='macro', zero_division=0)
        drifted = [k for k, v in feats.items() if v > ALERTS['psi']]
        rows.append({'day': d, 'n': len(day), 'psi': feats, 'ks_pvalue': ks, 'drifted_features': drifted,
                     'route_rate': route, 'f1_macro_labels': f1})
    base_f1 = rows[0]['f1_macro_labels']
    for r in rows:
        a = []
        if len(r['drifted_features']) >= ALERTS['min_drifted_features']:
            a.append('DATA_DRIFT')
        if r['route_rate'] > ALERTS['route_rate_ratio'] * ref_route:
            a.append('ROUTE_RATE_HIGH')
        if base_f1 - r['f1_macro_labels'] > ALERTS['f1_drop']:
            a.append('CONCEPT_DRIFT_F1_DROP')
        r['alerts'] = a
        r['retrain_triggered'] = bool(a)
    out = {'thresholds': ALERTS, 'reference_route_rate': ref_route, 'days': rows}
    json.dump(out, open(REPORTS / 'drift_results.json', 'w'), indent=2, default=float)

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    names = list(rows[0]['psi'])
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#008300']
    for i, f in enumerate(names):
        axs[0].plot([r['day'] for r in rows], [r['psi'][f] for r in rows], marker='o', ms=4, lw=1.8, color=cols[i],
                    label=f)
    axs[0].axhline(ALERTS['psi'], color='#52514e', ls='--', lw=1)
    axs[0].text(1, ALERTS['psi'] * 1.15, 'PSI = 0.2 (alert)', fontsize=8, color='#52514e')
    axs[0].set_yscale('symlog', linthresh=0.05)
    axs[0].set_xlabel('Kun')
    axs[0].set_ylabel('PSI')
    axs[0].set_title('Xususiyatlar drifti (PSI)', fontsize=10, fontweight='bold')
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    axs[1].plot([r['day'] for r in rows], [r['f1_macro_labels'] for r in rows], marker='o', color='#2a78d6', lw=2,
                label='F1-macro (teglar kelgach)')
    axs[1].plot([r['day'] for r in rows], [r['route_rate'] for r in rows], marker='s', color='#eb6834', lw=2,
                label='operatorga yo\'naltirish ulushi')
    for r in rows:
        if r['alerts']:
            axs[1].annotate('\n'.join(a.split('_')[0] for a in r['alerts']), (r['day'], r['f1_macro_labels']),
                            textcoords='offset points', xytext=(0, 10), ha='center', fontsize=7, color='#e34948')
    axs[1].set_xlabel('Kun')
    axs[1].set_title('Model ko\'rsatkichlari va alertlar', fontsize=10, fontweight='bold')
    axs[1].legend(frameon=False, fontsize=8, loc='lower left')
    for a in axs:
        a.grid(color='#e4e3df')
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'drift_simulation.png', dpi=150)
    plt.close(fig)
    for r in rows:
        print(r['day'], r['n'], {k: round(v, 3) for k, v in r['psi'].items()}, round(r['route_rate'], 3),
              round(r['f1_macro_labels'], 3), r['alerts'])


if __name__ == '__main__':
    main()
