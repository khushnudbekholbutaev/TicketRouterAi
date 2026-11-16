"""Bosqich 5: adolatlilik tahlili (Fairlearn MetricFrame) va mitigatsiya.
Himoyalangan atributlar: til (en/de/uz) va mijoz turi (jismoniy/yuridik).
Metrikalar: guruhlar bo'yicha accuracy va F1-macro farqi / nisbati, operatorga yo'naltirish ulushi farqi
(ko'p sinfli vazifada "selection rate" analogi), avtomatik qismdagi aniqlik farqi.
Mitigatsiya: (a) pre-processing — Reweighing (til × toifa bo'yicha og'irliklar);
            (b) post-processing — guruhga xos yo'naltirish chegarasi (ThresholdOptimizer g'oyasi).
Fairlearn ExponentiatedGradient / ThresholdOptimizer faqat binar vazifani qo'llaydi, shuning uchun
ko'p sinfli analoglar qo'lda amalga oshirildi."""
import json

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fairlearn.metrics import MetricFrame  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.config import FIG, MODELS, PROCESSED, REPORTS  # noqa: E402
from src.models.train import make, text  # noqa: E402

ATTRS = ['language', 'customer_type']
LIMITS = {'accuracy_ratio_min': 0.80, 'f1_gap_max': 0.10, 'route_gap_max': 0.10}


def f1m(y, p):
    return f1_score(y, p, average='macro', zero_division=0)


def frame(y, pred, routed, sf):
    auto = ~routed
    mf = MetricFrame(metrics={'accuracy': accuracy_score, 'f1_macro': f1m}, y_true=y, y_pred=pred,
                     sensitive_features=sf)
    rr = pd.Series(routed).groupby(np.asarray(sf)).mean()
    aa = pd.Series((pred == y)[auto]).groupby(np.asarray(sf)[auto]).mean()
    g = mf.by_group.copy()
    g['route_rate'] = rr
    g['auto_accuracy'] = aa
    g['n'] = pd.Series(1, index=range(len(y))).groupby(np.asarray(sf)).sum()
    summary = {'accuracy_diff': float(mf.difference()['accuracy']), 'accuracy_ratio': float(mf.ratio()['accuracy']),
               'f1_diff': float(mf.difference()['f1_macro']), 'route_rate_diff': float(rr.max() - rr.min()),
               'auto_accuracy_diff': float(aa.max() - aa.min()),
               'overall_f1': float(mf.overall['f1_macro']), 'overall_accuracy': float(mf.overall['accuracy']),
               'overall_route_rate': float(np.mean(routed))}
    return g, summary


def group_thresholds(conf, correct, groups, target):
    """Har bir guruh uchun avtomatik qismdagi aniqlik >= target bo'ladigan eng kichik chegara (val da)."""
    thr = {}
    for gname in np.unique(groups):
        m = groups == gname
        for t in np.round(np.arange(0.0, 0.95, 0.01), 2):
            auto = m & (conf >= t)
            if auto.sum() >= 20 and correct[auto].mean() >= target:
                thr[gname] = float(t)
                break
        else:
            thr[gname] = 0.95
    return thr


def main():
    pipe = joblib.load(MODELS / 'model.joblib')
    thr = json.load(open(MODELS / 'threshold.json'))['route_threshold']
    best = json.load(open(REPORTS / 'train_results.json'))['best_params']
    tr, va, te = (pd.read_parquet(PROCESSED / f'{s}.parquet') for s in ('train', 'val', 'test'))
    classes = np.array(pipe.classes_)
    y = te['queue'].to_numpy(dtype=object)

    def predict(model, df):
        p = model.predict_proba(text(df))
        return classes[p.argmax(1)], p.max(1)

    out, variants = {}, {}
    pred, conf = predict(pipe, te)
    variants['baseline'] = (pred, conf >= thr, pipe)

    # (a) Reweighing: har bir (til, toifa) kombinatsiyasi umumiy og'irlikka teng hissa qo'shadi
    key = tr['language'] + '|' + tr['queue']
    w = (1.0 / key.map(key.value_counts())).to_numpy(dtype=float)
    w = w / w.mean()
    rw = make('lr_word_char', **best)
    rw.fit(text(tr), tr['queue'].to_numpy(dtype=object), clf__sample_weight=w)
    joblib.dump(rw, MODELS / 'model_reweighed.joblib', compress=3)
    p2, c2 = predict(rw, te)
    variants['reweighing'] = (p2, c2 >= thr, rw)

    # (b) post-processing: tilga xos chegara — barcha tillarda avtomatik aniqlik bir xil maqsadga yetsin
    pv, cv = predict(pipe, va)
    ok_v = pv == va['queue'].to_numpy(dtype=object)
    target = float(ok_v[cv >= thr].mean())
    gthr = group_thresholds(cv, ok_v, va['language'].to_numpy(dtype=object), target)
    t_vec = te['language'].map(gthr).to_numpy(dtype=float)
    variants['group_threshold'] = (pred, conf < t_vec, pipe)
    json.dump({'per_language_threshold': gthr, 'target_auto_accuracy': target},
              open(MODELS / 'group_thresholds.json', 'w'), indent=2)

    for vname, (pr, routed_or_auto, _) in variants.items():
        routed = ~routed_or_auto if vname != 'group_threshold' else routed_or_auto
        out[vname] = {}
        for a in ATTRS:
            g, s = frame(y, pr, routed, te[a].to_numpy(dtype=object))
            out[vname][a] = {'by_group': g.reset_index().rename(columns={'index': a}).to_dict('records'), 'summary': s}
    out['limits'] = LIMITS
    out['group_thresholds'] = gthr
    json.dump(out, open(REPORTS / 'fairness_results.json', 'w'), indent=2, default=float)

    # trade-off grafigi
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = {'baseline': '#2a78d6', 'reweighing': '#eb6834', 'group_threshold': '#1baf7a'}
    labels = {'baseline': 'asosiy model', 'reweighing': 'Reweighing', 'group_threshold': 'guruh chegarasi'}
    for v in variants:
        s = out[v]['language']['summary']
        axs[0].scatter(s['overall_f1'], s['accuracy_diff'], s=70, color=colors[v], edgecolor='white', zorder=3,
                       label=labels[v])
        axs[1].scatter(s['overall_route_rate'], s['auto_accuracy_diff'], s=70, color=colors[v], edgecolor='white',
                       zorder=3, label=labels[v])
    axs[0].set_xlabel('Umumiy F1-macro')
    axs[0].set_ylabel('Tillar bo\'yicha accuracy farqi')
    axs[0].set_title('Aniqlik ↔ adolatlilik', fontsize=10, fontweight='bold')
    axs[1].set_xlabel('Operatorga yo\'naltirish ulushi')
    axs[1].set_ylabel('Avtomatik qism aniqligi farqi')
    axs[1].set_title('Yo\'naltirish ↔ tenglik', fontsize=10, fontweight='bold')
    for a in axs:
        a.grid(color='#e4e3df')
        a.spines[['top', 'right']].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'fairness_tradeoff.png', dpi=150)
    plt.close(fig)
    for v in variants:
        for a in ATTRS:
            s = out[v][a]['summary']
            print(v, a, {k: round(x, 3) for k, x in s.items()})
    print('group thresholds', gthr, 'target', round(target, 3))


if __name__ == '__main__':
    main()
