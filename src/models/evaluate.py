"""Bosqich 4: test baholash, kalibrlash, operatorga yo'naltirish chegarasi, slice-tahlil, xato tahlili.
Chegara (threshold) VAL to'plamida biznes-xarajat funksiyasi minimumi bo'yicha tanlanadi, TEST da bir marta baholanadi."""
import json
import re

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score  # noqa: E402

from src.config import FIG, MODELS, PROCESSED, REPORTS  # noqa: E402
from src.models.train import text  # noqa: E402

C_WRONG, C_OPERATOR = 5.0, 1.0  # noto'g'ri navbatga yuborish (qayta yo'naltirish + kechikish) va operator mehnati


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return e, rows


def metrics(y, proba, classes, conf=None, thr=None):
    pred = np.array(classes)[proba.argmax(1)]
    out = {'n': len(y), 'f1_macro': f1_score(y, pred, average='macro', labels=classes, zero_division=0),
           'accuracy': float((pred == y).mean()),
           'top3_acc': top_k_accuracy_score(y, proba, k=3, labels=classes)}
    if thr is not None:
        auto = conf >= thr
        out.update({'route_rate': float(1 - auto.mean()),
                    'auto_accuracy': float((pred[auto] == y[auto]).mean()) if auto.any() else float('nan')})
    return out


def cost_curve(conf, correct):
    ts = np.round(np.arange(0.0, 0.95, 0.01), 2)
    rows = []
    for t in ts:
        auto = conf >= t
        cost = (C_WRONG * ((~correct) & auto).sum() + C_OPERATOR * (~auto).sum()) / len(conf)
        rows.append({'t': float(t), 'cost': float(cost), 'coverage': float(auto.mean()),
                     'auto_acc': float(correct[auto].mean()) if auto.any() else 1.0})
    return rows


def main():
    pipe = joblib.load(MODELS / 'model.joblib')
    classes = list(pipe.classes_)
    va, te = (pd.read_parquet(PROCESSED / f'{s}.parquet') for s in ('val', 'test'))
    pv = pipe.predict_proba(text(va))
    cv_, okv = pv.max(1), np.array(classes)[pv.argmax(1)] == va['queue'].to_numpy(dtype=object)
    curve = cost_curve(cv_, okv)
    best = min(curve, key=lambda r: r['cost'])
    thr = best['t']
    json.dump({'route_threshold': thr, 'selected_on': 'val', 'C_wrong': C_WRONG, 'C_operator': C_OPERATOR,
               'val_cost': best['cost'], 'val_coverage': best['coverage']}, open(MODELS / 'threshold.json', 'w'), indent=2)

    pt = pipe.predict_proba(text(te))
    y = te['queue'].to_numpy(dtype=object)
    conf = pt.max(1)
    pred = np.array(classes)[pt.argmax(1)]
    correct = pred == y
    overall = metrics(y, pt, classes, conf, thr)
    e, rel = ece(conf, correct)
    overall['ece'] = e
    test_curve = cost_curve(conf, correct)
    overall['cost_at_thr'] = next(r['cost'] for r in test_curve if r['t'] == thr)
    overall['cost_no_routing'] = test_curve[0]['cost']

    # slice-tahlil
    slices = {}
    for col in ['language', 'customer_type', 'script', 'priority', 'type']:
        slices[col] = {str(g): metrics(y[m], pt[m], classes, conf[m], thr)
                       for g, m in ((g, (te[col] == g).to_numpy(dtype=bool)) for g in sorted(te[col].unique()))}
    per_class = {c: {'f1': float(f1_score(y == c, pred == c)), 'support': int((y == c).sum())} for c in classes}

    # xato tahlili: eng ishonchli 25 ta xato
    wrong = te.assign(pred=pred, conf=conf)[~correct].sort_values('conf', ascending=False).head(25)

    def cat(r):
        pair = {r['queue'], r['pred']}
        if pair <= {'Technical Support', 'Product Support', 'IT Support'}:
            return 'texnik toifalar chegarasi noaniq'
        if len(str(r['body'])) < 120:
            return 'qisqa / kontekst yetarli emas'
        if 'General Inquiry' in pair or 'Customer Service' in pair:
            return 'umumiy toifa (Customer Service / General) bilan aralashish'
        if re.search(r'\b(and|also|und|auch|shuningdek)\b', str(r['body']), re.I):
            return 'bir nechta niyat (multi-intent)'
        return 'yorliq shovqini / boshqa'
    wrong['category'] = wrong.apply(cat, axis=1)
    wrong[['ticket_id', 'language', 'queue', 'pred', 'conf', 'category', 'subject', 'body']].to_csv(
        REPORTS / 'error_analysis_top25.csv', index=False)

    res = {'threshold': thr, 'overall': overall, 'per_class': per_class, 'slices': slices,
           'error_categories': wrong['category'].value_counts().to_dict(), 'reliability': rel,
           'val_cost_curve': curve, 'test_cost_curve': test_curve}
    json.dump(res, open(REPORTS / 'eval_results.json', 'w'), indent=2, default=float)

    # grafiklar
    FIG.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, pred, labels=classes, normalize='true')
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    short = [c.replace(' and ', ' & ').replace('Service Outages & Maintenance', 'Outages & Maint.') for c in classes]
    ax.set_xticks(range(len(classes)), short, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > 0.5 else '#0b0b0b')
    ax.set_xlabel('Bashorat')
    ax.set_ylabel('Haqiqiy')
    ax.set_title('Chalkashlik matritsasi (test, qator bo\'yicha normallangan)', fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG / 'confusion.png', dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    xs, ys_ = [r[0] for r in rel], [r[1] for r in rel]
    axs[0].plot([0, 1], [0, 1], ls='--', color='#52514e', lw=1, label='ideal')
    axs[0].plot(xs, ys_, marker='o', color='#2a78d6', lw=2, label=f'model (ECE = {e:.3f})')
    axs[0].set_xlabel('O\'rtacha ishonch')
    axs[0].set_ylabel('Haqiqiy aniqlik')
    axs[0].set_title('Kalibrlash egri chizig\'i', fontsize=10, fontweight='bold')
    axs[0].legend(frameon=False, fontsize=8)
    ts = [r['t'] for r in test_curve]
    axs[1].plot(ts, [r['coverage'] for r in test_curve], color='#2a78d6', lw=2, label='avtomatik ulush')
    axs[1].plot(ts, [r['auto_acc'] for r in test_curve], color='#1baf7a', lw=2, label='avtomatik qismdagi aniqlik')
    axs[1].axvline(thr, color='#52514e', ls='--', lw=1)
    axs[1].text(thr + 0.01, 0.05, f'chegara = {thr:.2f}', fontsize=8)
    axs[1].set_xlabel('Ishonch chegarasi')
    axs[1].set_title('Operatorga yo\'naltirish', fontsize=10, fontweight='bold')
    axs[1].legend(frameon=False, fontsize=8)
    for a in axs:
        a.grid(color='#e4e3df')
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'calibration_routing.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'threshold': thr, **{k: round(v, 4) for k, v in overall.items()}}, indent=1))
    for col in ('language', 'customer_type'):
        print(col, {g: (round(m['f1_macro'], 3), round(m['accuracy'], 3), round(m['route_rate'], 3)) for g, m in slices[col].items()})
    print(res['error_categories'])


if __name__ == '__main__':
    main()
