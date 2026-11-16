"""Bosqich 3: baseline + nomzodlar, 5-karrali stratifikatsiyalangan CV, Optuna, MLflow tracking.
Himoyalangan atributlar (language, customer_type) modelga xususiyat sifatida berilmaydi."""
import hashlib
import json
import os
import time
import warnings

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.config import MODELS, PROCESSED, QUEUES, REPORTS, ROOT, SEED

warnings.filterwarnings('ignore')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
mlflow.set_tracking_uri('sqlite:///' + str(ROOT / 'mlflow.db'))
mlflow.set_experiment('ticket_router_v23')


def text(df):
    return (df['subject'].fillna('') + ' . ' + df['body'].fillna('')).str.lower().to_numpy(dtype=object)


def load():
    tr, va = (pd.read_parquet(PROCESSED / f'{s}.parquet') for s in ('train', 'val'))
    return tr, va


def data_hash():
    return hashlib.sha256(open(PROCESSED / 'train.parquet', 'rb').read()).hexdigest()[:16]


def word_tfidf(ngram=(1, 2), min_df=2, max_features=60000):
    return TfidfVectorizer(ngram_range=ngram, min_df=min_df, max_features=max_features, sublinear_tf=True)


def char_tfidf(max_features=80000):
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=max_features,
                           sublinear_tf=True)


def make(name, **p):
    if name == 'baseline_lr':
        return Pipeline([('tfidf', word_tfidf((1, 1))), ('clf', LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced'))])
    if name == 'cnb':
        return Pipeline([('tfidf', word_tfidf()), ('clf', ComplementNB(alpha=p.get('alpha', 0.3)))])
    if name == 'svm_word_char':
        feats = FeatureUnion([('w', word_tfidf()), ('c', char_tfidf())])
        return Pipeline([('tfidf', feats), ('clf', CalibratedClassifierCV(
            LinearSVC(C=p.get('C', 0.5), class_weight='balanced'), cv=3, method='sigmoid'))])
    if name == 'lr_word_char':
        feats = FeatureUnion([('w', word_tfidf(min_df=p.get('min_df', 2))), ('c', char_tfidf())])
        return Pipeline([('tfidf', feats), ('clf', LogisticRegression(
            C=p.get('C', 8.0), max_iter=3000, class_weight=p.get('cw', 'balanced')))])
    raise ValueError(name)


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)
    labels = list(pipe.classes_)
    pred = np.array(labels)[proba.argmax(1)]
    return {'f1_macro': f1_score(y, pred, average='macro'), 'accuracy': float((pred == y).mean()),
            'top3_acc': top_k_accuracy_score(y, proba, k=3, labels=labels)}


def log_run(name, params, metrics, pipe=None, tags=None):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({'data_sha256_16': data_hash(), **(tags or {})})


def main():
    tr, va = load()
    Xtr, ytr, Xva, yva = text(tr), tr['queue'].to_numpy(dtype=object), text(va), va['queue'].to_numpy(dtype=object)
    results = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # 1) baseline va nomzodlar: 5-fold CV (train) + val
    for name in ['baseline_lr', 'cnb', 'svm_word_char', 'lr_word_char']:
        t0 = time.time()
        pipe = make(name)
        cv = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring='f1_macro', n_jobs=2)
        pipe.fit(Xtr, ytr)
        m = evaluate(pipe, Xva, yva)
        m.update({'cv_f1_macro_mean': float(cv.mean()), 'cv_f1_macro_std': float(cv.std()),
                  'fit_s': time.time() - t0})
        results[name] = m
        log_run(name, {'model': name}, m, tags={'stage': 'candidate'})
        print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)

    # 2) Optuna: lr_word_char giperparametrlari (maqsad — val F1-macro)
    def objective(trial):
        p = {'C': trial.suggest_float('C', 0.5, 30, log=True), 'min_df': trial.suggest_int('min_df', 1, 4),
             'cw': trial.suggest_categorical('cw', ['balanced', None])}
        pipe = make('lr_word_char', **p).fit(Xtr, ytr)
        m = evaluate(pipe, Xva, yva)
        log_run(f'optuna_{trial.number:02d}', {'model': 'lr_word_char', **p}, m, tags={'stage': 'optuna'})
        return m['f1_macro']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=int(os.environ.get('N_TRIALS', 12)))
    best = study.best_params
    print('optuna best', best, round(study.best_value, 4), flush=True)
    final = make('lr_word_char', **best).fit(Xtr, ytr)
    m = evaluate(final, Xva, yva)
    results['lr_word_char_optuna'] = {**m, 'params': best}
    log_run('final_lr_word_char_optuna', {'model': 'lr_word_char', **best}, m, tags={'stage': 'staging'})
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / 'model.joblib', compress=3)
    REPORTS.mkdir(exist_ok=True)
    trials = [{'n': t.number, **t.params, 'f1_macro': t.value} for t in study.trials]
    json.dump({'candidates': results, 'optuna_trials': trials, 'best_params': best, 'classes': QUEUES},
              open(REPORTS / 'train_results.json', 'w'), indent=2, default=float)
    print('saqlandi: models/model.joblib')


if __name__ == '__main__':
    main()
