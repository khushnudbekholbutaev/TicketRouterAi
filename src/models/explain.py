"""Tushuntirish: chiziqli model uchun aniq SHAP-ekvivalent hissa (coef_k · x_j, xususiyatlar mustaqil deb).
predict + top-3 toifa + qaror sabablari (eng ko'p hissa qo'shgan so'zlar).

Model so'z (w__) va belgi n-gram (c__, char_wb 2-5) xususiyatlaridan iborat. Belgi n-gramlari ("lov", "ech")
o'zi foydalanuvchiga tushunarsiz, shuning uchun har bir xususiyat hissasi matndagi to'liq so'zlarga
taqsimlanadi va sabab sifatida to'liq so'z ("to'lov", "yechib") qaytariladi."""
import re
from collections import defaultdict

import numpy as np

# So'z chetidagi tinish belgilarini olib tashlash (o'zbekcha apostroflar so'z ichida qoladi)
_EDGE = re.compile(r"^[^\w'ʻʼ’`]+|[^\w'ʻʼ’`]+$")


class Explainer:
    def __init__(self, pipe):
        self.pipe = pipe
        self.vec = pipe.named_steps['tfidf']
        self.clf = pipe.named_steps['clf']
        self.names = np.array(self.vec.get_feature_names_out())
        self.classes = list(self.clf.classes_)
        self.linear = hasattr(self.clf, 'coef_')
        parts = dict(getattr(self.vec, 'transformer_list', []))
        self.word_tok = parts['w'].build_tokenizer() if 'w' in parts else None
        split = [s.split('__', 1) if '__' in s else ('', s) for s in self.names]
        self.kind = np.array([k for k, _ in split])
        self.feat = np.array([f for _, f in split], dtype=object)

    def word_reasons(self, text, X_row, k_idx, n=3):
        """Hissalarni matndagi to'liq so'zlarga yig'ish: so'z unigrami o'z so'ziga, bigram ikkala so'zga
        teng, belgi n-grami uni o'z ichiga olgan so'zlarga (char_wb so'z chegarasidan o'tmaydi) taqsimlanadi."""
        raw = text.split()
        if not raw:
            return []
        toks = [set(self.word_tok(t)) for t in raw]
        padded = [' ' + t + ' ' for t in raw]
        contrib = X_row.multiply(self.clf.coef_[k_idx]).tocsr()
        score = defaultdict(float)
        for j, v in zip(contrib.indices, contrib.data):
            kind, f = self.kind[j], self.feat[j]
            if kind == 'w':
                grams = f.split(' ')
                for g in grams:
                    hits = [i for i, s in enumerate(toks) if g in s]
                    for i in hits:
                        score[i] += v / len(grams) / len(hits)
            elif kind == 'c':
                hits = [i for i, s in enumerate(padded) for _ in range(s.count(f))]
                for i in hits:
                    score[i] += v / len(hits)
        by_word = defaultdict(float)
        for i, v in score.items():
            w = _EDGE.sub('', raw[i])
            if len(w) >= 2:
                by_word[w] += v
        best = sorted(((v, w) for w, v in by_word.items() if v > 0), reverse=True)[:n]
        return [{'feature': w, 'contribution': round(float(v), 4)} for v, w in best]

    def reasons(self, X_row, k_idx, n=3):
        if not self.linear:
            return []
        contrib = X_row.multiply(self.clf.coef_[k_idx]).tocsr()
        if contrib.nnz == 0:
            return []
        idx, val = contrib.indices, contrib.data
        order = np.argsort(-val)
        out, seen = [], set()
        for i in order:
            name = self.names[idx[i]].split('__', 1)[-1].strip()
            if val[i] <= 0 or len(name) < 3 or name in seen:
                continue
            if any(name in s or s in name for s in seen):  # char n-gram takrorlarini qisqartirish
                continue
            seen.add(name)
            out.append({'feature': name, 'contribution': round(float(val[i]), 4)})
            if len(out) == n:
                break
        return out

    def explain(self, texts, n_reasons=3):
        X = self.vec.transform(texts)
        proba = self.clf.predict_proba(X)
        res = []
        for r in range(X.shape[0]):
            order = np.argsort(-proba[r])
            k = int(order[0])
            res.append({'queue': self.classes[k], 'confidence': float(proba[r, k]),
                        'top3': [{'queue': self.classes[j], 'probability': round(float(proba[r, j]), 4)}
                                 for j in order[:3]],
                        'reasons': (self.word_reasons(texts[r], X[r], k, n_reasons)
                                    if self.linear and self.word_tok else self.reasons(X[r], k, n_reasons))})
        return res, proba

    def global_importance(self, n=8):
        """Har bir toifa uchun eng katta musbat koeffitsiyentli so'z xususiyatlari (global tushuntirish)."""
        out = {}
        word = np.array([s.startswith('w__') for s in self.names])
        for k, c in enumerate(self.classes):
            coef = np.where(word, self.clf.coef_[k], -np.inf)
            top = np.argsort(-coef)[:n]
            out[c] = [self.names[i].split('__', 1)[-1] for i in top]
        return out
