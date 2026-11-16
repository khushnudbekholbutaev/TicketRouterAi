"""Tushuntirish: chiziqli model uchun aniq SHAP-ekvivalent hissa (coef_k · x_j, xususiyatlar mustaqil deb).
predict + top-3 toifa + qaror sabablari (eng ko'p hissa qo'shgan so'z/n-gramlar)."""
import numpy as np


class Explainer:
    def __init__(self, pipe):
        self.pipe = pipe
        self.vec = pipe.named_steps['tfidf']
        self.clf = pipe.named_steps['clf']
        self.names = np.array(self.vec.get_feature_names_out())
        self.classes = list(self.clf.classes_)
        self.linear = hasattr(self.clf, 'coef_')

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
                        'reasons': self.reasons(X[r], k, n_reasons)})
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
