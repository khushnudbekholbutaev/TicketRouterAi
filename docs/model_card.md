# Model Card — TicketRouter AI v1.0.0

**Vazifa:** murojaatni 10 ta navbatdan biriga toifalash; ishonch past bo'lsa operatorga top-3 tavsiya bilan yuborish.
**Algoritm:** TF-IDF (so'z 1–2-gram + belgi 2–5-gram) + LogisticRegression (C = 15.91,
min_df = 1, class_weight = balanced). Optuna, 12 sinov, maqsad — val F1-macro.

| Metrika (test, n = 3295) | Qiymat |
|---|---|
| F1-macro | 0.509 |
| Accuracy | 0.536 |
| Top-3 accuracy | 0.817 |
| ECE (kalibrlash) | 0.047 |
| Yo'naltirish chegarasi | 0.85 (val da, C_wrong = 5, C_operator = 1) |
| Avtomatik yo'naltirilgan ulush / undagi aniqlik | 15.5% / 89.6% |
| p95 latency (1000 so'rov, 20 parallel) | 129 ms |

**Slice:** en F1 0.523, de 0.422, uz 0.734 (sintetik);
jismoniy 0.495, yuridik 0.591.

**Ethical considerations.** Nemis tilidagi murojaatlar sifati inglizchadan past (accuracy 0.457 vs
0.546) — nemis tilida yozgan mijozlar ko'proq operatorga tushadi (kechikish). Tilga xos chegara bilan
yo'naltirish ulushidagi farq kamaytirildi. Model hech qachon murojaatni rad etmaydi — eng yomon holatda u operatorga
boradi (human-in-the-loop). Cheklov: sintetik ma'lumotda o'qitilgan; real ishga tushirishdan oldin real murojaatlarda
qayta baholash shart. Mas'ul: loyiha jamoasi (F.I.Sh. ______).
