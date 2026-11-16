# Fairness hisobot

Himoyalangan atributlar: til (en/de/uz), mijoz turi (jismoniy/yuridik). Chegara qiymatlar: accuracy nisbati ≥ 0.80,
F1 farqi ≤ 0.10, yo'naltirish ulushi farqi ≤ 0.10 (talab sifatida jamoa tomonidan belgilangan).

| Variant | Til: accuracy farqi | Til: accuracy nisbati | Til: yo'naltirish farqi | Mijoz turi: accuracy farqi | Umumiy F1 |
|---|---|---|---|---|---|
| Asosiy model | 0.339 | 0.574 | 0.453 | 0.065 | 0.509 |
| Reweighing | 0.356 | 0.562 | 0.628 | 0.061 | 0.506 |
| Tilga xos chegara | 0.339 | 0.574 | 0.356 | 0.065 | 0.509 |

Tilga xos chegaralar: {'de': 0.91, 'en': 0.83, 'uz': 0.9} (maqsad: avtomatik qismdagi aniqlik ≥ 0.905).
Asosiy farq o'zbekcha sintetik qism hisobiga (u osonroq); real taqqoslanadigan juftlik en–de: accuracy
0.546 vs 0.457 (nisbat 0.837 ≥ 0.80).
