# Data Card — TicketRouter AI (variant 23)

| Maydon | Qiymat |
|---|---|
| Asosiy manba | Tobi-Bueck/customer-support-tickets (HuggingFace), fayl `dataset-tickets-multi-lang-4-20k.csv` |
| Litsenziya | CC BY-NC 4.0 — faqat notijorat / o'quv maqsadida |
| SHA-256 (raw) | `9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635` |
| Yuklangan | 2026-11-12T07:14:29.244631+00:00 |
| Hajm | 20000 ta (en 11923, de 8077) + 2000 ta o'zbekcha sintetik |
| Tozalashdan keyin | 21961 ta (dublikatlar 33, bo'sh/qisqa matn 6) |
| Bo'linish | train 15372 / val 3294 / test 3295; stratifikatsiya: toifa × til; random_state=23 |
| Maqsad ustuni | `queue` — 10 ta toifa (navbat) |
| Himoyalangan atributlar | `language` (en/de/uz), `customer_type` (jismoniy/yuridik) — modelga berilmaydi |

**Yig'ish usuli.** Asosiy to'plam muallif tomonidan sun'iy (LLM yordamida) yaratilgan murojaatlar; teglar shovqinli,
toifalar (ayniqsa Technical / Product / IT Support) bir-biriga yaqin. O'zbekcha qism `src/data/uz_synth.py` shablon
generatori bilan yaratilgan (15% kirill yozuvida, 8% teg shovqini, 50% murojaatga boshqa toifadan gap qo'shilgan).

**Ma'lum kamchiliklar.** (1) Matnlar real mijozlardan emas; (2) o'zbekcha qism shablonli — undagi yuqori aniqlik real
holatni aks ettirmaydi; (3) `customer_type` asosiy to'plamda yo'q, matndagi tashkilot belgilaridan (company, Unternehmen
va h.k.) **proksi** sifatida chiqarilgan; (4) toifalar nomutanosib: eng katta 28.9%, eng kichik 1.4%.

**Foydalanish chegaralari.** Faqat o'quv loyihasi; real mijoz ma'lumotlari bilan qayta o'qitmasdan ishlab chiqarishda
qo'llanmaydi. Matnlarda PII bo'lishi mumkin — API loglarida maskalanadi.
