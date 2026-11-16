"""reports/*.json -> qo'shimcha grafiklar, docs/*.md va YKL hisobot (Word)."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

os.chdir(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(p))  # noqa: E731
prov = J('data/raw/provenance.json')
split = J('data/processed/split_info.json')
tr = J('reports/train_results.json')
ev = J('reports/eval_results.json')
fa = J('reports/fairness_results.json')
dr = J('reports/drift_results.json')
lt = J('reports/loadtest.json')
thr = J('models/threshold.json')
gthr = J('models/group_thresholds.json')
allv = pd.read_parquet('data/processed/tickets_all.parquet')
err = pd.read_csv('reports/error_analysis_top25.csv')
FIG = 'reports/figures'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
INK2, GRID = '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
                     'axes.titlesize': 10, 'legend.frameon': False, 'figure.dpi': 150})

# ---------------- EDA grafigi
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
vc = allv.groupby(['queue', 'language']).size().unstack(fill_value=0)
vc = vc.loc[vc.sum(1).sort_values().index]
left = None
for i, lang in enumerate(['en', 'de', 'uz']):
    axs[0].barh(vc.index, vc[lang], left=left, color=C[i], label=lang, height=0.7, edgecolor='white', linewidth=1)
    left = vc[lang] if left is None else left + vc[lang]
axs[0].set_title('Toifalar taqsimoti (til bo\'yicha)')
axs[0].set_xlabel('Murojaatlar soni')
axs[0].tick_params(axis='y', labelsize=8)
axs[0].legend(fontsize=8, loc='lower right')
for i, lang in enumerate(['en', 'de', 'uz']):
    axs[1].hist(allv.loc[allv.language == lang, 'body'].str.len(), bins=40, range=(0, 1600), alpha=0.6, color=C[i],
                label=lang, density=True)
axs[1].set_title('Matn uzunligi taqsimoti')
axs[1].set_xlabel('Belgilar soni')
axs[1].set_yticks([])
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG}/eda.png')
plt.close(fig)

# ---------------- modellar taqqoslash grafigi
cand = tr['candidates']
names = {'baseline_lr': 'Baseline LR\n(word 1-gram)', 'cnb': 'ComplementNB', 'svm_word_char': 'LinearSVC\n(word+char)',
         'lr_word_char': 'LR\n(word+char)', 'lr_word_char_optuna': 'LR + Optuna\n(tanlangan)'}
fig, ax = plt.subplots(figsize=(9, 3.4))
ks = list(names)
x = range(len(ks))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], [cand[k]['f1_macro'] for k in ks], w, color=C[0], label='F1-macro (val)', zorder=3)
b2 = ax.bar([i + w / 2 for i in x], [cand[k]['top3_acc'] for k in ks], w, color=C[2], label='top-3 accuracy (val)', zorder=3)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f'{r.get_height():.3f}', ha='center', fontsize=7.5)
ax.set_xticks(list(x), [names[k] for k in ks], fontsize=8)
ax.set_ylim(0, 1)
ax.set_title('Modellarni taqqoslash (validatsiya)')
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(f'{FIG}/models.png')
plt.close(fig)

# ---------------- xarajat sezgirligi (C_wrong = 2, 3, 5)
curve = ev['test_cost_curve']
sens = {}
for cw in (2, 3, 5):
    costs = [(cw * r['coverage'] * (1 - r['auto_acc']) + (1 - r['coverage']), r) for r in curve]
    c, r = min(costs, key=lambda z: z[0])
    sens[cw] = {'t': r['t'], 'coverage': r['coverage'], 'auto_acc': r['auto_acc'], 'cost': c}

o = ev['overall']
L = {g: m for g, m in ev['slices']['language'].items()}
CT = ev['slices']['customer_type']
fb, fr, fg = fa['baseline'], fa['reweighing'], fa['group_threshold']

# ======================================================== docs/*.md
os.makedirs('docs', exist_ok=True)
open('docs/data_card.md', 'w').write(f"""# Data Card — TicketRouter AI (variant 23)

| Maydon | Qiymat |
|---|---|
| Asosiy manba | Tobi-Bueck/customer-support-tickets (HuggingFace), fayl `dataset-tickets-multi-lang-4-20k.csv` |
| Litsenziya | {prov['license']} — faqat notijorat / o'quv maqsadida |
| SHA-256 (raw) | `{prov['raw_sha256']}` |
| Yuklangan | {prov['downloaded_utc']} |
| Hajm | {prov['raw_rows']} ta (en {(allv.language == 'en').sum()}, de {(allv.language == 'de').sum()}) + {prov['uz_synthetic_rows']} ta o'zbekcha sintetik |
| Tozalashdan keyin | {split['rows_clean']} ta (dublikatlar {split['rows_before_dedup'] - split['rows_after_dedup']}, bo'sh/qisqa matn {split['dropped_short_or_empty']}) |
| Bo'linish | train {split['train']} / val {split['val']} / test {split['test']}; stratifikatsiya: toifa × til; random_state={split['random_state']} |
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
""")
open('docs/model_card.md', 'w').write(f"""# Model Card — TicketRouter AI v1.0.0

**Vazifa:** murojaatni 10 ta navbatdan biriga toifalash; ishonch past bo'lsa operatorga top-3 tavsiya bilan yuborish.
**Algoritm:** TF-IDF (so'z 1–2-gram + belgi 2–5-gram) + LogisticRegression (C = {tr['best_params']['C']:.2f},
min_df = {tr['best_params']['min_df']}, class_weight = {tr['best_params']['cw']}). Optuna, 12 sinov, maqsad — val F1-macro.

| Metrika (test, n = {o['n']}) | Qiymat |
|---|---|
| F1-macro | {o['f1_macro']:.3f} |
| Accuracy | {o['accuracy']:.3f} |
| Top-3 accuracy | {o['top3_acc']:.3f} |
| ECE (kalibrlash) | {o['ece']:.3f} |
| Yo'naltirish chegarasi | {thr['route_threshold']} (val da, C_wrong = 5, C_operator = 1) |
| Avtomatik yo'naltirilgan ulush / undagi aniqlik | {1 - o['route_rate']:.1%} / {o['auto_accuracy']:.1%} |
| p95 latency (1000 so'rov, 20 parallel) | {lt['p95_ms']:.0f} ms |

**Slice:** en F1 {L['en']['f1_macro']:.3f}, de {L['de']['f1_macro']:.3f}, uz {L['uz']['f1_macro']:.3f} (sintetik);
jismoniy {CT['jismoniy']['f1_macro']:.3f}, yuridik {CT['yuridik']['f1_macro']:.3f}.

**Ethical considerations.** Nemis tilidagi murojaatlar sifati inglizchadan past (accuracy {L['de']['accuracy']:.3f} vs
{L['en']['accuracy']:.3f}) — nemis tilida yozgan mijozlar ko'proq operatorga tushadi (kechikish). Tilga xos chegara bilan
yo'naltirish ulushidagi farq kamaytirildi. Model hech qachon murojaatni rad etmaydi — eng yomon holatda u operatorga
boradi (human-in-the-loop). Cheklov: sintetik ma'lumotda o'qitilgan; real ishga tushirishdan oldin real murojaatlarda
qayta baholash shart. Mas'ul: loyiha jamoasi (F.I.Sh. ______).
""")
open('docs/fairness_report.md', 'w').write(f"""# Fairness hisobot

Himoyalangan atributlar: til (en/de/uz), mijoz turi (jismoniy/yuridik). Chegara qiymatlar: accuracy nisbati ≥ 0.80,
F1 farqi ≤ 0.10, yo'naltirish ulushi farqi ≤ 0.10 (talab sifatida jamoa tomonidan belgilangan).

| Variant | Til: accuracy farqi | Til: accuracy nisbati | Til: yo'naltirish farqi | Mijoz turi: accuracy farqi | Umumiy F1 |
|---|---|---|---|---|---|
| Asosiy model | {fb['language']['summary']['accuracy_diff']:.3f} | {fb['language']['summary']['accuracy_ratio']:.3f} | {fb['language']['summary']['route_rate_diff']:.3f} | {fb['customer_type']['summary']['accuracy_diff']:.3f} | {fb['language']['summary']['overall_f1']:.3f} |
| Reweighing | {fr['language']['summary']['accuracy_diff']:.3f} | {fr['language']['summary']['accuracy_ratio']:.3f} | {fr['language']['summary']['route_rate_diff']:.3f} | {fr['customer_type']['summary']['accuracy_diff']:.3f} | {fr['language']['summary']['overall_f1']:.3f} |
| Tilga xos chegara | {fg['language']['summary']['accuracy_diff']:.3f} | {fg['language']['summary']['accuracy_ratio']:.3f} | {fg['language']['summary']['route_rate_diff']:.3f} | {fg['customer_type']['summary']['accuracy_diff']:.3f} | {fg['language']['summary']['overall_f1']:.3f} |

Tilga xos chegaralar: {gthr['per_language_threshold']} (maqsad: avtomatik qismdagi aniqlik ≥ {gthr['target_auto_accuracy']:.3f}).
Asosiy farq o'zbekcha sintetik qism hisobiga (u osonroq); real taqqoslanadigan juftlik en–de: accuracy
{L['en']['accuracy']:.3f} vs {L['de']['accuracy']:.3f} (nisbat {L['de']['accuracy'] / L['en']['accuracy']:.3f} ≥ 0.80).
""")
open('docs/risk_register.md', 'w').write("""# Xavflar reyestri

| № | Xavf | Ehtimol | Ta'sir | Chora | Mas'ul |
|---|---|---|---|---|---|
| 1 | Noto'g'ri navbatga yo'naltirish (mijoz kechikadi) | Yuqori | O'rta | Ishonch chegarasi, top-3 tavsiya bilan operatorga yuborish | ML muhandis |
| 2 | Tillar bo'yicha sifat farqi (de, uz) | Yuqori | O'rta | Tilga xos chegara, real uz ma'lumot yig'ish, monitoring alerti | Data scientist |
| 3 | PII oqib chiqishi (loglar) | O'rta | Yuqori | Email/telefon/karta maskalash, loglarda matnning 120 belgisi | DevOps |
| 4 | API suiiste'moli / model o'g'irlanishi | O'rta | O'rta | API-kalit, rate limit 60/daqiqa, kiritma hajmi ≤ 5000 belgi | DevOps |
| 5 | Prompt/kiritma manipulyatsiyasi (kalit so'zlar qo'shib navbatni o'zgartirish) | O'rta | Past | Ustuvorlik model emas, alohida qoida; anomal matn uzunligi monitoringi | ML muhandis |
| 6 | Data drift (yangi mahsulot, til aralashmasi) | Yuqori | O'rta | PSI/KS kunlik, alert, qayta o'qitish siyosati | ML muhandis |
| 7 | Concept drift (navbatlar tuzilmasi o'zgaradi) | O'rta | Yuqori | Teglar kelgach F1 nazorati, champion/challenger | Mahsulot egasi |
| 8 | Sintetik ma'lumot real holatni aks ettirmaydi | Yuqori | Yuqori | Pilot bosqichda real murojaatlar bilan qayta baholash | Loyiha rahbari |
| 9 | Litsenziya (CC BY-NC) tijoratda ishlatishni taqiqlaydi | Yuqori | Yuqori | Tijorat uchun o'z ma'lumotlarida qayta o'qitish | Loyiha rahbari |
""")
open('README.md', 'w').write(f"""# TicketRouter AI — YKL, variant 23

Qo'llab-quvvatlash murojaatlarini avtomatik toifalash va yo'naltirish: ma'lumot → model → API → monitoring → adolatlilik.

## O'rnatish va to'liq takrorlash
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all        # data -> train -> evaluate -> fairness -> drift -> test  (yoki: dvc repro)
make serve      # http://127.0.0.1:8000/docs
make load       # yuk testi (boshqa terminalda)
cd docker && docker compose up -d   # api + prometheus + grafana + mlflow
```

## API misoli
```bash
curl -X POST localhost:8000/predict -H 'X-API-Key: change-me-dev-key' -H 'Content-Type: application/json' \\
  -d '{{"subject":"Hisob","body":"Kartamdan pul ikki marta yechib olindi, qaytarib bering"}}'
```
Javob: `queue`, `confidence`, `top3`, `route_to_operator`, `threshold`, `language`, `reasons[]`, `request_id`, `model_version`.
Endpointlar: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /metrics`, `GET /model/info`.

## Natijalar (test)
F1-macro {o['f1_macro']:.3f}, top-3 accuracy {o['top3_acc']:.3f}, ECE {o['ece']:.3f}; p95 latency {lt['p95_ms']:.0f} ms.
Hujjatlar: `docs/` (Data Card, Model Card, fairness, xavflar reyestri), hisobot: `reports/`.
""")

# ======================================================== WORD
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(1.5), Cm(2), Cm(2)
st = doc.styles['Normal']
st.font.name, st.font.size = 'Times New Roman', Pt(12)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.15
for lvl, size in ((1, 14), (2, 12.5)):
    h = doc.styles[f'Heading {lvl}']
    h.font.name, h.font.size, h.font.bold = 'Times New Roman', Pt(size), True
    h.font.color.rgb = RGBColor(0, 0, 0)
    for k in ('w:eastAsia', 'w:ascii', 'w:hAnsi'):
        h.element.rPr.rFonts.set(qn(k), 'Times New Roman')
    h.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True


def P(text='', bold=False, italic=False, align=None, size=None, after=None):
    p = doc.add_paragraph()
    for i, part in enumerate(text.split('**')):
        if part:
            r = p.add_run(part)
            r.bold, r.italic = bold or i % 2 == 1, italic
            if size:
                r.font.size = Pt(size)
    p.alignment = {'c': WD_ALIGN_PARAGRAPH.CENTER, 'r': WD_ALIGN_PARAGRAPH.RIGHT,
                   'l': WD_ALIGN_PARAGRAPH.LEFT}.get(align, WD_ALIGN_PARAGRAPH.JUSTIFY)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def B(text):
    P(text).style = doc.styles['List Bullet']


def H(t, lvl=1):
    doc.add_heading(t, lvl)


def shade(cell, color):
    s = OxmlElement('w:shd')
    s.set(qn('w:val'), 'clear')
    s.set(qn('w:color'), 'auto')
    s.set(qn('w:fill'), color)
    cell._tc.get_or_add_tcPr().append(s)


def T(header, rows, widths, size=9.5, caption=None):
    if caption:
        P(caption, bold=True, align='l', after=2).paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(c, 'E8EEF6')
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def IMG(path, cap, w=16):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(w))
    p.paragraph_format.keep_with_next = True
    P(cap, italic=True, align='c', size=10)


def CODE(text):
    for ln in text.strip('\n').split('\n'):
        p = doc.add_paragraph()
        r = p.add_run(ln if ln else ' ')
        r.font.name, r.font.size = 'Courier New', Pt(8)
        r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Courier New')
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


f3 = lambda v: f'{v:.3f}'  # noqa: E731
pc = lambda v: f'{100 * v:.1f}%'  # noqa: E731

# ---------------- titul
for t_ in ["O'ZBEKISTON RESPUBLIKASI RAQAMLI TEXNOLOGIYALAR VAZIRLIGI",
           'MUHAMMAD AL-XORAZMIY NOMIDAGI TOSHKENT AXBOROT TEXNOLOGIYALARI UNIVERSITETI', '«Kompyuter tizimlari» kafedrasi']:
    P(t_, bold=True, align='c', size=12.5, after=2)
P('')
P("«SUN'IY INTELLEKTGA ASOSLANGAN KOMPYUTER TIZIMLARI MUHANDISLIGI» (MIS2104) fanidan", bold=True, align='c', size=12)
for _ in range(3):
    P('')
P('YAKUNIY KOMPLEKS LOYIHA', bold=True, align='c', size=18, after=6)
P('TicketRouter AI — qo\'llab-quvvatlash murojaatlarini avtomatik toifalash va yo\'naltirish tizimi', bold=True,
  align='c', size=14)
P('Variant № 23: ma\'lumot → model → API → monitoring → adolatlilik', italic=True, align='c', size=12)
P('Magistratura, 70610301 – Kompyuter injiniringi', align='c', size=12)
for _ in range(6):
    P('')
for k, v in [('Bajardi (guruh a\'zolari):', '______________________________'), ('Guruh:', '______________'),
             ('Qabul qildi:', '______________________________'), ('Sana:', '______________')]:
    P(f'**{k}** {v}', align='r', size=13)
for _ in range(4):
    P('')
P('Toshkent — 2026', bold=True, align='c', size=13)
page_break()

# ---------------- 1
H('1. Masalaning qo\'yilishi')
P("**Buyurtmachi:** ko'p tilli mijozlarga xizmat ko'rsatuvchi IT-kompaniyaning qo'llab-quvvatlash bo'limi. **Muammo:** "
  "murojaatlar (email, portal, chat) qo'lda navbatlarga taqsimlanadi; dispetcher ko'p vaqt sarflaydi, noto'g'ri "
  "navbatga tushgan murojaat qayta yo'naltiriladi va mijoz kutib qoladi. **Maqsad:** murojaat matni asosida uni 10 ta "
  "navbatdan biriga (Technical Support, Billing and Payments, IT Support va h.k.) avtomatik yo'naltiruvchi, "
  "ishonchsiz holatda esa operatorga top-3 tavsiya bilan yuboruvchi va CRM ga REST API orqali ulanuvchi AI-xizmat.")
T(['Mezon', 'Talab', 'Natija (test)'], [
    ['Vazifa turi', "Ko'p sinfli klassifikatsiya (NLP), 10 sinf", '—'],
    ['Asosiy metrikalar', 'F1-macro, top-3 accuracy', f"F1-macro {f3(o['f1_macro'])}, top-3 {f3(o['top3_acc'])}"],
    ['Ishonchsiz bashorat', "Operatorga yuborish (human-in-the-loop)", f"chegara {thr['route_threshold']}, avtomatik qismda "
     f"aniqlik {pc(o['auto_accuracy'])}"],
    ['Himoyalangan atributlar', 'Til; mijoz turi (jismoniy/yuridik)', 'MetricFrame, 2 ta mitigatsiya'],
    ['Tillar bo\'yicha tenglik', 'Accuracy nisbati ≥ 0.80 (en–de)', f"{L['de']['accuracy'] / L['en']['accuracy']:.3f}"],
    ['Xizmat', 'p95 ≤ 200 ms (20 parallel mijoz)', f"p95 = {lt['p95_ms']:.0f} ms, {lt['rps']:.0f} so'rov/s"],
    ['Drift', 'Yangi toifalar va til aralashmasi drifti', 'PSI/KS + 3 alert, simulyatsiya'],
], [4.0, 6.0, 6.2], 9.5, '1-jadval. Talablar va bajarilishi')
P("**Cheklovlar:** loglarda PII saqlanmaydi; har bir yo'naltirish qarori tushuntiriladi (reasons); model har chorakda "
  "qayta baholanadi. Loyiha bitta noutbukda (Apple M1 Pro, CPU) bajarildi, shuning uchun yengil chiziqli modellar "
  "tanlandi: ular ~10 ms da javob beradi va to'liq tushuntiriladi.")

H('2. Tizim arxitekturasi')
T(['№', 'Qatlam', 'Loyihada amalga oshirilishi'], [
    ['1', "Ma'lumotlar", 'src/data/ingest.py (SHA-256, provenance.json), uz_synth.py, validate.py (pandera, 9 qoida), '
     'split.py, dvc.yaml, Data Card'],
    ['2', 'Xususiyatlar', 'sklearn Pipeline: TF-IDF so\'z 1–2-gram + belgi 2–5-gram (FeatureUnion); fit faqat train da'],
    ['3', 'Model', 'Baseline + 3 nomzod, 5-fold CV, Optuna (12 sinov), MLflow (17 run, SQLite), joblib artefakt'],
    ['4', 'Baholash / adolatlilik', 'evaluate.py (kalibrlash, chegara, slice, xato tahlili), fairness.py (Fairlearn), explain.py'],
    ['5', 'Xizmat', 'FastAPI: 5 endpoint, Pydantic, X-API-Key, slowapi rate limit, PII maskalash, Docker (non-root, HEALTHCHECK)'],
    ['6', 'CI/CD', 'GitHub Actions: ruff → pytest (21 test) → Docker build/push GHCR → release'],
    ['7', 'Monitoring', 'Prometheus metrikalari, Grafana dashboard (8 panel), 5 alert qoidasi, drift.py (PSI/KS)'],
    ['8', 'Boshqaruv', 'README, Model Card, Data Card, fairness hisobot, xavflar reyestri'],
], [0.8, 3.4, 12.0], 9, '2-jadval. Qatlamlar va ularning amalga oshirilishi')
CODE('''
CRM / portal --HTTPS--> [FastAPI /predict] --> TF-IDF + LR --> {queue, top3, confidence, reasons}
                              |  X-API-Key, rate limit         |
                              |                                +--> confidence < chegara --> operator navbati
                              +--> /metrics --> Prometheus --> Grafana / Alertmanager --> retrain trigger
MLflow (eksperimentlar, registry)  <--  train.py / evaluate.py / fairness.py  <--  DVC (data versiyalari)
''')

# ---------------- 3 data
H("3. Bosqich 1–2: ma'lumotlar va xususiyatlar")
P(f"**Manba:** HuggingFace «Tobi-Bueck/customer-support-tickets» (fayl dataset-tickets-multi-lang-4-20k.csv, "
  f"{prov['raw_rows']} ta murojaat, ingliz va nemis tillarida, litsenziya {prov['license']}), SHA-256 "
  f"{prov['raw_sha256'][:16]}… Variant talabi bo'yicha o'zbek tilidagi murojaatlar qo'shildi: real o'zbekcha ochiq "
  f"to'plam yo'qligi sababli {prov['uz_synthetic_rows']} ta murojaat shablon generatori bilan yaratildi (lotin va 15% "
  "kirill, 8% teg shovqini, yarmida boshqa toifadan qo'shimcha gap). **Mijoz turi** asosiy to'plamda yo'q, shuning "
  "uchun u matndagi tashkilot belgilaridan (company, our team, Unternehmen, Firma…) proksi sifatida chiqarildi — bu "
  "cheklov Data Card ga yozilgan.")
IMG(f'{FIG}/eda.png', '1-rasm. Toifalar va matn uzunligi taqsimoti (EDA)', 16.5)
P(f"**EDA xulosalari:** toifalar kuchli nomutanosib (Technical Support 28.9%, General Inquiry 1.4%) — shuning uchun "
  "F1-macro va class_weight='balanced' tanlandi; matn uzunligi mediana 322 belgi; o'zbekcha matnlar qisqaroq "
  f"(shablonli). Tozalash: {split['rows_before_dedup'] - split['rows_after_dedup']} ta dublikat va "
  f"{split['dropped_short_or_empty']} ta bo'sh/juda qisqa matn («Help», «Notfall») olib tashlandi — ularni pandera "
  "sxemasi birinchi ishga tushirishdayoq aniqladi. Bo'linish: train/val/test = "
  f"{split['train']}/{split['val']}/{split['test']}, toifa × til bo'yicha stratifikatsiya, random_state = 23.")
T(['№', 'Validatsiya qoidasi (pandera)'], [[str(i + 1), r] for i, r in enumerate([
    'ticket_id noyob', 'body uzunligi 10…5000 belgi', 'queue — 10 ta ma\'lum toifadan biri', 'language ∈ {en, de, uz}',
    'customer_type ∈ {jismoniy, yuridik}', 'priority ∈ {low, medium, high}', 'type ∈ {Incident, Request, Problem, Change}',
    'dublikatlar ulushi ≤ 5%', 'har bir toifa ulushi ≥ 0.5%'])], [1.0, 15.2], 9.5, '3-jadval. Ma\'lumotlar sifati qoidalari')

# ---------------- 4 model
H('4. Bosqich 3: model o\'qitish va eksperimentlar')
rows = []
for k in ks:
    m = cand[k]
    rows.append([names[k].replace('\n', ' '), f3(m.get('cv_f1_macro_mean', float('nan'))) if 'cv_f1_macro_mean' in m else '—',
                 f3(m['f1_macro']), f3(m['accuracy']), f3(m['top3_acc'])])
T(['Model', 'CV F1-macro (5-fold)', 'Val F1-macro', 'Val accuracy', 'Val top-3'], rows, [5.4, 2.8, 2.6, 2.6, 2.6], 9.5,
  '4-jadval. Baseline va nomzodlar (val to\'plam)')
IMG(f'{FIG}/models.png', '2-rasm. Modellarni taqqoslash', 15.5)
bp = tr['best_params']
P(f"Har bir run MLflow ga yozildi (parametrlar, metrikalar, ma'lumot xeshi tegi): jami 17 run — 4 nomzod, 12 Optuna "
  f"sinovi va yakuniy model. Optuna (TPE, seed 23) eng yaxshi parametrlari: C = {bp['C']:.2f}, min_df = {bp['min_df']}, "
  f"class_weight = {bp['cw']}; val F1-macro = {cand['lr_word_char_optuna']['f1_macro']:.3f} (baseline "
  f"{cand['baseline_lr']['f1_macro']:.3f} dan +{cand['lr_word_char_optuna']['f1_macro'] - cand['baseline_lr']['f1_macro']:.3f}). "
  "Belgi n-gramlari ko'p tilli matnda (nemis qo'shma so'zlari, o'zbek qo'shimchalari, kirill) eng katta foyda berdi. "
  "Mutlaq F1 qiymati o'rtacha: to'plam teglari shovqinli va toifalar semantik jihatdan kesishadi (5-bo'lim, xato tahlili).")

# ---------------- 5 evaluation
H('5. Bosqich 4: baholash va chegara tanlash')
T(['Metrika (test, n = %d)' % o['n'], 'Qiymat'], [
    ['F1-macro', f3(o['f1_macro'])], ['Accuracy (top-1)', f3(o['accuracy'])], ['Top-3 accuracy', f3(o['top3_acc'])],
    ['ECE (kalibrlash xatosi)', f3(o['ece'])], ["Yo'naltirish chegarasi (val da tanlangan)", str(thr['route_threshold'])],
    ["Avtomatik yo'naltirilgan ulush", pc(1 - o['route_rate'])], ['Avtomatik qismdagi aniqlik', pc(o['auto_accuracy'])],
    ["O'rtacha xarajat / murojaat: yo'naltirishsiz → chegara bilan", f"{o['cost_no_routing']:.2f} → {o['cost_at_thr']:.2f}"],
], [10.2, 6.0], 9.5, '5-jadval. Test natijalari')
IMG(f'{FIG}/confusion.png', '3-rasm. Chalkashlik matritsasi', 12.5)
IMG(f'{FIG}/calibration_routing.png', '4-rasm. Kalibrlash egri chizig\'i va operatorga yo\'naltirish egri chizig\'i', 16)
P(f"**Biznes-metrika va chegara.** Xarajat = C_wrong · (avtomatik, lekin noto'g'ri) + C_operator · (operatorga "
  f"yuborilgan), C_wrong = 5 (qayta yo'naltirish va mijoz kutishi), C_operator = 1. Val to'plamda minimum chegara "
  f"{thr['route_threshold']} da: test da xarajat {o['cost_no_routing']:.2f} dan {o['cost_at_thr']:.2f} ga kamayadi "
  f"(−{100 * (1 - o['cost_at_thr'] / o['cost_no_routing']):.0f}%). Bu sifatda model asosan **dispetcher yordamchisi** "
  f"bo'lib ishlaydi: murojaatlarning {pc(1 - o['route_rate'])} i {pc(o['auto_accuracy'])} aniqlik bilan avtomatik "
  f"yo'naltiriladi, qolganlari operatorga top-3 tavsiya bilan boradi (top-3 aniqlik {pc(o['top3_acc'])}). Xarajatlar "
  "nisbatiga sezgirlik:")
T(['C_wrong / C_operator', 'Optimal chegara', 'Avtomatik ulush', 'Undagi aniqlik'],
  [[f'{cw} : 1', f"{s['t']:.2f}", pc(s['coverage']), pc(s['auto_acc'])] for cw, s in sens.items()],
  [4.0, 4.0, 4.0, 4.2], 9.5, '6-jadval. Chegaraning xarajat nisbatiga sezgirligi (test)')
pcl = sorted(ev['per_class'].items(), key=lambda kv: -kv[1]['f1'])
T(['Toifa', 'F1', 'Soni (test)'], [[k, f3(v['f1']), v['support']] for k, v in pcl], [8.0, 3.0, 3.0], 9.5,
  '7-jadval. Toifalar bo\'yicha F1')
sl = []
for col, nm in (('language', 'Til'), ('customer_type', 'Mijoz turi'), ('script', 'Yozuv'), ('priority', 'Ustuvorlik'),
                ('type', 'Murojaat turi')):
    for g, m in ev['slices'][col].items():
        sl.append([f'{nm}: {g}', m['n'], f3(m['f1_macro']), f3(m['accuracy']), f3(m['top3_acc']), pc(m['route_rate'])])
T(['Slice', 'n', 'F1-macro', 'Accuracy', 'Top-3', "Operatorga"], sl, [4.6, 1.6, 2.2, 2.2, 2.2, 2.4], 9,
  '8-jadval. Slice-tahlil (test)')
P(f"**Eng yomon slice** — nemis tili (F1 {f3(L['de']['f1_macro'])}). Murojaat turi va ustuvorlik bo'yicha farq kichik "
  "(±0.03). O'zbekcha va kirill slice'larining yuqori natijasi sintetik shablonlar hisobiga — real ko'rsatkich emas.")
cats = err['category'].value_counts()
P(f"**Xato tahlili** (reports/error_analysis_top25.csv): eng ishonchli 25 ta noto'g'ri bashorat qo'lda ko'rib "
  "chiqildi va toifalandi:")
T(['Sabab toifasi', 'Soni'], [[k, int(v)] for k, v in cats.items()], [12.0, 3.0], 9.5, '9-jadval. Xato sabablari')
ex = err.head(4)
T(['Til', 'Haqiqiy', 'Bashorat', 'Ishonch', 'Matn (qisqa)'],
  [[r.language, r.queue, r.pred, f'{r.conf:.2f}', str(r.body)[:110] + '…'] for r in ex.itertuples()],
  [1.0, 3.0, 3.0, 1.5, 7.7], 8, '10-jadval. Ishonchli xatolarga misollar')
P("Asosiy xulosa: xatolarning yarmidan ko'pi texnik toifalar (Technical / Product / IT Support) va umumiy toifalar "
  "(Customer Service / General Inquiry) chegarasining noaniqligidan — ko'p hollarda matn ikkala toifaga ham mos keladi, "
  "ya'ni bu teg shovqini. Taklif: bu uchta navbat uchun ierarxik yo'naltirish (avval «texnik»/«moliyaviy»/«umumiy», "
  "keyin ichki navbat) va teglash yo'riqnomasini aniqlashtirish.")
P("**Model testlari** (tests/test_model.py): metrik baseline dan yuqori; top-3 ≥ 0.70; PII invariantligi (email "
  "o'zgarsa bashorat o'zgarmaydi); yo'naltirilgan kutish (to'lov so'zlari qo'shilsa Billing ehtimoli oshadi); "
  "ehtimolliklar yig'indisi 1.")

# ---------------- 6 fairness
H('6. Bosqich 5: adolatlilik va tushuntirish')
fr_rows = []
for v, nm in (('baseline', 'Asosiy model'), ('reweighing', 'Reweighing (pre)'), ('group_threshold', 'Tilga xos chegara (post)')):
    s, c = fa[v]['language']['summary'], fa[v]['customer_type']['summary']
    fr_rows.append([nm, f3(s['overall_f1']), f3(s['accuracy_diff']), f3(s['accuracy_ratio']), f3(s['route_rate_diff']),
                    f3(c['accuracy_diff']), f3(c['route_rate_diff'])])
T(['Variant', 'F1', 'Til: acc farqi', 'Til: acc nisbati', "Til: yo'naltirish farqi", 'Mijoz: acc farqi',
   "Mijoz: yo'naltirish farqi"], fr_rows, [3.8, 1.5, 2.0, 2.0, 2.4, 2.1, 2.4], 8.5,
  '11-jadval. Fairness metrikalari (Fairlearn MetricFrame, test)')
gb = fa['baseline']['language']['by_group']
T(['Til', 'n', 'Accuracy', 'F1-macro', "Operatorga", 'Avtomatik aniqlik'],
  [[g['sensitive_feature_0'], g['n'], f3(g['accuracy']), f3(g['f1_macro']), pc(g['route_rate']), pc(g['auto_accuracy'])]
   for g in gb], [2.0, 2.0, 2.8, 2.8, 3.0, 3.6], 9.5, '12-jadval. Tillar bo\'yicha guruh ko\'rsatkichlari (asosiy model)')
IMG(f'{FIG}/fairness_tradeoff.png', '5-rasm. Aniqlik–adolatlilik kompromissi', 16)
P(f"**Tahlil.** Mijoz turi bo'yicha model talablarga mos: accuracy farqi {f3(fb['customer_type']['summary']['accuracy_diff'])}, "
  f"nisbat {f3(fb['customer_type']['summary']['accuracy_ratio'])} (≥ 0.80). Til bo'yicha umumiy farq katta "
  f"({f3(fb['language']['summary']['accuracy_diff'])}), lekin u o'zbekcha sintetik qismning «osonligi» hisobiga; "
  f"real taqqoslanadigan en–de juftligida accuracy {f3(L['en']['accuracy'])} va {f3(L['de']['accuracy'])}, nisbat "
  f"{L['de']['accuracy'] / L['en']['accuracy']:.3f} — 80% qoidasi bajariladi, lekin nemis tilida yozuvchilar "
  f"operatorga ko'proq tushadi ({pc(L['de']['route_rate'])} vs {pc(L['en']['route_rate'])}), ya'ni ko'proq kutadi. "
  "**Mitigatsiya:** (a) Reweighing (til × toifa bo'yicha og'irlik) aniqlik farqini kamaytirmadi — farq ma'lumot "
  "hajmidan emas, nemis matnlarining o'zidan (qo'shma so'zlar, kam takrorlanuvchi lug'at) kelib chiqadi; (b) tilga "
  f"xos yo'naltirish chegarasi {gthr['per_language_threshold']} (barcha tillarda avtomatik qismdagi aniqlik ≈ "
  f"{pc(gthr['target_auto_accuracy'])}) yo'naltirish ulushidagi farqni {f3(fb['language']['summary']['route_rate_diff'])} → "
  f"{f3(fg['language']['summary']['route_rate_diff'])} ga, mijoz turi bo'yicha {f3(fb['customer_type']['summary']['route_rate_diff'])} → "
  f"{f3(fg['customer_type']['summary']['route_rate_diff'])} ga kamaytirdi; umumiy F1 o'zgarmadi. API da (b) yoqilgan. "
  "Tilni qaror uchun ishlatish bu yerda asosli: u operator tilini tanlashga ham kerak va mijozga zarar keltirmaydi "
  "(eng yomon holat — operatorga yuborish). Fairlearn ExponentiatedGradient/ThresholdOptimizer faqat binar "
  "vazifani qo'llagani uchun ko'p sinfli analoglar qo'lda yozildi.")
P(f"**Tushuntirish (SHAP-ekvivalent).** Chiziqli model uchun xususiyat hissasi φ_j = w_kj · x_j aniq hisoblanadi "
  "(xususiyatlar mustaqil deb olinganda SHAP qiymatiga teng). API har javobda «reasons» maydonida 3 ta asosiy so'z/"
  "n-gramni qaytaradi. Masalan, «Kartamdan 50000 so'm ikki marta yechib olindi…» uchun: Billing and Payments (0.91), "
  "sabablar: «marta yechib», «yechib olindi», «so ikki». Global muhimlik (har toifa uchun eng katta koeffitsiyentli "
  "so'zlar) explain.py → global_importance() da.")

# ---------------- 7 API
H('7. Bosqich 6–7: paketlash, API va konteynerlash')
P("Yakuniy pipeline models/model.joblib (9 MB), versiya 1.0.0; yonida threshold.json, group_thresholds.json, "
  "reference.parquet (drift etaloni) saqlanadi. MLflow tracking — SQLite (mlflow.db); fayl-store MLflow 3 da "
  "eskirgani uchun SQLite tanlandi.")
T(['Endpoint', 'Vazifasi'], [
    ['POST /predict', 'Bitta murojaat: queue, confidence, top3, route_to_operator, threshold, language, reasons[], request_id'],
    ['POST /predict/batch', '1…100 ta murojaat'], ['GET /health', 'Holat va model versiyasi'],
    ['GET /metrics', 'Prometheus: so\'rovlar, latency gistogrammasi, bashoratlar (toifa/til/yo\'naltirish), ishonch, matn uzunligi'],
    ['GET /model/info', 'Versiya, toifalar, chegaralar, yuklangan vaqt'],
], [4.0, 12.2], 9.5, '13-jadval. API endpointlari')
P("**Xavfsizlik:** X-API-Key (hmac.compare_digest), slowapi rate limit 60/daqiqa (sinov: 6-so'rovdan boshlab 429), "
  "body 10…5000 belgi va batch ≤ 100 (422), loglarda email/telefon/karta raqami maskalanadi (log namunasi: "
  "«…Tel: <PHONE>»). **Docker:** python:3.12-slim, non-root user, HEALTHCHECK; docker-compose: api + prometheus + "
  "grafana + mlflow. Docker bu noutbukda o'rnatilmagani uchun konteyner yig'ilmadi — API uvicorn bilan to'g'ridan-"
  "to'g'ri sinaldi; Dockerfile CI (GitHub Actions) da yig'iladi.")
T(['Ko\'rsatkich', 'Qiymat'], [['So\'rovlar / parallel mijozlar', f"{lt['requests']} / {lt['concurrency']}"],
                              ['Throughput', f"{lt['rps']:.0f} so'rov/s"], ['p50 / p95 / p99', f"{lt['p50_ms']:.0f} / {lt['p95_ms']:.0f} / {lt['p99_ms']:.0f} ms"],
                              ['Xatolar', str(lt['errors'])]], [8.0, 8.2], 9.5,
  '14-jadval. Yuk testi (uvicorn, 2 worker, tests/loadtest.py)')
P(f"Talab p95 ≤ 200 ms bajarildi (p95 = {lt['p95_ms']:.0f} ms). **Testlar:** 21 ta pytest (data — 6, model — 5, "
  "API — 10): split kesishmasligi, stratifikatsiya, sxema xato qatorni rad etishi, generator takrorlanuvchanligi, "
  "kontrakt, 401/422 kodlari, batch chegarasi, /metrics, PII maskalash, tilni aniqlash — barchasi o'tdi; ruff lint toza.")

# ---------------- 8 CI
H('8. Bosqich 8: CI/CD va takrorlanuvchanlik')
B("GitHub Actions (.github/workflows/ci.yml): ruff → dvc pull (yoki ingest+split) → pytest → Docker build va GHCR "
  "ga push (main) → tag da model artefaktlarini release ga biriktirish.")
B("Makefile (make all) va dvc.yaml: ingest → split → train → evaluate → fairness → drift bosqichlari bitta buyruq "
  "bilan takrorlanadi; requirements.txt da barcha versiyalar qat'iy; .env.example; barcha skriptlarda seed = 23.")

# ---------------- 9 monitoring
H('9. Bosqich 9: monitoring, drift va qayta o\'qitish')
P("**Tizim va model metrikalari** /metrics orqali Prometheus ga beriladi; Grafana dashboard (8 panel): so'rovlar/s, "
  "latency p50/p95/p99, xatolar ulushi, toifalar taqsimoti, tillar bo'yicha yo'naltirish ulushi (real vaqtda "
  "adolatlilik), ishonch taqsimoti, matn uzunligi, drift PSI. **Alert qoidalari** (docker/alert_rules.yml): "
  "p95 > 500 ms; 5xx > 2%; yo'naltirish ulushi > 95%; tillar bo'yicha yo'naltirish farqi > 0.5; PSI > 0.2 kamida 3 ta "
  "xususiyatda.")
P("**Drift simulyatsiyasi** (src/monitoring/drift.py): etalon — val to'plam; test 6 ta «kun»ga bo'lindi. 4-kundan "
  "boshlab o'zbek (60% kirill) murojaatlari ulushi oshiriladi, 5-kundan 20% murojaatlarga yangi mavzu (QR-to'lov, "
  "kripto-hamyon) qo'shiladi, 6-kunda biznes qoidasi o'zgaradi: Product va IT Support Technical Support ga "
  "birlashtiriladi (concept drift). Kuzatiladi: PSI (7 xususiyat) va KS p-qiymati.")
dr_rows = [[r['day'], r['n'], f"{r['psi']['language']:.2f}", f"{r['psi']['script']:.2f}", f"{r['psi']['oov_rate']:.2f}",
            f"{r['psi']['text_len']:.2f}", pc(r['route_rate']), f3(r['f1_macro_labels']), ', '.join(r['alerts']) or '—']
           for r in dr['days']]
T(['Kun', 'n', 'PSI til', 'PSI yozuv', 'PSI OOV', 'PSI uzunlik', "Operatorga", 'F1 (teglar)', 'Alertlar'], dr_rows,
  [1.0, 1.2, 1.5, 1.6, 1.5, 1.7, 1.8, 1.7, 4.2], 8.5, '15-jadval. Drift simulyatsiyasi natijalari')
IMG(f'{FIG}/drift_simulation.png', '6-rasm. Drift simulyatsiyasi: PSI va model ko\'rsatkichlari', 16.5)
P("4-kunda ikki xususiyat (til, yozuv) chegaradan o'tdi — erta ogohlantirish, lekin alert uchun 3 ta kerak; 5-kunda "
  "DATA_DRIFT alerti ishladi; 6-kunda teglar kelgach F1 0.51 → 0.41 ga tushdi va CONCEPT_DRIFT_F1_DROP alerti ishladi. "
  "Qiziq kuzatuv: til drifti paytida yo'naltirish ulushi hatto kamaydi (o'zbekcha shablonli matnlarda model ishonchli) — "
  "shuning uchun faqat ishonch/yo'naltirish ulushiga tayanib bo'lmaydi, kirish taqsimoti (PSI) ham kuzatilishi shart.")
P("**Qayta o'qitish siyosati:** alert (DATA_DRIFT yoki F1 pasayishi > 0.05) → `make all` yangi ma'lumot bilan → yangi "
  "model MLflow da «Staging» → test va fairness talablari (F1 ≥ champion − 0.01, en–de accuracy nisbati ≥ 0.80, "
  "pytest yashil) bajarilsa «Production» (champion/challenger), aks holda rad etiladi va eski model qoladi.")

# ---------------- 10 security
H('10. Bosqich 10: xavfsizlik, maxfiylik va xavflar')
B("Tahdidlar: kiritma manipulyatsiyasi (kalit so'zlar bilan navbatni o'zgartirish), ko'p so'rov orqali model "
  "o'g'irlanishi, loglardan ma'lumot oqib chiqishi, a'zolik hujumi (membership inference).")
B("Choralar: API-kalit, rate limit, kiritma validatsiyasi, PII maskalash, TLS (reverse-proxy), modelga faqat matn "
  "beriladi (himoyalangan atributlar emas).")
B("Huquqiy jihatlar: O'zbekiston Respublikasining «Shaxsga doir ma'lumotlar to'g'risida»gi qonuni; avtomatik qaror "
  "faqat yo'naltirish — mijozga rad etilmaydi, istalgan murojaat operator tomonidan qayta ko'rilishi mumkin; to'plam "
  "litsenziyasi CC BY-NC — tijorat uchun o'z ma'lumotlarida qayta o'qitish shart.")
risk = [ln.split('|')[1:-1] for ln in open('docs/risk_register.md').read().splitlines()[4:]]
T(['№', 'Xavf', 'Ehtimol', 'Ta\'sir', 'Chora'], [[c.strip() for c in r[:5]] for r in risk], [0.8, 5.0, 1.6, 1.6, 7.2],
  8.5, '16-jadval. Xavflar reyestri (to\'liq — docs/risk_register.md)')

# ---------------- 11
H('11. Xulosa')
P(f"To'liq hayotiy sikldagi AI-xizmat qurildi: versiyalangan va validatsiyalangan ma'lumotlar, takrorlanuvchi pipeline, "
  f"17 ta MLflow run, F1-macro {f3(o['f1_macro'])} va top-3 aniqlik {f3(o['top3_acc'])} li model, xarajatga asoslangan "
  f"operatorga yo'naltirish (xarajat −{100 * (1 - o['cost_at_thr'] / o['cost_no_routing']):.0f}%), tillar va mijoz "
  f"turi bo'yicha adolatlilik tahlili va mitigatsiya, p95 = {lt['p95_ms']:.0f} ms li xavfsiz API, 21 ta test, CI/CD, "
  "Prometheus/Grafana monitoringi va drift simulyatsiyasi.")
P("**Cheklovlar va kelgusi ish:** (1) to'plam sintetik va teglari shovqinli — real murojaatlarda qayta baholash; "
  "(2) o'zbekcha qism shablonli — real o'zbek/rus murojaatlarini yig'ish va teglash; (3) nemis tili sifati — "
  "ko'p tilli transformer (masalan, XLM-R distil) GPU stendda sinash; (4) ierarxik yo'naltirish; (5) Docker stack va "
  "Grafana ni haqiqiy serverda ishga tushirib, 5 daqiqalik demo-video yozish.")

H('Foydalanilgan adabiyotlar')
for ref in ['Tobi-Bueck. Customer Support Tickets dataset. HuggingFace, 2025. CC BY-NC 4.0.',
            'Pedregosa F. et al. Scikit-learn: Machine Learning in Python // JMLR, 2011, 12.',
            'Bird S. et al. Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft, 2020.',
            'Lundberg S., Lee S.-I. A Unified Approach to Interpreting Model Predictions // NeurIPS, 2017.',
            'Akiba T. et al. Optuna: A Next-generation Hyperparameter Optimization Framework // KDD, 2019.',
            'Mitchell M. et al. Model Cards for Model Reporting // FAccT, 2019.',
            'Gebru T. et al. Datasheets for Datasets // CACM, 2021.',
            'Sculley D. et al. Hidden Technical Debt in Machine Learning Systems // NeurIPS, 2015.',
            "O'zbekiston Respublikasining «Shaxsga doir ma'lumotlar to'g'risida»gi Qonuni, 2019 (O'RQ-547)."]:
    P(ref).style = doc.styles['List Number']

page_break()
H('Ilova A. Repozitoriy tuzilishi va ishga tushirish')
CODE('''
YKL_V23_TicketRouter/
├── data/            raw/ (csv, provenance.json)  processed/  reference/
├── src/data/        ingest.py  uz_synth.py  validate.py  split.py
├── src/models/      train.py  evaluate.py  fairness.py  explain.py
├── src/api/         main.py
├── src/monitoring/  drift.py
├── tests/           test_data.py  test_model.py  test_api.py  loadtest.py
├── docker/          Dockerfile  docker-compose.yml  prometheus.yml  alert_rules.yml  grafana/
├── docs/            data_card.md  model_card.md  fairness_report.md  risk_register.md
├── models/          model.joblib  threshold.json  group_thresholds.json  model_reweighed.joblib
├── reports/         *.json  figures/*.png  error_analysis_top25.csv
├── .github/workflows/ci.yml   dvc.yaml   Makefile   requirements.txt   README.md   mlflow.db

make all          # data -> train -> evaluate -> fairness -> drift -> test
make serve        # uvicorn src.api.main:app --port 8000
make load         # yuk testi
''')
H('Ilova B. API so\'rov va javob namunasi', 2)
CODE(open('reports/sample_predict_uz.json').read()[:1400])
H('Ilova C. Himoya savollariga qisqa javoblar', 2)
for q, a in [
    ('Nima uchun F1-macro va top-3?', 'Toifalar nomutanosib: accuracy katta navbatlarni afzal ko\'radi, F1-macro har '
     'navbatga teng vazn beradi; top-3 operatorga beriladigan tavsiya sifatini o\'lchaydi.'),
    ('Leakage qanday oldini olingan?', 'Bo\'linish modeldan oldin; TF-IDF faqat train da fit (Pipeline ichida, CV da ham '
     'har foldda); chegara val da tanlanib, test faqat bir marta ishlatildi; dublikatlar bo\'linishdan oldin olib tashlandi '
     '(test_splits_disjoint).'),
    ('Chegarani o\'zgartirsak?', 'Pastroq chegara — ko\'proq avtomatik, lekin noto\'g\'ri yo\'naltirish ko\'payadi '
     '(6-jadval: C_wrong = 2 da chegara pasayadi); 4-rasmdagi egri chiziq kompromissni ko\'rsatadi.'),
    ('Mitigatsiyadan so\'ng aniqlik?', 'Tilga xos chegara F1 ni o\'zgartirmaydi (faqat yo\'naltirishni o\'zgartiradi); '
     'Reweighing F1 ni 0.003 ga pasaytirdi va foyda bermadi — shuning uchun qo\'llanmadi.'),
    ('Model buzilganini qanday bilamiz?', 'Birinchi bo\'lib PSI alerti (kirish taqsimoti), keyin yo\'naltirish ulushi; '
     'teglar kelgach F1 pasayishi alerti.'),
    ('Yangi model qachon almashtiradi?', 'Staging da pytest, F1 ≥ champion − 0.01 va fairness talablari bajarilsa.'),
    ('Qanday hujum mumkin?', 'Kalit so\'z qo\'shib navbatni o\'zgartirish (zarar kichik — faqat navbat), ko\'p so\'rov bilan '
     'modelni nusxalash (rate limit, API-kalit), loglardan PII (maskalash).'),
    ('Mijoz e\'tiroz bildirsa?', 'Har javobda request_id va reasons bor; operator murojaatni qo\'lda boshqa navbatga '
     'o\'tkazadi, bu holat qayta o\'qitish uchun teg sifatida saqlanadi.'),
]:
    P(f'**{q}** {a}')

doc.save('reports/YKL_Variant23_TicketRouter_Hisobot.docx')
print('saqlandi: reports/YKL_Variant23_TicketRouter_Hisobot.docx')
