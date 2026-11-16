"""O'zbek tilidagi sintetik murojaatlar generatori (variant talabi: "+ o'zbek tilidagi murojaatlar").
Shablonlar + tasodifiy to'ldirish; ~15% matnlar kirill yozuvida; mijoz turi (jismoniy/yuridik) aniq belgilanadi.
MUHIM: bu sintetik ma'lumot; real o'zbek murojaatlarida sifat pastroq bo'lishi kutiladi (Data Card ga yozilgan)."""
import random

import pandas as pd

PRODUCTS = ['mobil ilova', 'internet-bank', 'veb-sayt', 'CRM tizimi', 'hisob-kitob moduli', 'router', 'noutbuk',
            'printer', 'bulutli xizmat', 'kassa apparati', 'smartfon', 'onlayn do\'kon']
SUBJ_COMMON = ['Murojaat', 'Yordam kerak', 'Savol', 'Muammo haqida', 'Iltimos, yordam bering']
WHO = {'jismoniy': ['Men', 'Men shaxsan', 'Hurmatli jamoa, men'],
       'yuridik': ['Kompaniyamiz', '"Orzu Savdo" MChJ', 'Tashkilotimiz', 'Bizning korxonamiz', 'Firmamiz']}
T = {
    'Technical Support': [
        '{who} {p} bilan ishlashda xatolik yuz berdi: dastur ishga tushganda "500" kodi chiqmoqda.',
        '{who} {p} yangilangandan keyin tizim qotib qolyapti, qayta yuklash ham yordam bermadi.',
        '{who} {p} sozlamalarini o\'zgartira olmayapmiz, saqlash tugmasi ishlamaydi.'],
    'Product Support': [
        '{who} sotib olgan {p} funksiyalaridan qanday foydalanishni tushuntirib bera olasizmi?',
        '{who} {p} da yangi imkoniyat qayerda joylashganini topa olmadi, qo\'llanma kerak.',
        '{who} {p} mahsulotining kafolat shartlari va imkoniyatlari haqida ma\'lumot so\'raydi.'],
    'Customer Service': [
        '{who} xizmat ko\'rsatish sifatidan norozi, operator javob bermadi.',
        '{who} buyurtma holatini bilmoqchi, hali hech qanday xabar kelmadi.',
        '{who} shaxsiy kabinetdagi ma\'lumotlarni yangilashda yordam so\'raydi.'],
    'IT Support': [
        '{who} korporativ pochtaga kira olmayapti, parolni tiklash kerak.',
        '{who} VPN ulanmayapti, masofadan ishlash to\'xtab qoldi.',
        '{who} yangi xodim uchun hisob qaydnomasi va {p} ga ruxsat ochib berishingizni so\'raydi.'],
    'Billing and Payments': [
        '{who} hisobidan {n} so\'m ikki marta yechib olindi, qaytarib berishingizni so\'raymiz.',
        '{who} oxirgi hisob-fakturada noto\'g\'ri summa ko\'rsatilgan, tekshirib bering.',
        '{who} to\'lov kartasi orqali to\'lov o\'tmayapti, {p} da xatolik chiqmoqda.'],
    'Returns and Exchanges': [
        '{who} {p} ni qaytarib, pulini qaytarib olmoqchi, mahsulot nuqsonli chiqdi.',
        '{who} buyurtma qilingan {p} o\'rniga boshqa model keldi, almashtirish kerak.',
        '{who} qaytarish muddati va tartibi haqida ma\'lumot so\'raydi.'],
    'Service Outages and Maintenance': [
        '{who} hududimizda {n} daqiqadan beri internet yo\'q, xizmat to\'liq to\'xtagan.',
        '{who} {p} ishlamayapti, rejali texnik ishlar qachon tugaydi?',
        '{who} serverlaringiz ertalabdan beri javob bermayapti, uzilish sababi nima?'],
    'Sales and Pre-Sales': [
        '{who} {p} ning korporativ tarifi va narxlari haqida taklif olmoqchi.',
        '{who} {n} ta litsenziya xarid qilish uchun tijorat taklifini yuboring.',
        '{who} {p} ni sinab ko\'rish uchun demo-versiya so\'raydi.'],
    'Human Resources': [
        '{who} bo\'sh ish o\'rinlari va ishga qabul tartibi haqida so\'raydi.',
        '{who} xodimning ta\'til arizasi va ish haqi hisob-kitobi bo\'yicha savoli bor.',
        '{who} amaliyot (stajirovka) dasturiga hujjat topshirmoqchi.'],
    'General Inquiry': [
        '{who} ish vaqtingiz va manzilingiz haqida ma\'lumot so\'raydi.',
        '{who} hamkorlik bo\'yicha kim bilan bog\'lanish mumkinligini bilmoqchi.',
        '{who} sizning xizmatlaringiz haqida umumiy ma\'lumot olmoqchi.'],
}
TAILS = ['Iltimos, tezroq javob bering.', 'Oldindan rahmat.', 'Bu masala biz uchun juda muhim.',
         'Telefon raqamim: +998 90 123 45 67.', 'Javobingizni kutamiz.', '', '']
LAT2CYR = {"o'": 'ў', "g'": 'ғ', 'sh': 'ш', 'ch': 'ч', 'ya': 'я', 'yu': 'ю', 'yo': 'ё', 'a': 'а', 'b': 'б',
           'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г', 'h': 'ҳ', 'i': 'и', 'j': 'ж', 'k': 'к', 'l': 'л', 'm': 'м',
           'n': 'н', 'o': 'о', 'p': 'п', 'q': 'қ', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'x': 'х',
           'y': 'й', 'z': 'з', "'": 'ъ'}


def to_cyr(s):
    out, i, low = [], 0, s.lower()
    while i < len(s):
        for k in sorted(LAT2CYR, key=len, reverse=True):
            if low.startswith(k, i):
                c = LAT2CYR[k]
                out.append(c.upper() if s[i].isupper() else c)
                i += len(k)
                break
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def generate(n=2000, seed=23, label_noise=0.08):
    rnd = random.Random(seed)
    qs = list(T)
    weights = [29, 19, 16, 11, 10, 5, 4, 3, 2, 1]  # asosiy to'plamdagi nisbatlarga yaqin
    rows = []
    for i in range(n):
        q = rnd.choices(qs, weights)[0]
        ct = rnd.choices(['jismoniy', 'yuridik'], [0.6, 0.4])[0]
        body = rnd.choice(T[q]).format(who=rnd.choice(WHO[ct]), p=rnd.choice(PRODUCTS), n=rnd.randint(2, 500))
        if rnd.random() < 0.5:  # boshqa toifadan "shovqin" gap qo'shish (vazifani soddalashtirmaslik uchun)
            q2 = rnd.choice(qs)
            body += ' ' + rnd.choice(T[q2]).format(who='Shuningdek,', p=rnd.choice(PRODUCTS), n=rnd.randint(2, 99))
        body += ' ' + rnd.choice(TAILS)
        subj = rnd.choice(SUBJ_COMMON)
        script = 'cyrl' if rnd.random() < 0.15 else 'latn'
        if script == 'cyrl':
            body, subj = to_cyr(body), to_cyr(subj)
        label = q if rnd.random() > label_noise else rnd.choice(qs)
        rows.append({'subject': subj, 'body': body.strip(), 'queue': label, 'language': 'uz', 'script': script,
                     'customer_type': ct, 'type': rnd.choice(['Incident', 'Request', 'Problem', 'Change']),
                     'priority': rnd.choice(['low', 'medium', 'high']), 'source': 'uz_synthetic'})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(generate(5).to_string())
