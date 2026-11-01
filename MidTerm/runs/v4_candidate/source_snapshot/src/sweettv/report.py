"""Five-page main report plus references, compiled directly from measured artifacts."""
import html
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts'
OUT=ROOT/'output/pdf/nikolai_evdokimov_midterm_report.pdf'
BLUE=colors.HexColor('#153f67')
GRAY=colors.HexColor('#59636d')
PALE=colors.HexColor('#edf3f7')


def load(name):return json.loads((ART/(name+'.json')).read_text())
def esc(s):return html.escape(str(s))
def num(x):return f'{x:,}'.replace(',',' ')


class Report:
    def __init__(self):
        fonts=Path(os.getenv('SWEETTV_FONT_DIR','/System/Library/Fonts/Supplemental'))
        regular=fonts/'Arial.ttf';bold=fonts/'Arial Bold.ttf'
        if not regular.exists():
            fonts=Path('/usr/share/fonts/truetype/dejavu');regular=fonts/'DejaVuSans.ttf';bold=fonts/'DejaVuSans-Bold.ttf'
        pdfmetrics.registerFont(TTFont('Body',str(regular)))
        pdfmetrics.registerFont(TTFont('Body-Bold',str(bold)))
        pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body',boldItalic='Body-Bold')
        OUT.parent.mkdir(parents=True,exist_ok=True)
        self.c=Canvas(str(OUT),pagesize=A4,pageCompression=1)
        self.c.setTitle('SWEET.TV: временная валидация и гибрид рекомендаций')
        self.c.setAuthor('Евдокимов Николай Николаевич')
        self.w,self.h=A4;self.x=38;self.width=self.w-76;self.page=0
        self.style=ParagraphStyle('body',fontName='Body',fontSize=9.3,leading=12,spaceAfter=5,textColor=colors.HexColor('#17232f'))
        self.small=ParagraphStyle('small',parent=self.style,fontSize=8,leading=10)
        self.cell=ParagraphStyle('cell',parent=self.style,fontSize=8,leading=9.5,spaceAfter=0)
        self.tiny=ParagraphStyle('tiny',parent=self.style,fontSize=7.5,leading=8.9,spaceAfter=0)
        self.measurements=[]

    def page_start(self,label):
        if self.page:
            self.finish_page()
            self.c.showPage()
        self.page+=1;self.y=self.h-52
        self.c.setFont('Body',8);self.c.setFillColor(GRAY)
        self.c.drawString(self.x,self.h-26,'LLM4Rec · Индивидуальный экзамен · ФКН · 29.10.2026')
        self.c.drawRightString(self.w-self.x,self.h-26,label)
        self.c.setStrokeColor(BLUE);self.c.setLineWidth(.6);self.c.line(self.x,self.h-33,self.w-self.x,self.h-33)

    def finish_page(self):
        assert self.y>=38,(self.page,self.y,'page overflow')
        self.measurements.append({'page':self.page,'last_content_y':self.y,'no_vertical_overflow':True})
        self.c.setFont('Body',8);self.c.setFillColor(GRAY)
        self.c.drawString(self.x,22,'Евдокимов Николай Николаевич · nickevdokimov')
        self.c.drawRightString(self.w-self.x,22,str(self.page))

    def p(self,text,small=False,gap=5):
        p=Paragraph(text,self.small if small else self.style);_,h=p.wrap(self.width,1000)
        self.y-=h;p.drawOn(self.c,self.x,self.y);self.y-=gap

    def heading(self,text,size=13):
        self.y-=size+3;self.c.setFont('Body-Bold',size);self.c.setFillColor(BLUE)
        self.c.drawString(self.x,self.y,text);self.y-=9

    def table(self,rows,widths,style=None,span_headers=False):
        cs=style or self.cell
        data=[[v if isinstance(v,Paragraph) else Paragraph(str(v),cs) for v in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),PALE),
          ('LINEBELOW',(0,0),(-1,0),.5,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe2e8')),
          ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
          ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        if span_headers:
            t.setStyle(TableStyle([('SPAN',(0,0),(2,0)),('SPAN',(3,0),(5,0))]))
        _,height=t.wrap(self.width,1000);self.y-=height;t.drawOn(self.c,self.x,self.y);self.y-=8

    def image(self,path):
        from reportlab.lib.utils import ImageReader
        im=ImageReader(str(path));w,h=im.getSize();height=self.width*h/w
        self.y-=height;self.c.drawImage(im,self.x,self.y,width=self.width,height=height);self.y-=6

    def done(self):
        self.finish_page();self.c.save()
        (ART/'report_layout_checks.json').write_text(json.dumps({'pages':self.page,'main_text_pages':5,'reference_pages':self.page-5,'page_checks':self.measurements},indent=2)+'\n')


def build():
    a=load('data_audit');v=load('validation_protocol');results=load('experiments');s=load('selection');checks=load('submission_checks');env=load('environment')
    ex=load('user_examples');best=s['selected'];r={x['method']:x for x in results}
    kaggle=load('kaggle_result') if (ART/'kaggle_result.json').exists() else None
    personal=json.loads((ROOT/'report/personal_verification.json').read_text()) if (ROOT/'report/personal_verification.json').exists() else None
    d=Report();d.page_start('Задача и данные')
    d.heading('SWEET.TV: прогноз любимых телепрограмм',17)
    d.p('Студент: <b>Евдокимов Николай Николаевич</b> · Individual<br/>Email: nevdokimov@hse.ru · Kaggle: nickevdokimov · Midterm Quiz / ФКН',small=True)
    d.p('<b>Аннотация.</b> Требуется предсказать пять будущих предпочитаемых программ по неполным логам телеканалов. '
        'История соединена с расписанием по каналу и пересечению интервалов; на одном временном hold-out сравнены популярность, CBF, Association Rules и частотные стратегии. '
        f'Выбранный гибрид получил MAP@5={r[best]["map5"]:.5f} против {r["popularity"]["map5"]:.5f} у популярности'+(f', Kaggle private score={kaggle["private_score"]:.5f}.' if kaggle else '; score Kaggle пока не получен.')+
        ' Повторные предпочтения дают сильный ориентир, а небольшой выигрыш гибрида над личной частотой требует осторожной интерпретации.')
    d.heading('1. Постановка и источники требований')
    d.p('Вход: история недель 11–30 и расписание до недели 42; выход: пять упорядоченных ID для каждого из 1260 пользователей. '
        'Экзамен [1, 2] требует два допустимых метода сверх popularity и общий протокол. Официальная страница [3] сообщает: цель составлена из пяти наиболее частых программ с долей просмотра ≥80%; public/private разделены по пользователям 50/50. '
        'Место замороженного leaderboard не используется для оценки. Дедлайн: 01.11.2026, 19:00 МСК.')
    d.heading('2. Фактические данные и сопоставление')
    table=[['<b>CSV</b>','<b>Строк</b>','<b>Уникальные ID / пользователи</b>']]
    table.append(['dataset11-30',num(a['logs']['rows']),f'{num(a["logs"]["users"])} пользователей; 202 канала'])
    for entry,label in zip(a['schedule'],['Расписание 11–20','Расписание 21–30','Расписание 31–42']):
        table.append([label,num(entry['rows']),f'{num(entry["programs"])} положительных ID; {entry["channels"]} каналов'])
    table.append(['submission',num(a['template']['rows']),'user_id,tv_show_id; 5 ID через пробел'])
    d.table(table,[166,105,d.width-271])
    d.p('Исходные CSV имеют запятую как разделитель и UTF-8. user_id читается строкой: многие ID превышают знаковый int64 и точность float. '
        'Нулевые tv_show_id исключены как неразмеченный эфир: в первом расписании 115613/1074235 = 10,76% строк имеют положительный ID. '
        'В дополнительных материалах число «уникальных программ» на единицу больше фактического числа положительных ID: там включён 0.')
    d.p('Фактические логи: 09.03.2020 00:00:02 – 26.07.2020 23:59:58. Формат логов YYYY-MM-DD, расписания DD.MM.YYYY. '
        'Часовые пояса в файлах отсутствуют; принято одинаковое локальное wall-clock время без сдвига, абсолютная зона не подтверждена. '
        'Удалено 17 дублей логов и один эфир нулевой длительности. Пустых ключей/времён в логах нет; 117 значений duraton расходятся с временами, поэтому используется stop−start. Медиана просмотра 396 s, максимум 342145 s.')
    d.p('Равенство vsetv_id и channel_id явно подтверждено [3] и проверено по множествам: 5708392/5821970 логов имеют известный канал; '
        'отсутствуют каналы 1174, 1510, 1509, 1508, 149 (113578 логов). Расписания немного выходят за номинальные недели: разделение сделано по timestamp. '
        'Пропуски жанров и персоналий допустимы: отсутствующие признаки не заменяются выдуманными значениями.',small=True)

    d.page_start('Метод и валидация');d.heading('3. Взаимодействия и отсутствие утечек')
    d.p('Для просмотра [s,e) и эфира [a,b) на одном канале пересечение равно max(0,min(e,b)−max(s,a)). '
        'Day-bucketing ограничивает соединение, затем применяется точная проверка интервалов. Один лог может пересечь несколько эфиров; '
        'фрагменты каждого user–airing объединяются в непересекающиеся интервалы. Просмотр засчитывается при union_seconds / duration ≥0,8. '
        'Каждый повторный эфир даёт отдельное событие, затем события одного user–ID суммируются.')
    d.p(f'Покрытие: с любым эфиром пересеклись {num(a["coverage_any_airtime"]["matched_logs"])} логов, с положительным ID — {num(a["labelled_overlap"]["matched_logs"])} '
        f'({a["labelled_overlap"]["matched_logs"]/a["channel_mapping"]["logs"]:.1%}); получено {num(a["air_views"]["user_air_pairs"])} пар user–airing, '
        f'{num(a["air_views"]["fraction_80_pairs"])} покрытий ≥80% до граничных фильтров. Без такой истории остались 73 из 4914 пользователей; в финальном шаблоне — 4 из 1260. '
        'У 590192 логов несколько размеченных эфиров. Найдено 13 строк расписания с перекрытием предыдущего эфира канала: сохранены заданные эфиры, возможна неоднозначность.')
    d.p('Абсолютный порог 300 s допускает 2078912 user–airing пар, включая малую долю длинной программы. Порог 80% выбран по официальному правилу, '
        'а 300 s проверен только как альтернатива для обучения при неизменной цели. Отсутствующие или невалидные времена не восстанавливаются. '
        'ID=0 используется лишь в аудите покрытия, не в моделях.')
    d.heading('4. Единый временной протокол и MAP@5')
    d.p(f'Обучение [09.03,18.05), hold-out [18.05,27.07), 2020 год. Эфиры, пересекающие границу, не создают метки; исключено {v["boundary_crossing_user_air_pairs"]} таких user–airing пар. '
        f'Обучающих событий {num(v["train_air_views"])}, user–ID пар {num(v["train_user_item_pairs"])}. Модели обучаются на всей доступной прошлой истории 4914 пользователей; оценка — на фиксированном списке шаблона. '
        'Цель: до пяти уникальных ID по частоте квалифицированных просмотров после границы, равенства разрешаются меньшим ID.')
    d.p(f'Общие кандидаты: {v["candidate_count"]} положительных ID эфиров, пересекающих hold-out; покрытие релевантных ID 100%. '
        f'Из 1260 пользователей у {v["eligible_users"]} непустая цель, у {v["users_with_5_ground_truth"]} — пять ID; 22 пустые цели исключены из основной MAP и отдельно получают 0. '
        f'Без обучающей истории — 18 пользователей, из них 14 с целью. Повторные ID разрешены: {v["ground_truth_seen_in_train_fraction"]:.1%} целей уже встречались в обучении. '
        'Рекомендации уникальны внутри списка. Метаданные, словарь и IDF — строго до границы; будущее расписание даёт только ID/канал/время, не жанры или названия для scoring.')
    d.p('<b>AP@5(u)</b> = [Σ по r=1…5 Precision@r · 1{ID на ранге r релевантен и ещё не засчитан}] / min(5, |Rᵤ|). '
        '<b>MAP@5</b> — среднее AP по пользователям выбранной группы. Порядок предсказаний влияет на precision; порядок целевого набора не взвешивается. '
        'Формула соответствует слайду 4 [1], примеры слайда 5 проверены кодом (11/12 и 8/15). Kaggle не публикует полный код цели; слияние фрагментов и tie-break локально зафиксированы, их тождество скрытой разметке не заявляется.')
    d.heading('5. Реализованные модели')
    d.p('<b>Popularity:</b> число уникальных обучающих зрителей ID. <b>Repeat:</b> число квалифицированных эфиров пользователя; вариант 28 использует вес каждого события 2^(−age_days/28). '
        '<b>CBF:</b> DictVectorizer + TF-IDF [4], L2-нормировка item-векторов и суммы векторов просмотренного. Score — cosine. Признаки: категория/название (вес 1), жанры/режиссёр (2), актёры (1,5), год (0,5); '
        'пропуски игнорируются, значения поля выбираются модой прошлого расписания. ID не является признаком.')
    d.p('<b>Association Rules:</b> корзина — уникальная история пользователя. Правило i→j имеет вес n(i,j)/(n(i)+λ); минимум 3 общих зрителя, максимум 100 следствий, i→i удалено. '
        'Score — сумма правил с весом √частоты antecedent; λ=5 или 20. Разреженные матрицы SciPy [5]. Для всех методов положительные score ранжируются первыми, затем общий fallback Popularity; ties — popularity и ID. '
        'У пользователя без истории все методы возвращают один и тот же Top-5 popularity. Модели могут переоценивать популярные передачи и не различают домочадцев.')

    d.page_start('Эксперименты');d.heading('6. Сравнение и выбор стратегии')
    d.p(f'Среда: Python {env["python"]}, macOS 26.5.1 / arm64, 11 логических CPU; DuckDB {env["libraries"]["duckdb"]}, NumPy {env["libraries"]["numpy"]}, '
        f'SciPy {env["libraries"]["scipy"]}, scikit-learn {env["libraries"]["scikit-learn"]}. DuckDB: 4 потока, memory_limit=3 GB. '
        f'Подготовка: {a["preparation_seconds"]:.2f} s, peak RSS {a["peak_rss_bytes"]/1024**3:.2f} GiB. Полный цикл моделей: {checks["total_experiment_seconds"]:.2f} s; '
        f'финальное обучение и ранжирование: {checks["fit_and_predict_seconds"]:.2f} s. Seed 42 для bootstrap; сами модели детерминированы. Время ниже включает построение score и Top-5, но не I/O подготовки.')
    labels={'popularity':'Popularity','repeat':'Repeat','repeat28':'Repeat-28','content':'CBF','content28':'CBF-28','association5':'AR λ=5','association20':'AR λ=20','association20_28':'AR λ=20, 28d',
       'hybrid80_20':'H 80/20/0, 28d','hybrid90_10':'H 90/10/0, 28d','hybrid80_10_10':'H 80/10/10, 28d','hybrid60_20_20':'H 60/20/20, 28d',
       'hybrid80_10_10_static_repeat':'H 80/10/10, Repeat без decay','hybrid60_20_20_static_repeat':'H 60/20/20, Repeat без decay','repeat28_absolute300':'Repeat-28, порог 300 s (диагн.)'}
    table=[['<b>Метод / конфигурация</b>','<b>MAP@5</b>','<b>Все 1260</b>','<b>|R|=5</b>','<b>Время, s</b>']]
    for x in results:
        name=esc(labels[x['method']]);name='<b>'+name+'</b>' if x['method']==best else name
        table.append([name,f'{x["map5"]:.5f}',f'{x["map5_all_users_empty_zero"]:.5f}',f'{x["map5_gt5"]:.5f}',f'{x["fit_and_predict_seconds"]:.3f}'])
    d.table(table,[d.width-244,62,62,60,60])
    d.p('H: веса Repeat / AR-20-28 / CBF-28. Перед сложением каждый score делится на максимум этого пользователя среди кандидатов (нулевой максимум → нулевой вектор). '
        'Рейтинги не являются вероятностями. Первоначально проверены 12 конфигураций; после ухудшения Repeat-28 добавлены два гибрида с отключением decay только у Repeat. '
        'Порог 300 s — отдельная диагностика, не кандидат выбора. Полная сетка и отрицательные результаты сохранены в artifacts/experiments.json.')
    d.p(f'<b>Выбрано:</b> H 60/20/20 с Repeat без decay, MAP={r[best]["map5"]:.5f}, прирост к popularity {s["delta_vs_popularity"]:.5f}. '
        f'Парный percentile bootstrap (2000 выборок пользователей): [{s["paired_bootstrap_95_percentile_interval"][0]:.5f}; {s["paired_bootstrap_95_percentile_interval"][1]:.5f}]. '
        f'К Repeat прирост {s["delta_vs_repeat"]:.5f}, интервал [{s["paired_interval_vs_repeat"][0]:.5f}; {s["paired_interval_vs_repeat"][1]:.5f}], включая 0. '
        'Интервалы описательные: та же выборка использована для настройки, поправки на выбор модели нет. Надёжное превосходство над Repeat не установлено.')
    d.heading('7. Проверки и реальный исправленный дефект')
    d.p('7 unit tests проверяют AP, пустые цели, дубли предсказаний, fallback, касание интервалов, вложенные/одинаковые интервалы, направленность AR и нулевую диагональ. '
        'Независимая Python-проверка 200 user–airing пар совпала с SQL. Тест загрузчика исключил будущие метаданные. Два обучения с нуля дали одинаковые SHA-256 всех предсказаний и submission.')
    d.p('Аудит обнаружил долю 2,0 у user 1388513685702887520, эфир 22.05.2020 07:45–07:50. Два лога обрезались до одинаковых границ; '
        'оконные функции с ORDER BY s,e могли по-разному упорядочить равенства и создать два острова. Добавлен общий tie-break log_id; доля стала 1,0, '
        'добавлены regression test и assert fraction≤1 для всей базы. Это проверка агента, а не заявление о личном выполнении студентом.')
    if personal and personal.get('discussion'):
        d.heading('Личная интерпретация студента',11)
        d.p('«'+esc(personal['discussion'])+'».',small=True)
        d.p('Уточнение реализации: отдельного веса новизны нет; предложение можно задать снижением доли AR/CBF для однородной истории. Это следующий эксперимент, он не проводился.',small=True)

    d.page_start('Два пользователя');d.heading('8. Истории и одинаковые примеры для всех методов')
    d.p('Выбор детерминирован: самая насыщенная обучающая история среди оцениваемых пользователей и самая короткая непустая история (ties по user_id). '
        'Примеры не выбраны по удачным рекомендациям. Названия/категории ниже из расписания до конца hold-out нужны только для объяснения, не для обучения.',small=True)
    heavy,light=ex
    d.p(f'<b>A: {heavy["user_id"]}</b> — {heavy["history_air_views"]} эфира, {heavy["history_unique_items"]} ID. '
        'Главные ID: 12001682 «Губка Боб» (401), 12002856 «Элвиннн» (68), 2400480 «ТСН» (62). '
        'Цель: 12001682 (198), 12002856 (55), 12001697 (43), 2400480 (36), 12002355 (36).<br/>'
        f'<b>B: {light["user_id"]}</b> — 1 эфир, ID 200337 «Утро с Интером». Цель: 240081 (9), 2400480 (3), 5200303 (3), 20088 (2), 200337 (1). '
        'Числа в скобках — частоты квалифицированных эфиров в соответствующем периоде.',small=True)
    methods=[('popularity','Popularity'),('content28','CBF-28'),('association20_28','AR-20-28'),('repeat','Repeat'),(best,'Итоговый H 60/20/20')]
    for method,title in methods:
        table=[[f'<b>A · {title} · AP={heavy["ap5"][method]:.3f}</b>','','',f'<b>B · AP={light["ap5"][method]:.3f}</b>','','']]
        for rank,(ha,la) in enumerate(zip(heavy['recommendations'][method],light['recommendations'][method]),1):
            row=[]
            for rec in [ha,la]:
                marker='+' if rec['hit'] else ''
                row.extend([f'{rank}. {rec["id"]}{marker}',esc(rec['title']),esc(rec['category'])])
            table.append(row)
        d.table(table,[58,147,54,58,147,d.width-464],d.tiny,span_headers=True)
    d.p('«+» — ID входит в Top-5 ground truth; отсутствие знака не доказывает отсутствие просмотра или интереса. '
        'Полные истории Top-10 и списки всех конфигураций сохранены в artifacts/user_examples.json.',small=True)

    d.page_start('Интерпретация и посылка');d.heading('9. Стратегия платформы и ограничения')
    d.p('Для этого офлайн-прогноза выбрана смесь с опорой на личную частоту. У A CBF сохраняет детский контент, AR тяготеет к совместно популярным новостям; '
        'гибрид вводит «Факты» и «Викна», не попавшие в цель, и получает AP=0,55. У B единственное утреннее шоу мало определяет будущий интерес; AR добавляет «Завтрак с 1+1», '
        'а CBF переносит сходство на другие утренние передачи. Гибрид имеет AP=0,453. В обоих примерах Repeat лучше выбранного гибрида: средний выигрыш не гарантирует улучшения каждому зрителю.')
    d.p('Механизм AR основан на совместных зрителях, CBF — на общих атрибутах; первые усиливают массовые новости, вторые могут путать похожий контент с релевантностью. '
        'Вычислительные затраты гибрида малы на этой выборке, но преимущества над Repeat неустойчивы. Для реального сервиса полезно сначала проверить простой Repeat и ограничение доли новостей на новом временном срезе. '
        'MF и PPR рассматривались как альтернативы, но не реализованы и эмпирически не отвергаются: выбрана интерпретируемая пара без дополнительных латентных/итерационных настроек.')
    d.p('Неразмеченный эфир, 22 пустые цели, неполные атрибуты и 2391 кандидатов hold-out без прошлых метаданных ограничивают переносимость. '
        'Один hold-out 10 недель отличается от будущих 12; оценка на пользователях финального шаблона отражает выбранную аудиторию. '
        'Незнакомая программа не равна полезному открытию; отсутствие в логах не означает «никогда не смотрел». A/B-тесты и выручка не измерялись.',small=True)
    d.heading('10. Итоговая посылка и проверка идентичности')
    if kaggle:
        d.p(f'<b>Private {kaggle["private_score"]:.5f}; public {kaggle["public_score"]:.5f}; Complete.</b> Отправлено пользователем {kaggle["submitted_at_msk"]} МСК. '
            'Модель выбрана до score; после отправки не менялась. Скачанный из карточки Kaggle CSV побитово совпал с локальным файлом. '
            'Использована одна попытка; форма показывала 100 до / 99 после. Выбор для frozen leaderboard недоступен и не нужен для экзамена.',small=True)
        d.image(ART/'kaggle_private_score_detail.jpg')
    else:
        d.p('<b>Черновик: Kaggle score и обязательный скриншот ещё не зафиксированы. Этот PDF не готов к сдаче.</b>')
    d.p('CSV: artifacts/final_submission.csv · 1260 строк + заголовок, 2 столбца, 5 уникальных положительных ID; все ID есть и в исходном тестовом расписании. '
        'SHA-256: '+checks['sha256'],small=True)

    d.heading('11. AI usage disclosure и личная проверка',11)
    d.p('Codex Desktop, GPT-6.1 Sol (gpt-6.1-sol), runtime 0.163.0-alpha.10.2: поиск, Python/SQL, отладка, запуски, проверки и текст/PDF; версия подтверждена нативным журналом. '
        'Внешние данные, модели и чужой рекомендательный код не использовались. Проверки агента перечислены в §7 и не приписываются студенту.',small=True)
    if personal:
        d.p('<b>Личная проверка студента:</b> R={10,20,30}, список [10,99,20,88,30]; precision 1, 2/3, 3/5; AP≈0,7556=34/45. '
            'Его обоснование: один ID может объединять разные выпуски, поэтому исключение после первого просмотра скроет новые выпуски. '
            'Его личная интерпретация приведена в §7. Личный перезапуск pipeline не заявляется; остальной текст подготовлен AI и требует окончательного прочтения студентом.',small=True)
    d.p('<b>Репозиторий:</b> <link href="https://github.com/dryjins/RecSys-LLMs">github.com/dryjins/RecSys-LLMs</link>, MidTerm; по просьбе студента публикация отложена. '
        '<b>AI-export:</b> оригинальный tool-native журнал после завершения работы; JSONL не преобразуется в JSON. Если канал ограничивает расширение, нужно уточнение преподавателя. '
        'До публикации, личной правки и окончательного экспорта комплект не готов к сдаче.',small=True)
    d.page_start('Источники');d.heading('Источники')
    refs=[
      '[1] S. Jin. LLM4Rec. Midterm Quiz. Предоставленный PPTX, слайды 2–9, включая изображения с формулой и примерами MAP. Прочитан 29.10.2026.',
      '[2] Final Exam Quiz (FKN): TV Program Recommendation on sweet.tv. Предоставленный exam_quiz_FKN.md, §§2–10. Прочитан 29.10.2026.',
      '[3] ogurbych. sweet.tv — TV program recommender. Kaggle, 2020. Data, Evaluation, Rules и Submission Details проверены в авторизованном браузере 29.10.2026. <link href="https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data">kaggle.com/competitions/sweettv-tv-program-recommender/data</link>.',
      '[4] scikit-learn developers. Feature extraction; DictVectorizer, TF-IDF. Официальная документация. Прочитана 29.10.2026. <link href="https://scikit-learn.org/stable/modules/feature_extraction.html">scikit-learn.org/stable/modules/feature_extraction.html</link>.',
      '[5] SciPy developers. Sparse arrays (scipy.sparse). Официальная документация. Прочитана 29.10.2026. <link href="https://docs.scipy.org/doc/scipy/reference/sparse.html">docs.scipy.org/doc/scipy/reference/sparse.html</link>.',
      '[6] DuckDB Foundation. CSV Import, DuckDB 2.0. Проверены all_varchar, strict_mode и параметры CSV. Прочитано 29.10.2026. <link href="https://duckdb.org/docs/2.0/data/csv/overview">duckdb.org/docs/2.0/data/csv/overview</link>.',
      '[7] S. Jin. LLM4Rec — Homework Guidelines. Локальный ../homework_guidelines.pdf, §§2, 4–5, 7–10, FAQ. Проверены требования Gate 0, личной проверки и неизменённого журнала. Прочитан 29.10.2026.'
    ]
    for ref in refs:d.p(ref,small=True)
    d.done();print(OUT)


if __name__=='__main__':build()
