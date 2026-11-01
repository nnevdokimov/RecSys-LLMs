"""Five main pages + sources, using actual frozen v3 artifacts; pending score is explicit."""
import html
import json
from pathlib import Path

from sweettv import report as base

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/v3_candidate'
def read(path):return json.loads((ROOT/path).read_text())
def e(x):return html.escape(str(x))

def build():
    selection=read('runs/v3_candidate/selection.json')
    checks=read('runs/v3_candidate/submission_checks.json');ex=read('runs/v3_candidate/user_examples.json')
    audit=read('artifacts/data_audit.json');personal=read('report/personal_verification.json')
    outcome=read('runs/v3_candidate/kaggle_result.json') if (RUN/'kaggle_result.json').exists() else None
    if outcome:assert outcome['sha256']==checks[0]['sha256']
    base.OUT=ROOT/'output/pdf'/('nikolai_evdokimov_midterm_report_v3.pdf' if outcome else 'nikolai_evdokimov_midterm_report_v3_DRAFT.pdf')
    base.ART=RUN
    d=base.Report()
    d.page_start('Задача, данные и протокол')
    d.heading('SWEET.TV: время, предпочтения и Top-5',16)
    d.p('Евдокимов Николай Николаевич · nevdokimov@hse.ru · nickevdokimov<br/>Индивидуальный экзамен LLM4Rec / ФКН · 01.11.2026, 19:00 МСК',small=True)
    d.p(f'<b>Результат.</b> Реализованы popularity, Content-Based Filtering (CBF), Association Rules (AR), implicit ALS и гибриды. '
        f'Выбран вложенно обученный ранжировщик: средний MAP@5 по трём временным срезам {selection["mean_map5"]:.5f}; '
        'у ранее отправленной v2 — 0,40622. '+(f'Именно CSV этой версии получил Kaggle private {outcome["private_score"]:.5f}.' if outcome else '<b>Черновик: кандидат v3 ещё не имеет подтверждённого Kaggle score.</b>'))
    d.heading('1. Данные и интервальные взаимодействия')
    d.table([['Файл','Фактические строки','Назначение'],['dataset11-30', '5 821 987','4914 пользователя, 202 канала'],
             ['export 11–20','1 074 235','5847 положительных ID'],['export 21–30','1 074 394','6363 положительных ID'],
             ['export 31–42','1 295 378','6636 положительных ID'],['submission','1260','user_id,tv_show_id; 5 ID через пробел']], [150,105,d.width-255])
    d.p('Логи: 09.03–26.07.2020; даты логов YYYY-MM-DD, расписаний DD.MM.YYYY. Разделитель CSV — запятая, UTF-8; user_id хранится строкой из-за размера чисел. '
        'Часовой пояс не указан: сохранена общая наивная шкала без сдвига. Удалены 17 дублей логов и один нулевой эфир. 117 duraton расходятся со stop−start; используется интервал timestamp. Медиана 396 s, максимум 342145 s.')
    d.p('vsetv_id=channel_id подтверждено Data [3] и проверено по множествам: 5708392/5821970 логов на известном канале. '
        'Соединение: одинаковый канал и непустое пересечение [s,e)∩[a,b); секунды=max(0,min(e,b)−max(s,a)). '
        'Фрагменты user–airing объединяются в union; ≥80% duration означает квалифицированный просмотр. Повторные эфиры — отдельные события; затем суммируются по user–ID. Один лог может покрыть несколько эфиров.')
    d.p('С положительным ID пересеклись 2964335 логов (50,9%); сформированы 3014129 user–airing пар, 611760 покрытий ≥80% до граничных фильтров. '
        'У 73 из 4914 нет такой истории; в финальном шаблоне — 4. Только 10,76% строк первого расписания размечены. ID=0 — не объект; его эфир используется лишь как прошлый контекст канала/времени. '
        'Порог 300 s проверен как диагностическая альтернатива: он допускает 2078912 пар, но не соответствует официальному правилу 80%.')
    d.heading('2. Временная оценка и MAP@5')
    d.p('Обучение всегда начинается 09.03.2020. Границы / конец прогноза: 20.04 / 29.06; 18.05 / 27.07; 15.06 / 27.07. '
        'Те же 1260 пользователей шаблона; оцениваются 1223/1238/1233 с непустой целью. Цель — пять ID по частоте будущих квалифицированных событий, ties по ID. '
        'Кандидаты — положительные ID из расписания соответствующего будущего окна; повторные ID разрешены. Общий legal pool, fallback popularity и tie-break одинаковы. Граничные эфиры не входят в labels.')
    d.p('AP@5(u)=Σᵢ₌₁⁵ Precision@i·1[IDᵢ∈Rᵤ] / min(5,|Rᵤ|); MAP — среднее AP. Дубликаты предсказаний не дают повторных попаданий. '
        'Основная оценка исключает пустой R; дополнительная считает их AP=0. На основном срезе 22 пустые цели, 1150 целей длины 5; все релевантные ID входят в legal pool. '
        'Код цели Kaggle не опубликован: точное совпадение правил фрагментов/ties не заявляется.',small=True)

    d.page_start('Модели и сравнительные результаты');d.heading('3. Базовые и дополнительные методы')
    d.p('<b>Popularity:</b> число уникальных прошлых зрителей программы. <b>Repeat:</b> персональная частота. <b>CBF:</b> DictVectorizer + TF-IDF по прошлым title/category/genre/director/actors/year, '
        'L2 item/user cosine; веса title/category 1, genre/director 2, actors 1,5, year 0,5; профиль с 28d decay. <b>AR:</b> confidence n(i,j)/(n(i)+20), support≥3, top-100 соседей, sqrt(decayed count), без диагонали. '
        '<b>ALS [4]:</b> 48 факторов, confidence=1+20·log(1+n), λ=5, 12 точных alternating solves, seed 42; наблюдения implicit, включая confidence=1 у отсутствующих взаимодействий.')
    d.p('<b>v2:</b> личный channel×тип дня×4h профиль из прошлых сырых логов, 21 день сглаживания. Будущие эфиры интегрируются по часовым пересечениям; '
        'program lift=(n+1)/(expected_past+1)·expected_future. Score нормируется максимумом пользователя и смешивается с item cosine (binary co-view cosine, support/(support+10), top-80). '
        'Вес cosine 0,05 при доле прошлой доминирующей категории ≥0,8 и ≥50 событиях, иначе 0,2. В качестве самостоятельных компонентов они также оценены.')
    d.p('<b>Тренд и мягкое время:</b> Gaussian sigma=3h на циклических часах; дни недели проверены отдельно. 28d half-life применяется согласованно к прошлым просмотрам, часовой экспозиции и эфирам. '
        'Это недавняя история и недельные привычки; годовая сезонность по марту–июлю не оценивается. Само сглаживание почти не помогло; чистый недавний профиль дал 0,40965 в среднем.')
    d.p('<b>Итоговый HGB:</b> кандидатный гибрид + pointwise HistGradientBoostingClassifier [5]. 39 признаков: оценки и ранги v2/Repeat/CBF/AR/cosine/времени, recent counts, давность, '
        'объём истории и эфиры. Сырые user/item ID и будущие содержательные metadata отсутствуют. '+
        f'Генератор: {e(str(selection["parameters"]["candidate_top_k"]))}. Вес HGB={selection["classifier_score_weight"]:g}. '+
        '100 итераций, learning rate 0,07, 15 листьев, min leaf 100, L2=10, seed 42, early stopping отключён.')
    d.p('На каждом внешнем срезе HGB fit использует только внутренние окна: минимум 14 дней прошлой истории, шаг 14 дней, горизонт до 56 дней, конец labels≤внешняя обучающая граница. '
        'Внешние labels читаются после fit. Неотобранным кандидатам score=0; пустая история получает общий popularity fallback. Все методы возвращают ровно 5 уникальных положительных ID, ties popularity→ID.')
    measured=read('runs/v2_candidate/measured_methods.json')
    rows=[['Метод','MAP@5 на 18.05–27.07','Время, s']]
    # Existing measured file is a list of complete actual runs.
    for r in measured:
        rows.append([e(r['method']),f'{r["map5"]:.5f}',f'{r["fit_and_rank_seconds"]:.3f}'])
    rows.append(['Итоговый HGB',f'{selection["split_scores"]["original_10w"]:.5f}','см. §5'])
    d.table(rows,[220,160,d.width-380],d.tiny)

    d.page_start('Поиск, проверка и ограничения');d.heading('4. Выбор по исторической валидации')
    rows=[['Стратегия','20.04','18.05','15.06','Среднее']]
    for label,path,name in [('v1','runs/v2_experiments/summary.json','v1_hybrid'),('v2','runs/v4_refine/summary.json','context21_mean_lift1__adaptive_cos'),
                            ('Недавняя история','runs/v7_soft_trend/summary.json','trend28'),('Итоговый HGB',selection['folder']+'/summary.json',selection['method'])]:
        r=next(x for x in read(path) if x['method']==name)
        rows.append([label]+[f'{r["split_scores"][s]:.5f}' for s in ['early_10w','original_10w','late_6w']]+[f'{r["mean_map5"]:.5f}'])
    d.table(rows,[155,88,88,88,d.width-419])
    d.p('Варианты сохранены полностью: supply/recency, channel priors, ALS, cosine/KNN, clock resolution, soft time, trend, narrow/wide HGB. '
        'В первой расширенной сетке 115 именованных конфигураций, часть — эквивалентные references; затем добавлены отдельные v5–v9. '
        'Неудачные результаты не удалены. Срезы перекрываются и многократно использованы для выбора, поэтому прирост — исторический, без независимой проверки значимости. '
        'Ранее рассчитанные bootstrap-интервалы описательные, без поправки на широкий поиск.')
    d.p('Чистое сокращение истории до 56 дней ухудшило среднее до 0,40204; смесь полной/56d — 0,40760. 7 дней недели с 4h интервалом — 0,40704, '
        'одночасовые weekday/weekend — 0,40636; мягкое время+28d — 0,40582. Больше компонентов не гарантирует лучший результат. '
        'Narrow HGB дал 0,41511, смесь с v2 — 0,41388. Расширение кандидатов проверено из-за неполного coverage, а не по новому private score.')
    d.heading('5. Проверки, среда и исправления')
    d.p('Python 3.12.14, macOS 26.5.1 arm64, 11 logical CPU; DuckDB 2.0.0 (4 потока), NumPy 2.5.4, SciPy 1.18.1, sklearn 1.9.1. '
        f'Подготовка {audit["preparation_seconds"]:.2f} s, peak RSS {audit["peak_rss_bytes"]/1024**3:.2f} GiB. '
        f'Полный fit+predict финального reranker: {checks[0]["fit_predict_seconds"]:.2f} s; повтор с cache: {checks[1]["fit_predict_seconds"]:.2f} s. '
        'Кэш только ускоряет воспроизведение; его ключ включает код и параметры. Время classifier.fit отдельно записано в fit_audit.json.')
    d.p('12 unit tests: AP, fallback, интервалы и union, направленность AR, ALS objective/determinism/cold start, будущие логи, midnight wrap и вложенные границы. '
        'Независимый Python union для 200 пар совпал с SQL. Финальный hold-out переобучен: все 1260 списков идентичны. Два полных fit дали одинаковый CSV SHA-256; предыдущие CSV сохранены.')
    d.p('Исправлены реальные дефекты: равные clipped intervals давали fraction=2,0 из-за разных SQL tie-break (добавлен log_id, assert fraction≤1); '
        'исследовательский KNN использовал число всех roster rows для IDF (заменено числом активных прошлых пользователей; старые KNN числа superseded). '
        'v2 не использует этот KNN; после ограничения roster прошлым все её списки на трёх срезах совпали. Integer popularity нельзя было делить с integer output: нормирование переведено в float, добавлен тест.')
    d.heading('6. Честность и личная проверка')
    d.p('Codex Desktop / gpt-6.1-sol, runtime 0.163.0-alpha.10.2: код, отладка, вычисления и PDF подготовлены с AI. Нативная сессия сохраняется неизменённой; '
        'AI-проверки не приписываются студенту. Студент лично вычислил AP=(1+2/3+3/5)/3=34/45≈0,7556 и объяснил повторы выпусков. '
        'Он предложил проверить мягкое «утро/вечер» и тренд последних недель; оба проверены, отрицательные результаты сохранены. Личный запуск всего pipeline студентом не заявляется.',small=True)

    d.page_start('Одни и те же два пользователя');d.heading('7. Истории и рекомендации')
    d.p('A=10500740270951698446: максимальная история — 904 эфира, 97 ID; Губка Боб (401), Элвинн (68), ТСН (62). '
        'B=12453026253397466416: минимальная непустая история — 1 эфир «Утро с Интером». Ties выбора по user_id, не по качеству рекомендаций. '
        'Цели A: 12001682,12002856,12001697,2400480,12002355; B: 240081,2400480,5200303,20088,200337.',small=True)
    methods=[('popularity','Popularity'),('repeat','Repeat'),('cbf28','CBF-28'),('ar20_28','AR-20-28'),('als48','ALS'),('item_cosine','Item cosine'),('time_context','Время'),('program_lift','Program lift'),('advanced_hybrid','v2'),('recent_trend','Недавний тренд'),('nested_final','Итоговый HGB')]
    rows=[['Метод','A: ранги 1→5','AP','B: ранги 1→5','AP']];catalog={}
    for name,title in methods:
        ids=[]
        for user in ex:
            recs=user['recommendations'][name]
            ids.append(' '.join(str(r['id'])+('+' if r['hit'] else '') for r in recs))
            for r in recs:catalog[r['id']]=r
        rows.append([title,ids[0],f'{ex[0]["ap5"][name]:.3f}',ids[1],f'{ex[1]["ap5"][name]:.3f}'])
    d.table(rows,[67,191,31,191,d.width-480],d.tiny)
    d.p('«+» — попадание в Top-5 цели; нет «+» не означает отсутствие интереса или любого просмотра. Каталог ниже даёт название и категорию для каждого ID в таблице. '
        'Metadata получены после предсказаний для объяснения; будущие содержательные поля не участвуют в обучении.',small=True)
    items=sorted(catalog.items());half=(len(items)+1)//2
    rows=[['ID · название · категория','ID · название · категория']]
    for i in range(half):
        row=[]
        for j in [i,i+half]:
            if j<len(items):
                item,r=items[j];row.append(f'<b>{item}</b> · {e(r["title"])} · {e(r["category"])}')
            else:row.append('')
        rows.append(row)
    d.table(rows,[d.width/2,d.width/2],d.tiny)

    d.page_start('Стратегия и итоговая посылка');d.heading('8. Выбор для платформы')
    d.p(f'Для этой офлайн-задачи выбран {e(selection["method"])}: выше среднее историческое качество, чем у отдельных обязательных методов и v2. '
        f'У A итоговый AP={ex[0]["ap5"]["nested_final"]:.3f}, у B — {ex[1]["ap5"]["nested_final"]:.3f}; '
        'сравнение с Repeat и v2 приведено на тех же пользователях. У A преобладает детский контент, но есть новости; добавление новостей в v1/v2 ухудшало AP относительно Repeat. '
        'У B одного утреннего эфира мало для уверенного профиля. Эти примеры показывают индивидуальные ограничения, а не доказывают эффект для каждого зрителя.')
    d.p('CBF переносит общие атрибуты, AR/cosine — совместную аудиторию, Repeat — устойчивые повторные предпочтения. Временной контекст учитывает доступность эфиров и привычки, '
        'HGB учится сочетать косвенные сигналы по прошлым задачам прогноза. Проверенный ALS уступил; это не доказательство непригодности всех MF. '
        'Новые ID не обязательно полезные открытия; отсутствие в логах не означает «никогда не смотрел». Годовая сезонность, состав домохозяйства и текущий контекст зрителя неизвестны.')
    d.p('Для реальной платформы требуется независимая следующая временная проверка, отдельно по типам истории и доле детских просмотров, с контролем повторов и нерелевантных новостей. '
        'Расписание покрыто ID лишь частично, горизонты 6–10 недель отличаются от тестовых 12. Никаких A/B-тестов, онлайн-эффектов или влияния на выручку не измерялось.',small=True)
    d.heading('9. Точный файл и Kaggle')
    if outcome:
        d.p(f'<b>Private {outcome["private_score"]:.5f}; public {outcome["public_score"]:.5f}; {e(outcome["status"])}.</b> '
            f'Отправлено {e(outcome["submitted_at_msk"])} МСК. Параметры выбраны до отправки; скачанный обратно CSV совпал по SHA-256.',small=True)
        d.image(RUN/'kaggle_private_score_detail.jpg')
    else:
        d.p('<b>ЧЕРНОВИК: v3 ещё не отправлена / результат не подтверждён. Обязательного скриншота этой версии нет; этот PDF не готов к сдаче.</b> '
            'Последний подтверждённый private score 0,43003 относится к другому CSV — v2. Цель 0,45 не достигнута подтверждёнными данными.')
    d.p('Файл: runs/v3_candidate/submission_v3.csv. 1260 строк + header, 2 столбца, ровно 5 уникальных положительных ID в исходном порядке шаблона; '
        'все ID в буквальном тестовом расписании.<br/>SHA-256: '+checks[0]['sha256'],small=True)
    d.p('Репозиторий: <link href="https://github.com/dryjins/RecSys-LLMs">github.com/dryjins/RecSys-LLMs</link>, каталог MidTerm. '
        '<b>Пока работа локальная по просьбе студента; публикация кода и точного CSV ещё требуется.</b> Через канал курса: ровно PDF и оригинальный AI-export. '
        'Сессия продолжается; окончательный экспорт фиксируется после работы.',small=True)
    d.p('Личная интерпретация студента: «'+e(personal['discussion_v2'])+'»',small=True)

    d.page_start('Источники и происхождение материалов');d.heading('Источники и использованная помощь')
    sources=[('1','LLM4Rec. Midterm Quiz.pptx','Прочитаны текст и изображения слайдов 2–9: методы, MAP@5, формат и скриншот.'),
             ('2','exam_quiz_FKN.md; ../homework_guidelines.pdf','Прочитаны условия экзамена и HW, включая Gate 0, личное объяснение и оригинальную AI-сессию. Строгий формат: 5 основных страниц + источники.'),
             ('3','https://www.kaggle.com/competitions/sweettv-tv-program-recommender','Авторизованные Data / Overview / Rules / Submissions: канал, Top-5 с ≥80%, public/private по пользователям, правила приняты. Перед третьей попыткой форма показала 98 remaining.'),
             ('4','https://yifanhu.net/PUB/cf.pdf','Hu, Koren, Volinsky. Collaborative Filtering for Implicit Feedback Datasets. Прочитаны основная часть §4, формулы 3–6; собственная реализация ALS.'),
             ('5','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html','Официальная документация estimator, параметры fit и отключение случайного early stopping.'),
             ('6','https://scikit-learn.org/stable/modules/feature_extraction.html','Официальная документация DictVectorizer / TF-IDF, формирование разреженных атрибутов.'),
             ('7','https://docs.scipy.org/doc/scipy/reference/sparse.html','Официальная документация CSR и матричных операций.'),
             ('8','https://duckdb.org/docs/2.0/data/csv/overview','Официальная документация CSV ingestion; типы ключей задаются явно.')]
    for number,title,note in sources:
        link=f'<link href="{e(title)}">{e(title)}</link>' if title.startswith('http') else e(title)
        d.p(f'<b>[{number}]</b> {link}<br/>{e(note)}',small=True,gap=9)
    d.p('Документация прочитана для использованных операций; сторонний рекомендательный код, дополнительные данные и готовые модели не подключались. '
        'AI-помощь и собственные рассуждения студента разделены в report/personal_verification.json. Журнал всех расширений: report/EXPERIMENT_JOURNAL.md; '
        'проверка ограничений: report/RULES_AUDIT.md. Материалы PDF генерируются из фактических JSON; отсутствие score не заменяется вымышленным числом.',small=True)
    d.done();print(base.OUT)

if __name__=='__main__':build()
