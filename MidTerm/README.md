# SWEET.TV — экзамен LLM4Rec / ФКН

**Дедлайн: 1 ноября 2026, 19:00 МСК. Работа индивидуальная.**

Лучшая из шести реальных посылок — **v5: private 0.44891, public 0.45937**.
Дополнительная цель private 0.45 не достигнута; это не обязательный порог экзамена.
Место в экзаменационной когорте неизвестно, frozen leaderboard не определяет оценку.

## Актуальные файлы

| Материал | Путь |
|---|---|
| Точный CSV выбранной посылки | [`runs/v5_candidate/submission_v5.csv`](runs/v5_candidate/submission_v5.csv) |
| Отчёт: 5 основных страниц + источники | [`output/pdf/nikolai_evdokimov_midterm_report_FINAL.pdf`](output/pdf/nikolai_evdokimov_midterm_report_FINAL.pdf) |
| Реальный score, дата, screenshot и download-back | [`runs/v5_candidate/kaggle_result.json`](runs/v5_candidate/kaggle_result.json) |
| Выбор файла и SHA-256 отчёта | [`runs/final_review/designation.json`](runs/final_review/designation.json) |
| Все 188 поисковых строк, включая references | [`runs/final_review/full_search_summary.csv`](runs/final_review/full_search_summary.csv) |
| Подробный чек-лист сдачи | [`report/READINESS.md`](report/READINESS.md) |

SHA-256 выбранного CSV:

```text
344c50703e3a6e2e4851e04577305c70538c3ef3687cda55409d08ef6f86b13f
```

`artifacts/final_submission.csv` и PDF без суффикса `_FINAL` относятся к **первой** посылке.
Версии v6 и v7 получили private 0.44767 и 0.44583; итоговой остаётся v5.
V3 проверена локально, но не отправлялась. Все отправленные CSV сохранены неизменными.

## Что осталось перед сдачей

- [x] Код и точный CSV опубликованы локальным Git в [nnevdokimov/RecSys-LLMs — MidTerm](https://github.com/nnevdokimov/RecSys-LLMs/tree/main/MidTerm).
  Коммит реализации `173aed2e4424`; удалённый CSV скачан из этого коммита и совпал побайтово и по SHA-256.
- [x] Ссылка и подтверждённый статус публикации обновлены в PDF.
- [ ] До сдачи согласовать с преподавателем передачу AI-журнала больше 10 MB: это прямо требует FAQ HW-правил; текущий оригинал около 39 MiB. Не редактировать и не сжимать его без согласованного способа передачи.
- [ ] После последнего сообщения AI-сессии сформировать её оригинальный экспорт и комплект из двух файлов.
- [ ] Передать в канал курса **ровно PDF и оригинальный AI-export**. CSV должен быть в GitHub, а не третьим файлом.

Публикация подтверждена; окончательный AI-export и отправка в канал курса ещё не подтверждены.
Запись фактической проверки GitHub: `runs/final_review/publication.json`.

## Воспроизведение выбранной v5

Python 3.12; версии библиотек закреплены в `requirements.txt`.
Из корня `MidTerm`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m sweettv.prepare
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python scripts/reproduce_final_ranker.py
```

Перед `prepare` положите пять оригинальных Kaggle CSV в `sweettv-tv-program-recommender/`:
`dataset11-30.csv`, три `export_arh_*-final.csv` и исходный `submission.csv`.
Принять правила и скачать данные нужно своим аккаунтом Kaggle. Данные, DuckDB, feature cache
и AI-export исключены из Git. Для базы и временных файлов нужны несколько GB свободного диска.
На выполненном Mac для LightGBM использована уже установленная libomp.

Скрипт обучает финальный ranker заново на разрешённой истории, формирует
`tmp/reproduced_submission_v5.csv` и проверяет его SHA-256 и побайтовое совпадение
с замороженным submission. Порядок 1260 пользователей, пять уникальных положительных ID
и общий допустимый пул проверяются. Для этого запуска не нужны архивные per-user predictions
и повторение всего адаптивного поиска.

Воспроизведение отчёта и локальная проверка комплекта:

```bash
PYTHONPATH=src python scripts/report_final_best.py
python scripts/finish_local_delivery.py --check-only
```

Для PDF нужны ReportLab и шрифт с кириллицей; путь можно задать через `SWEETTV_FONT_DIR`.
Визуальная проверка выполнялась через Poppler (`pdftoppm`).
Среда и реально выполненные проверки: `runs/final_review/environment.json`,
`runs/final_review/package_checks.json`, `runs/v5_candidate/reproduction.json`,
`runs/v5_candidate/standalone_reproduction.json`, `runs/lambdarank_tests.log`.

## Методы и протокол

Реализованы отдельно popularity, CBF, Association Rules, Repeat, собственный implicit ALS
и дополнительные гибриды. Итоговая v5 — candidate generation + LightGBM LambdaRank,
86 признаков, 150 деревьев / 15 листьев, seed 42. Цель обучения v5 — бинарная
принадлежность будущему Top-5. Graded gains относятся только к невыбранной v7.

Логи соединяются с расписанием по каналу и пересечению полуоткрытых интервалов.
Фрагменты user–airing объединяются; квалифицированный просмотр — покрытие ≥80% эфира.
ID=0 означает неразмеченный эфир и не является item. Повторные эфиры считаются отдельно.
`user_id` хранится строкой; ненадёжный `duraton` заменён разницей timestamp.

История начинается 09.03.2020. Границы / конец прогноза трёх срезов:
20.04 / 29.06, 18.05 / 27.07, 15.06 / 27.07. Финальная история заканчивается 27.07;
тестовое окно — 27.07–19.10.2020. Все признаки и обучающие labels вложенной схемы
ограничены соответствующей обучающей границей.

У методов общие пользователи, legal pool, фильтры и popularity fallback; ранее просмотренные
ID разрешены. Будущее расписание даёт ID, канал и время/длительность/число эфиров.
Будущие просмотры и содержательные metadata не входят в scoring. Основная MAP@5 исключает
пустой ground truth; дополнительная считает его AP=0. Знаменатель AP — `min(5, |R|)`.

На срезе 18.05–27.07: popularity 0.26970, CBF 0.29400, AR 0.26694, v5 0.42362.
Среднее v5 по трём срезам — 0.420119. Поиск широкий, срезы перекрываются;
независимая значимость прироста не заявлена. Резервная группа не подтвердила преимущества.
У двух выбранных пользователей ошибки и худшие AP также показаны, примеры не подменены.

Полная история экспериментов, включая проигравшие: [`report/EXPERIMENT_JOURNAL.md`](report/EXPERIMENT_JOURNAL.md).
Проверка требований и неоднозначностей: [`report/RULES_AUDIT.md`](report/RULES_AUDIT.md).
Личные ответы студента отделены от действий AI в `report/personal_verification.json`.
Файлы в `runs/*/source_snapshot/` — исторические версии, а не актуальная точка запуска.

## Оригинальный AI-export и два файла для курса

После **последнего сообщения этой сессии**, из корня проекта:

```bash
.venv/bin/python scripts/finish_local_delivery.py
```

Команда создаст ровно `output/delivery/nikolai_evdokimov_midterm_report.pdf` и
`output/delivery/nikolai_evdokimov_midterm_session.jsonl`. Она копирует нативный журнал побайтово,
проверяет идентификатор сессии и SHA-256; manifest остаётся снаружи каталога двух файлов.
Пока чат продолжается, экспорт нужно обновлять. Пересказ или искусственный JSON не подходят.

Нативный журнал имеет формат JSONL; §4.1 HW-правил разрешает tool-native формат.
**FAQ требует связаться с преподавателем до сдачи, если журнал больше 10 MB.**
Текущий оригинал около 39 MiB, поэтому этот пункт обязателен, даже если LMS принимает файл.
Согласуйте способ передачи, сохраняя исходник без изменений; не подменяйте его пересказом
и не переименовывайте в JSON-объект. AI-export и `output/delivery/` не публикуются в Git.

Правила HW доступны. Противоречие 5/6 страниц разрешено в пользу пяти основных плюс источники.
Абсолютный timezone и скрытые детали Kaggle labels не подтверждены и раскрыты в отчёте.
На слайде лимит назван «обычно 5», фактическая форма показывала 100/day; все шесть попыток раскрыты.
Онлайн-эффекты, выручка, личность зрителя и совместный просмотр ребёнка со взрослым не установлены.
