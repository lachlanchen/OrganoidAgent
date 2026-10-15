[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner — apps and four buddies](https://lazying.art/media/brand/lazyingart-intro.gif)](https://lazying.art)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent — это лёгкий backend на Tornado и Progressive Web App (PWA) для локального просмотра и предпросмотра датасетов органоидов с минимальной настройкой. Он предоставляет предпросмотр файлов по типу для таблиц, микроскопических изображений (включая TIFF), архивов, текста gzip и объектов анализа AnnData `.h5ad`.

## 🎯 Кратко

| Цель | Что даёт этот репозиторий |
|---|---|
| Исследование датасетов в local-first режиме | Поиск датасетов, метаданные и просмотр файлов из локальной рабочей директории `datasets/` |
| Богатые предпросмотры | Пути предпросмотра таблиц, изображений (включая TIFF), архивов, `.gz` и `.h5ad` |
| Дружелюбный к офлайн-работе frontend | Устанавливаемая оболочка PWA с service worker и manifest |
| Практические операции | Распаковка архивов + индексирование с фильтрацией по категориям |

## Обзор 🔭

Основное приложение создано для интерактивного изучения датасетов с минимальной настройкой:

- Backend API и механизм предпросмотров в `app.py`
- PWA frontend в `web/`
- Скрипты загрузки в `scripts/`
- Локальная рабочая область для данных в `datasets/` (игнорируется Git)

В этом репозитории также есть связанные рабочие пространства (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, подмодуль `papers`). Основной сценарий выполнения, описанный в этом README, — это top-level приложение `OrganoidAgent`.

## Функции ✨

- Локальное индексирование датасетов с суммарной информацией о размере и количестве файлов
- Рекурсивный список файлов с выводом определённого типа
- Поддержка предпросмотра таблиц CSV/TSV/XLS/XLSX
- Поддержка предпросмотра изображений TIFF/JPG/PNG
- Поддержка предпросмотра `.h5ad` с генерацией scatter-просмотра embedding/PCA
- Поддержка предпросмотра архивов ZIP/TAR/TGZ и попытка показа первого изображения
- Поддержка предпросмотра текстовых `.gz` по первым строкам
- Endpoint для распаковки архивов для больших упакованных датасетов
- Карточки метаданных датасетов, построенные из Markdown
- PWA frontend с service worker и manifest
- Базовая санитизация путей (`safe_dataset_path`) для ограничения доступа к файлам внутри `datasets/`

### Кратко

| Область | Что даёт |
|---|---|
| Открытие датасетов | Список датасетов на уровне папок с числом файлов и сводкой по размеру |
| Исследование файлов | Рекурсивный список и определение типа (`image`, `table`, `analysis`, `archive` и т.д.) |
| Богатые предпросмотры | Таблицы, TIFF/изображения, фрагменты gzip-текста, содержимое архивов, сводки AnnData |
| Визуализация анализа | Сводные scatter-просмотры `.h5ad` по embedding из `obsm` или fallback через PCA |
| Поддержка упаковок | Просмотр содержимого архивов + endpoint распаковки для крупных архивных комплектов |
| Web UX | Устанавливаемая PWA с asset-ами, удобными для офлайн-использования |

## Структура проекта 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Требования ✅

- Python `3.10+`
- Рекомендуемый менеджер окружений: `conda` или `venv`

Необходимые/опциональные Python-пакеты, вытекающие из исходников:

| Пакет | Роль |
|---|---|
| `tornado` | Обязателен для запуска сервера |
| `pandas` | Необязателен: предпросмотр таблиц |
| `anndata`, `numpy` | Необязателен: предпросмотр `.h5ad` и построение графиков анализа |
| `Pillow` | Необязателен: рендеринг изображений и созданные предпросмотры |
| `tifffile` | Необязателен: поддержка предпросмотра TIFF |
| `requests` | Необязателен: скрипты загрузки датасетов |
| `kaggle` | Необязателен: загрузки Kaggle в скрипте скрининга лекарств |

Примечание: в текущей версии нет `requirements.txt`, `pyproject.toml` или `environment.yml` для top-level приложения.

## Установка ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## Использование 🚀

### Быстрый старт

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Откройте `http://localhost:8080`.

### Проверка API (smoke test)

```bash
curl http://localhost:8080/api/datasets
```

### Загрузка данных (необязательно)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Загруженные данные лежат в `datasets/` (игнорируются Git).

## API endpoints 🌐

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/api/datasets` | Получить список датасетов со сводной статистикой |
| `GET` | `/api/datasets/{name}` | Получить список файлов для одного датасета |
| `GET` | `/api/datasets/{name}/metadata` | Вернуть метаданные в формате markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Список файлов по категориям |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Ответ предпросмотра с учётом типа файла |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Распаковать архив в соседнюю папку `_extracted` |
| `GET` | `/files/<path>` | Отдача исходного файла датасета |
| `GET` | `/previews/<path>` | Отдача сгенерированного предпросмотра |

Пример запроса предпросмотра:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Конфигурация 🧩

Текущая конфигурация runtime намеренно минимальна:

- Порт сервера: параметр `--port` в `app.py` (по умолчанию `8080`)
- Каталог данных: жёстко привязан к `datasets/` относительно корня репозитория
- Кэш предпросмотров: `datasets/.cache/previews`
- Карта метаданных: словарь `DATASET_METADATA` в `app.py`
- GitHub API token для загрузчика (опционально): переменная окружения `GITHUB_TOKEN` или `--github-token`

Примечание: если вам нужны настраиваемые корневые каталоги датасетов или production-параметры сервера, они пока не вынесены в отдельные файлы конфигурации top-level.

## Примеры 🧪

### Просмотр файлов по категории

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Распаковка архива

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Запуск выборочных режимов загрузки

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Заметки по разработке 🛠️

- Backend отдает статические файлы frontend из `web/`.
- Service worker и manifest находятся в `web/sw.js` и `web/manifest.json`.
- Маршрутизация по типам файлов и предпросмотр реализованы в `app.py`.
- Ручная проверка (текущая): PWA открывается на `http://localhost:8080`
- Ручная проверка (текущая): `/api/datasets` возвращает JSON
- Ручная проверка (текущая): предпросмотр работает для CSV/XLSX/изображений/архивов

## Устранение неполадок 🩺

- `ModuleNotFoundError` для библиотек предпросмотра: установите недостающие пакеты (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Пустой список датасетов: убедитесь, что данные есть в `datasets/`, и каталоги не начинаются с точки.
- Предпросмотр `.h5ad` без scatter-изображения: проверьте, что установлены `anndata`, `numpy` и `Pillow`.
- Проблемы с предпросмотром/распаковкой больших архивов: используйте endpoint распаковки и смотрите файлы уже после распаковки.
- Ошибки лимита GitHub API: передайте `GITHUB_TOKEN` через переменную окружения или CLI-флаг.
- Не работает загрузка с Kaggle: установите `kaggle` и настройте credentials в `~/.kaggle/kaggle.json`.

## Дорожная карта 🧭

Потенциальные улучшения в будущем (в корневом приложении пока не полностью реализованы):

- Добавить корневой manifest зависимостей (`requirements.txt` или `pyproject.toml`)
- Добавить автоматические тесты для API-хендлеров и функций предпросмотра
- Добавить настраиваемые корень датасета и параметры кэша
- Добавить явный production-профиль запуска (non-debug, рекомендации reverse-proxy)
- Расширить многоязычную документацию в `i18n/`

## Вклад 🤝

Вклад приветствуется. Практичный workflow:

1. Сделайте fork и создайте фокусированный branch.
2. Оставляйте изменения в одной логической зоне.
3. Ручной запуск: проверьте запуск приложения и ключевые endpoints.
4. Создайте PR с кратким summary, выполненными командами и скриншотами UI-изменений.

Локальные соглашения по стилю в этом репозитории:

- Python: 4 пробела отступа, snake_case для функций/файлов, классы в CapWords
- Для этого приложения держите логику frontend в `web/app.js` (не делать переписывание на фреймворках)
- Комментарии делайте лаконичными и только там, где логика неочевидна

## Каноническая структура проекта 📌

- `app.py`: Tornado server and API routes.
- `web/`: PWA assets.
- `scripts/`: dataset download helpers.
- `datasets/`: local data storage.
- `papers/`: submodule with reference materials.

## Лицензия 📄

Файл лицензии `LICENSE` на уровне корня сейчас отсутствует в репозитории.

Примечание: до добавления лицензии на top-level кодовой базе OrganoidAgent условия повторного использования/распространения остаются неуточнёнными.


## ❤️ Support

<!-- support:start -->
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsors-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)
[![LazyingArt Support](https://img.shields.io/badge/LazyingArt-Support-0EA5E9?style=for-the-badge)](https://chat.lazying.art/donate)
[![PayPal Support](https://img.shields.io/badge/PayPal-Support-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen)
[![Stripe Support](https://img.shields.io/badge/Stripe-Support-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400)
<!-- support:end -->
