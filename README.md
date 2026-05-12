# Практическая работа №10 — универсальный классификатор изображений

Раздел 1 — сравнение моделей (Fashion MNIST). Разделы 2/3 — развёрнутый FastAPI + Streamlit универсальный классификатор на базе **CLIP-ViT-B/32 (zero-shot, ~1400 классов из ImageNet + Places365)** — распознаёт **и объекты** (банан, собака, пицца), **и сцены** (лес, пляж, кухня, портрет, город).

## Структура проекта

```
.
├── Practical_10.ipynb                # ноутбук с полным выполнением задания
├── best_classification_model.keras   # лучшая модель Fashion MNIST (Раздел 1)
├── backend/                          # FastAPI — универсальный классификатор (CLIP)
│   ├── main.py
│   ├── labels.txt                    # ~1400 меток (ImageNet + Places365 + extras)
│   ├── Dockerfile, fly.toml, render.yaml, Procfile
│   ├── requirements.txt
│   └── pyproject.toml
└── streamlit_app/                    # Streamlit интерфейс
    ├── streamlit_app.py
    └── requirements.txt
```

## Раздел 1. Сравнение моделей

В ноутбуке `Practical_10.ipynb` обучаются и сравниваются четыре модели:

| # | Модель | Источник |
|---|--------|----------|
| 1 | DNN (полносвязная) | Практическая 2 |
| 2 | CNN | Практическая 3 |
| 3 | CNN + BatchNorm + Dropout | Практическая 4 |
| 4 | MobileNetV2 (Transfer Learning) | Практическая 5 |

Сравнение по метрикам: Accuracy, Precision, Recall, F1-мера, время инференса.
Лучшая модель выбирается по F1-мере и сохраняется в `best_classification_model.keras`.

## Раздел 2. FastAPI backend (универсальный классификатор)

Бэкенд использует **CLIP-ViT-B/32** в режиме zero-shot classification. На старте сервер скачивает веса CLIP с HuggingFace и один раз считает текстовые эмбеддинги всех меток из `labels.txt`. При каждом запросе кодируется только изображение и считается косинусная похожесть с предвычисленными текстами — это быстро (≈0.3 с на CPU).

```bash
cd backend
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Эндпоинты:
- `GET /` — описание API + имя модели
- `GET /health` — healthcheck
- `POST /predict` — отправить файл-изображение (multipart, поле `file`); ответ — топ-1 класс + топ-5 предсказаний с вероятностями

Пример:
```bash
curl -X POST -F "file=@image.png" http://localhost:8000/predict
```

## Раздел 3. Streamlit frontend

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

В сайдбаре можно указать адрес API. Загрузи картинку → нажми «Классифицировать» → получишь топ-1 класс, уверенность и горизонтальные бары для топ-5.

## Раздел 4. Развертывание

См. подробные пошаговые инструкции в `HOW_TO_RUN.md`:
- **Render.com** — бэкенд FastAPI
- **Streamlit Community Cloud** — фронтенд
- альтернативы: Fly.io / Railway / Docker

## Быстрый старт

```bash
# 1. Backend
cd backend
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Streamlit (в новом терминале)
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
# Открой http://localhost:8501
```
