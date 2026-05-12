# Как запустить проект в VS Code и задеплоить на Render + Streamlit Cloud

**Модель в бэкенде:** OpenAI **CLIP-ViT-B/32** в режиме zero-shot classification. Веса (`pytorch_model.bin`, ~600 МБ) и токенизатор подтягиваются автоматически из HuggingFace Hub при первом запуске и кешируются. Словарь меток — `backend/labels.txt` (≈1400 классов = ImageNet-1k + Places365 + ручные категории). Это значит, что модель различает не только объекты (банан, собака, пицца), но и сцены (лес, пляж, кухня, портрет).

Если понадобится откатиться на Fashion MNIST модель из Раздела 1, она лежит в `backend/best_classification_model.keras` — для неё нужен будет отдельный API (текущий main.py написан только под CLIP).

---

## 1. Открыть проект в VS Code

```bash
unzip practical10.zip
cd practical10
code .
```

VS Code предложит установить рекомендованные расширения — согласись:
- **Python** (`ms-python.python`)
- **Python Debugger** (`ms-python.debugpy`)
- **Jupyter** (`ms-toolsai.jupyter`)

## 2. Создать виртуальное окружение и поставить зависимости

В терминале VS Code (`` Ctrl+` ``):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# CPU-only PyTorch (легче, ~200 МБ вместо ~2 ГБ с CUDA)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

В правом нижнем углу VS Code выбери интерпретатор `.venv` (или Ctrl+Shift+P → "Python: Select Interpreter" → ".venv").

## 3. Запустить через F5 (отладчик)

В `.vscode/launch.json` уже настроены три конфигурации. Открой вкладку **Run and Debug** (Ctrl+Shift+D) и выбери:

| Конфигурация | Что делает |
|---|---|
| `FastAPI backend (uvicorn)` | Запускает API на http://localhost:8000 |
| `Streamlit frontend` | Запускает интерфейс на http://localhost:8501 |
| `Backend + Streamlit` | Запускает обе сразу |

Нажми F5. Первый запуск backend займёт ~30 секунд (скачивает CLIP с HuggingFace и считает текстовые эмбеддинги меток); последующие — 5–10 секунд.

## 4. Открыть ноутбук в VS Code

Открой `Practical_10.ipynb` — VS Code сам подхватит ядро из `.venv`. Все ячейки уже выполнены, можно просто пролистать.

## 5. Запустить вручную (без VS Code)

```bash
# Терминал 1 — backend
cd backend
uvicorn main:app --reload --port 8000

# Терминал 2 — streamlit
cd streamlit_app
streamlit run streamlit_app.py
```

Открой http://localhost:8501.

---

## 6. Деплой на Render.com (бэкенд FastAPI)

> **Требования к памяти:** CLIP-ViT-B/32 + PyTorch + ~1400 текстовых эмбеддингов держат в RAM ~1.2–1.6 ГБ. Бесплатный план Render (512 МБ) **не потянет**. Нужен платный план: либо **Starter ($7/мес, 512 МБ)** — будет работать на грани OOM, либо **Standard ($25/мес, 2 ГБ)** — стабильно. В `backend/render.yaml` стоит `plan: standard` — поменяй на `starter`/`free`, если хочешь сэкономить (но возможны падения).

### Шаг 1. Запушь проект в GitHub

```bash
cd practical10
git init
git add .
git commit -m "Practical 10: universal image classifier"
git branch -M main
# создай пустой репозиторий на GitHub (через сайт)
git remote add origin https://github.com/<твой_username>/practical10.git
git push -u origin main
```

### Шаг 2. Создай Web Service на Render

1. Зарегистрируйся на https://render.com и привяжи GitHub-аккаунт.
2. Внизу слева **+ New** → **Web Service**.
3. Выбери репозиторий `practical10` → **Connect**.
4. Заполни форму:

   | Поле | Значение |
   |---|---|
   | Name | `image-classifier` (или что хочешь) |
   | Language | `Python 3` |
   | Branch | `main` |
   | Root Directory | `backend` |
   | Build Command | `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Instance Type | `Standard` (или `Starter`, см. выше про память) |

5. В разделе **Environment Variables** добавь:
   - `PYTHON_VERSION` = `3.11.10`
   - `HF_HOME` = `/opt/render/project/src/.hf_cache`
   - `TRANSFORMERS_VERBOSITY` = `error`

6. Внизу: **Health Check Path** = `/health`.

7. Нажми **Create Web Service**. Сборка займёт ~10–15 минут (первый раз качается PyTorch + CLIP).

8. Когда статус станет **Live**, скопируй URL вида `https://image-classifier-XXXX.onrender.com`.

### Шаг 3. Проверь работу бэкенда

```bash
curl https://image-classifier-XXXX.onrender.com/health
# {"status":"ok","model_loaded":true,"num_labels":1397,"device":"cpu"}

curl -X POST -F "file=@/path/to/photo.jpg" \
     https://image-classifier-XXXX.onrender.com/predict
```

### Альтернатива: Blueprint (через render.yaml)

Если не хочешь заполнять форму руками — у нас лежит `backend/render.yaml`. На Render → **+ New** → **Blueprint** → выбери репозиторий, и Render сам прочитает yaml и создаст сервис. Только в Free плане Blueprint работает только для платных тарифов, поэтому план в yaml поставлен `standard`.

---

## 7. Деплой на Streamlit Community Cloud (фронтенд)

> Streamlit Cloud — это бесплатный сервис для деплоя Streamlit-приложений (1 ГБ RAM, 1 CPU). Тебе нужен только фронтенд там; бэкенд при этом крутится на Render (см. п.6).

### Шаг 1. Убедись, что код уже в GitHub

(Если делал п.6 шаг 1 — пропусти.)

### Шаг 2. Зарегистрируйся и привяжи репо

1. Зайди на https://share.streamlit.io и войди через GitHub.
2. Нажми **New app** → **Deploy a public app from GitHub**.
3. Заполни форму:

   | Поле | Значение |
   |---|---|
   | Repository | `<твой_username>/practical10` |
   | Branch | `main` |
   | Main file path | `streamlit_app/streamlit_app.py` |
   | App URL | подбери поддомен, напр. `my-classifier` |

4. Раскрой **Advanced settings** → **Secrets** и добавь:

   ```toml
   API_URL = "https://image-classifier-XXXX.onrender.com/predict"
   ```

   Это переменная окружения, которую `streamlit_app.py` подхватывает как дефолтный API URL. Можно не задавать, тогда пользователь введёт URL руками в сайдбаре.

5. Нажми **Deploy**. Сборка займёт 2–3 минуты — Streamlit Cloud прочитает `streamlit_app/requirements.txt`.

6. Когда статус **Running**, твой URL — `https://my-classifier.streamlit.app` (его сразу видно вверху панели).

### Шаг 3. Проверь работу

Открой `https://my-classifier.streamlit.app`, загрузи картинку — Streamlit пошлёт её на Render, получит топ-5 и нарисует график.

> **Холодный старт:** Render на Free/Starter плане усыпляет сервис после 15 минут простоя. Первый запрос после простоя занимает 30–60 секунд (поднимается контейнер + грузится CLIP). Последующие запросы — 0.5–2 секунды.

---

## 8. После деплоя — обнови ноутбук

Открой `Practical_10.ipynb` в VS Code, найди ячейку **«ССЫЛКА НА БЭКЕНД»** в Разделе 2 и впиши свой Render URL. Если деплоил и Streamlit Cloud — добавь её URL ниже.

---

## 9. Локальный Docker (опционально)

```bash
cd backend
docker build -t image-classifier .
docker run -p 8000:8000 image-classifier
```

---

## 10. Проверка работоспособности API

```bash
# здоровье
curl https://<твой-URL>/health

# классификация
curl -X POST -F "file=@path/to/image.png" https://<твой-URL>/predict
```

Ожидаемый ответ `/predict` (пример с фото леса):
```json
{
  "predicted_class": "broadleaf forest",
  "confidence": 0.3992,
  "top_k": [
    {"label": "broadleaf forest", "confidence": 0.3992},
    {"label": "tree farm",        "confidence": 0.2169},
    {"label": "forest path",      "confidence": 0.1279},
    {"label": "forest road",      "confidence": 0.0216},
    {"label": "wild field",       "confidence": 0.0139}
  ]
}
```
