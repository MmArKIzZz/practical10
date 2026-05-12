"""Streamlit интерфейс для универсального классификатора изображений."""

import io
import os

import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = os.environ.get('API_URL', 'http://localhost:8000/predict')

st.set_page_config(
    page_title='Universal Image Classifier', layout='centered'
)

st.title('Универсальный классификатор изображений')
st.caption('CLIP-ViT-B/32 · zero-shot · ~1400 классов (объекты + сцены)')
st.markdown(
    'Загрузите любое изображение — модель определит, что на нём, '
    'и покажет топ-5 наиболее вероятных классов.'
)


def _pretty(label: str) -> str:
    """`Egyptian_cat` → `Egyptian cat`, capitalize first letter."""
    s = label.replace('_', ' ').strip()
    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    return s


api_url = st.sidebar.text_input('API URL', value=DEFAULT_API_URL)
st.sidebar.caption(
    'Эндпоинт FastAPI, который принимает изображение и возвращает классы.'
)

uploaded = st.file_uploader(
    'Выберите изображение', type=['png', 'jpg', 'jpeg', 'bmp', 'webp']
)

image_to_classify = None
if uploaded is not None:
    image_to_classify = Image.open(uploaded)
    st.image(image_to_classify, caption='Загруженное изображение', width=320)

if image_to_classify is not None and st.button('Классифицировать'):
    buf = io.BytesIO()
    image_to_classify.convert('RGB').save(buf, format='PNG')
    buf.seek(0)

    with st.spinner('Отправка на сервер...'):
        try:
            resp = requests.post(
                api_url,
                files={'file': ('image.png', buf, 'image/png')},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            st.success(
                f'**Предсказанный класс:** `{_pretty(data["predicted_class"])}`'
            )
            st.metric('Уверенность', f'{data["confidence"] * 100:.2f}%')

            top = data.get('top_k', [])
            if top:
                st.subheader('Топ-5 классов')
                labels = [_pretty(t['label']) for t in top][::-1]
                values = [t['confidence'] for t in top][::-1]
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#4CAF50' if i == len(values) - 1 else '#2196F3'
                          for i in range(len(values))]
                ax.barh(labels, values, color=colors)
                ax.set_xlabel('Уверенность')
                ax.set_xlim(0, 1)
                for i, v in enumerate(values):
                    ax.text(v + 0.01, i, f'{v*100:.1f}%', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
        except requests.exceptions.ConnectionError:
            st.error(f'Не удалось подключиться к API: {api_url}')
        except Exception as e:
            st.error(f'Ошибка: {e}')
