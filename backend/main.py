"""FastAPI backend для универсального классификатора изображений.

Использует OpenAI CLIP-ViT-B/32 в режиме zero-shot classification.
В отличие от чистой ImageNet-модели, CLIP различает не только объекты
(собака, банан, пицца), но и сцены (лес, пляж, город, кухня и т.д.).
Словарь меток лежит в `labels.txt` — это объединение ImageNet-1k,
Places365 и нескольких ручных категорий (всего ~1400 классов).

При старте сервер один раз считает текстовые эмбеддинги всех меток
и затем при каждом запросе кодирует только изображение — это быстро.
"""

import io
import os
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = os.environ.get('MODEL_NAME', 'openai/clip-vit-base-patch32')
LABELS_PATH = os.environ.get(
    'LABELS_PATH',
    os.path.join(os.path.dirname(__file__), 'labels.txt'),
)
TOP_K = int(os.environ.get('TOP_K', '5'))
PROMPT_TEMPLATE = os.environ.get('PROMPT_TEMPLATE', 'a photo of {label}')

app = FastAPI(
    title='Universal Image Classifier API',
    description=(
        'Универсальный zero-shot классификатор изображений на CLIP-ViT-B/32. '
        'Поддерживает и объекты (ImageNet), и сцены (Places365): '
        'лес/пляж/город/кухня/портрет и т.д. (~1400 классов).'
    ),
    version='4.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def _load_labels(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


print(f'Loading CLIP model: {MODEL_NAME}')
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

LABELS: List[str] = _load_labels(LABELS_PATH)
print(f'Loaded {len(LABELS)} labels')

def _features(out):
    """Normalize the API: transformers 5.x returns a pooled output object."""
    return getattr(out, 'pooler_output', out)


print('Precomputing text features...')
_text_inputs = processor(
    text=[PROMPT_TEMPLATE.format(label=l) for l in LABELS],
    return_tensors='pt',
    padding=True,
    truncation=True,
).to(device)
with torch.no_grad():
    _text_features = _features(model.get_text_features(**_text_inputs))
_text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
print(f'Text features ready: {tuple(_text_features.shape)}')


def predict(image: Image.Image, k: int = TOP_K) -> List[dict]:
    """Возвращает топ-K (label, confidence) для одного изображения."""
    image = image.convert('RGB')
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        img_features = _features(model.get_image_features(**inputs))
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    sims = (100.0 * img_features @ _text_features.T).softmax(dim=-1)[0]
    topk = torch.topk(sims, min(k, len(LABELS)))
    return [
        {
            'label': LABELS[i.item()],
            'confidence': round(float(v.item()), 6),
        }
        for v, i in zip(topk.values, topk.indices)
    ]


@app.get('/')
async def root():
    return {
        'message': 'Universal Image Classifier API',
        'model': f'CLIP-ViT-B/32 zero-shot ({MODEL_NAME})',
        'num_labels': len(LABELS),
        'top_k': TOP_K,
        'endpoints': {
            '/predict': 'POST image (multipart, поле file) — топ-K классов',
            '/health': 'GET healthcheck',
        },
    }


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'model_loaded': True,
        'num_labels': len(LABELS),
        'device': str(device),
    }


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    top = predict(image, k=TOP_K)
    return {
        'predicted_class': top[0]['label'],
        'confidence': top[0]['confidence'],
        'top_k': top,
    }
