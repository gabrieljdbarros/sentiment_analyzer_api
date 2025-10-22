# api_sentiment.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

app = FastAPI(title="Sentiment API")

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# carrega uma vez (slow tokenizer para SentencePiece)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
pipe = TextClassificationPipeline(model=model, tokenizer=tok)

class Item(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(items: List[Item]):
    texts = [it.text for it in items]
    preds = pipe(texts)
    # normaliza saída
    out = []
    for p in preds:
        label = p["label"].lower()
        if "star" in label:  # caso mude o modelo
            if "1" in label or "2" in label: label = "negative"
            elif "3" in label: label = "neutral"
            else: label = "positive"
        out.append({"label": label, "score": float(p["score"])})
    return out
