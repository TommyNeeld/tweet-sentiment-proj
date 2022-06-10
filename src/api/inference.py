from typing import Dict
import config
from transformers import pipeline
import joblib

CLASS_MAPPING = {0: "NEG", 1: "NEU", 2: "POS"}


def sentiment_inference_transformer(text: str) -> Dict:
    # transformers pipeline for sentiment classification
    sentiment_pipeline = pipeline(model=config.BASE_MODEL)
    prediction = sentiment_pipeline(text)[0]
    return prediction


def sentiment_inference_tfidf_bow(text: str) -> Dict:
    pipeline = joblib.load(config.TFIDF_MODEL_FILEMANE)
    prediction = pipeline.predict([text])[0]
    return {"label": CLASS_MAPPING[prediction]}
