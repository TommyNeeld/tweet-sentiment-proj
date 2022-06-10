from enum import Enum
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import json

from concurrent.futures import ProcessPoolExecutor
import asyncio
from functools import partial

import config
from inference import sentiment_inference_transformer, sentiment_inference_tfidf_bow
from preprocess import preprocess_tweet, clean_for_bow


class ModelChoices(Enum):
    pretrained = "pretrained-finiteautomata__bertweet-base-sentiment-analysis"
    tfidf = "custom-tfidf"


class Tweet(BaseModel):
    text: str
    model: ModelChoices = ModelChoices.pretrained


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict/")
async def create_item(tweet: Tweet):
    # clean tweet
    cln_tweet = tweet.text
    if config.USE_CLEANED_TWEET:
        cln_tweet = preprocess_tweet(cln_tweet)

    # store in shared location - return unique ref.
    name = f"/storage/{str(uuid.uuid4())}.json"

    # run in background and store in shared location
    asyncio.create_task(async_tweet_sentiment(name, cln_tweet, tweet.model))
    return {"name": name}


async def async_tweet_sentiment(name: str, text: str, model: str):
    # initiate async predictions
    executor = ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    if model == ModelChoices.pretrained:
        await event_loop.run_in_executor(executor, partial(tweet_sentiment_pretrained, text, name))
    elif model == ModelChoices.tfidf:
        await event_loop.run_in_executor(executor, partial(tweet_sentiment_tfidf, text, name))
    else:
        raise Exception("Model type not suppported")


def tweet_sentiment_pretrained(text: str, name: str):
    # make a prediction and then dump in shared location
    predictions = sentiment_inference_transformer(text)
    with open(name, "w") as fp:
        json.dump(predictions, fp)


def tweet_sentiment_tfidf(text: str, name: str):
    # make a prediction and then dump in shared location
    text_for_bow = clean_for_bow(text)
    predictions = sentiment_inference_tfidf_bow(text_for_bow)
    with open(name, "w") as fp:
        json.dump(predictions, fp)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
