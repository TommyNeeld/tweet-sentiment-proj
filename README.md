# Mini Tweet sentiment analysis project

## The data
This is a dataset from kaggle.com containing tweets about products and the sentiment of the tweet

## The application
This is a quick front and backend to support text sentiment analysis.

The frontend is a Streamlit application for entering text and selecting a model.

The backend is a FastAPI endpoint, it either calls a very simple sentiment classification model trained on the tweet sentiment dataset or a pre-trained sentiment classification model pulled from HuggingFace.

There is a shared file storage location between the front and back to allow for a simple async response and polling type operation. In production, the 'short polling' operation would be replaced by 'long polling' or a websocket, in addition the shared file storage would likely be replaced by a DB.


## To run
```bash
cd src
docker-compose up -d
```

**Warning, images are not small ensure you have enough RAM allocated to Docker - the APP is 1.5GB, the API is 1.3GB**

(or follow `/src/Makefile`)

Go to http://localhost:8501/ for frontend

Go to http://localhost:8080/docs for backend, to call API can run:

```bash
curl -X 'POST' \
  'http://localhost:8080/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This is a good tweet",
  "model": "custom-tfidf"
}'
```
The response will be a path to a `<UUID>.json` file which, when the prediction is complete, will be stored in the shared `/storage` volume of the container. Accessible by running `docker exec -it src_app_1 bash` `>ls /storage/` `>cat <UUID>.json`

## Inspiration
- Backend and frontend code insipired by [this arcile](https://testdriven.io/blog/fastapi-streamlit/) by [Amal Shaji](https://github.com/amalshaji)
- Sentiment analysis using huggingface inspired by this [article](https://huggingface.co/blog/sentiment-analysis-python) by HuggingFace :hugs:
