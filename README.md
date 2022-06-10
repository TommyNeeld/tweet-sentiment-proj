# Mini Tweet sentiment analysis project
## The task
This is a dataset from kaggle.com containing tweets about products and the sentiment of the tweet. 

Your task is to explore the data with some visualisations, and then train and "productionalise" a sentiment analyzer that you can input some text into and it will run your sentiment analysis model on it.
- This should be completable in 1 hour, and you're free to use any resources you find online. 
- Please cite (just a link or a site name is fine) references that you feel like you're using more than just a small part of.

### Subtasks
- Explore the data and come up with a visualization of which products have the most positive and negative tweets
- Train a classifier to determine the sentiment of the tweets (Positive/Negative/Neutral)
- "deploy" the classifier using a docker container and an endpoint of "/predict" to run predictions on new tweets
  - I'd recommend using flask as a simple server

## Links to stuff I used

- Backend and frontend code insipired by [this arcile](https://testdriven.io/blog/fastapi-streamlit/) by [Amal Shaji](https://github.com/amalshaji)
- Sentiment analysis using huggingface inspired by this [article](https://huggingface.co/blog/sentiment-analysis-python) by HuggingFace :hugs:

### Tools used
- FastAPI: for the API - using async and shared file store for model inference
- Streamlit: for the frontend
- Docker: to containerize the app
### To run
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

### Next setps
- Additional analysis of results - Cross Validation of custom model
- Hyperparam tuning of custom model
- Test more models!
  - Likely get best result by fine-tuning a pre-built transformer using sentence transformers, see [this](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis?text=london+is+great) article 
  - Dataset is currently very limited
- Optimise size of docker images
- Should probably first understand the use-case