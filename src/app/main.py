import streamlit as st
import requests
import os
import json
import time

from plots import get_tweet_data, get_figure


def main():
    # set page to wide layout
    st.set_page_config(layout="wide")

    # assume always running from docker-compose
    api_endpoint = "api"

    st.title("Tweet sentiment classifier")

    # get plotly figure
    with st.spinner("Loading plot..."):
        tweet_data = get_tweet_data()
        figure = get_figure(tweet_data)

    st.plotly_chart(figure, use_container_width=True)

    tweet = st.text_area("Input tweet")
    model_options = {"Pretrained HF model": 0, "Custom TF-IDF SVM": 1}
    model_selection = st.selectbox("Select model", model_options, index=0)

    st.write("*Note, predictions on the API are running asynchronously*")

    if st.button("Calculate sentiment"):
        if tweet is not None:

            st.write("Tweet:", tweet)

            # model selection message
            index = model_options[model_selection]
            holding_message = f"Calculating sentiment using {model_selection}..."
            if index == 0:
                holding_message += " This will take some time on first run (model being downloaded)"

            # spinner while waiting for response
            with st.spinner(holding_message):

                # if using other model, need to specify in request
                body = {"text": tweet}
                if index == 1:
                    body = {"text": tweet, "model": "custom-tfidf"}
                res = requests.post(f"http://{api_endpoint}:8080/predict/", json=body)
                response_path = res.json()
                path_name = response_path.get("name")

                # loop until model output path exists
                while not os.path.exists(path_name):
                    time.sleep(1)

                # open json
                with open(path_name) as json_file:
                    sentiment = json.load(json_file)
                st.write("Sentiment:", sentiment)
                st.success("Done!")


if __name__ == "__main__":
    main()
