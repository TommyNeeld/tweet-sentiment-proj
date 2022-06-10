import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import streamlit as st

warnings.simplefilter(action="ignore")
TWEET_DATA = "data/01_raw/product_sentiment.csv"

LABEL_MAPPING = {
    "No emotion toward brand or product": {
        "alt_label": "NEU",
        "class": 1,
    },
    "Positive emotion": {
        "alt_label": "POS",
        "class": 2,
    },
    "Negative emotion": {
        "alt_label": "NEG",
        "class": 0,
    },
}


@st.cache
def get_tweet_data() -> pd.DataFrame:
    tweet_data = pd.read_csv(TWEET_DATA)
    tweet_data.rename(columns={"is_there_an_emotion_directed_at_a_brand_or_product": "label"}, inplace=True)
    tweet_data = tweet_data.copy()[tweet_data["label"] != "I can't tell"]

    def _rename_labels(row):
        row["alt_label"] = LABEL_MAPPING[row["label"]]["alt_label"]
        row["class"] = LABEL_MAPPING[row["label"]]["class"]
        return row

    tweet_data = tweet_data.apply(_rename_labels, axis=1)
    return tweet_data


@st.cache
def get_figure(tweet_data: pd.DataFrame):
    # product sentiment
    product_setiment = (
        tweet_data.groupby(["emotion_in_tweet_is_directed_at", "alt_label"])["label"].count().to_frame().reset_index()
    )

    # plotly Marimekko Chart
    val_counts = tweet_data["emotion_in_tweet_is_directed_at"].value_counts()
    labels = val_counts.index.tolist()
    widths = val_counts.values

    # cut labels
    max_label_len = 12
    labels_cut = [f"{label[0:max_label_len]}..." if len(label) > max_label_len + 3 else label for label in labels]

    data = {"NEG": [], "NEU": [], "POS": []}
    # this could be done better!
    for label in labels:
        filter_sentiment = product_setiment[product_setiment["emotion_in_tweet_is_directed_at"] == label]
        for sentiment in data.keys():
            filter_sentiment["label_scale"] = 100 * filter_sentiment["label"] / filter_sentiment["label"].sum()
            sentiment_count = filter_sentiment[filter_sentiment["alt_label"] == sentiment]["label_scale"].values[0]
            data[sentiment].append(sentiment_count)

    fig = go.Figure()
    for key in data:
        fig.add_trace(
            go.Bar(
                name=key,
                y=data[key],
                x=np.cumsum(widths) - widths,
                width=widths,
                offset=0,
                customdata=np.transpose([labels, widths * data[key]]),
                texttemplate="%{y:.1f}%",
                textposition="inside",
                textangle=0,
                textfont_color="white",
                hovertemplate="<br>".join(
                    [
                        "Total number of device tweets: %{width}",
                        "device: %{customdata[0]}",
                        "proportion: %{y:.2f}%",
                    ]
                ),
            )
        )

    fig.update_xaxes(
        tickvals=np.cumsum(widths) - widths / 2,
        ticktext=["%s - %d tweets" % (l, w) for l, w in zip(labels_cut, widths)],
        tickangle=45,
    )

    fig.update_xaxes(range=[0, widths.sum()])
    fig.update_yaxes(range=[0, 100])

    fig.update_layout(
        title_text="Product sentiment visualisation - from 8937 tweets",
        barmode="stack",
        uniformtext=dict(mode="hide", minsize=10),
        height=500,
        yaxis_title="Sentiment proportion (%)",
    )
    return fig
