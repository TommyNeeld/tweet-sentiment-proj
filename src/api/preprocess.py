import re
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords

stopwords = stopwords.words("english")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def _strip_links(text):
    link_regex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ", ")
    return text


def _strip_all_entities(text):
    entity_prefixes = ["@", ".@", "#", ".#"]
    # replace all other punctuation with a space
    # for separator in string.punctuation:
    #     if separator not in entity_prefixes:
    #         text = text.replace(separator, " ")
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return " ".join(words)


def preprocess_tweet(raw_text: str) -> str:
    return _strip_all_entities(_strip_links(raw_text))


def clean_for_bow(text):
    cln_text = []
    for word in nltk.word_tokenize(text):
        # lower
        word = word.lower()
        # remove non-alpha
        word = re.sub("[^A-Za-z]+", "", word)
        # stop word removal
        if word in stopwords:
            continue
        # lemmatize
        word = lemmatizer.lemmatize(word)
        word = word.lstrip()
        if word:
            cln_text.append(word)

    return " ".join(cln_text)
