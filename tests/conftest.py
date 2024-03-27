import pytest


def normalize_for_bm(text: str):
    import re

    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    stopwords_en = set(stopwords.words("english"))
    text = re.sub("[^a-zA-z1-9]", " ", text)
    words = text.lower().split()
    words = [word for word in words if word not in stopwords_en]
    return [word for word in words if len(word) > 2]


def normalize_corpus(corpus):
    return [normalize_for_bm(txt) for txt in corpus]


@pytest.fixture
def corpus():
    return normalize_corpus(
        [
            "The quick brown fox jumps over the lazy dog dog",
            "The lazy dog is brown",
            "The fox is brown",
            "Hello there good man!",
            "It is quite windy in London",
            "How is the weather today my man?",
        ]
    )
