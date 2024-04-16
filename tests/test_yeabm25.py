import pytest
from yeabm25 import YeaBM25


def test_doc_frequency(corpus):
    yeabm = YeaBM25(epsilon=1)
    yeabm.fit(corpus=corpus)
    assert yeabm.doc_freqs[0] == {'dog': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'lazy': 1}
    assert yeabm.doc_freqs[-1] == {'weather': 1, 'today': 1, 'man': 1}


@pytest.mark.parametrize("epsilon", [0.25, 0.75, 1])
def test_fit_udpate(corpus, epsilon):
    yeabm = YeaBM25(epsilon=epsilon)
    yeabm.fit(corpus=corpus)
    half = len(corpus) // 2
    yeabm_updated = YeaBM25(epsilon=epsilon)
    yeabm_updated.fit(corpus=corpus[:half])
    yeabm_updated.update(corpus=corpus[half:])
    assert yeabm.average_idf == yeabm_updated.average_idf
    assert yeabm.idf == yeabm_updated.idf


def test_average_idf(corpus):
    yeabm = YeaBM25(epsilon=1)
    yeabm.fit(corpus=corpus)
    # the terms that are present in more than half of the corpus would get corrected
    terms_with_idf_correction = [k for k, v in yeabm.word_df.items() if v >= yeabm.corpus_size // 2]
    for term in terms_with_idf_correction:
        assert yeabm.idf[term] == yeabm.average_idf


@pytest.mark.parametrize("epsilon", [0.25, 0.5, 0.75])
def test_top_n(corpus, epsilon):
    yeabm = YeaBM25(epsilon=epsilon)
    yeabm.fit(corpus=corpus)
    scores = yeabm.get_top_n(['brown', 'quick', 'fox'], n=3)
    assert [i for i in scores] == [0, 2, 1]
    scores = yeabm.get_top_n(['man', 'windy', 'london'], n=3)
    assert [i for i in scores] == [4, 5, 3]


@pytest.mark.parametrize("query, expected", [
    (['fox', 'quick'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (['cat', 'raven'], [0] * 14)
])
def test_encode_query_dense(corpus, query, expected):
    yeabm = YeaBM25()
    yeabm.fit(corpus)
    assert yeabm.encode_query_dense(query) == expected


@pytest.mark.parametrize("query, expected", [
    (['fox', 'lazy'], {1: 1.0, 3: 1.0}),
    (['cat', 'raven'], {})
])
def test_encode_query(corpus, query, expected):
    yeabm = YeaBM25()
    yeabm.fit(corpus)
    assert yeabm.encode_query(query) == expected


def test_fromdict_encodeonly(corpus):
    yeabm = YeaBM25()
    yeabm.fit(corpus=corpus)
    state_dict = yeabm.state_dict(state='encodeonly')
    new_yeabm = YeaBM25.from_state_dict(state_dict)

    assert yeabm.idf == new_yeabm.idf