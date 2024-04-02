# yeabm25
Yet Another BM25 algorithm implementation with helpful implementation of:
1. functionallity to update the index with .update() method. In fack you can use just update. 
2. per document vector.

```python
from yeabm25 import YeaBM25
import nltk 
nltk.download('stopwords', quiet=True)
stopwords_en = set(stopwords.words('english'))

def normalize_for_bm(text: str):
    text = re.sub("[^a-zA-z1-9]", " ", text)
    words = text.lower().split()
    return [word for word in words if word not in stopwords_en]

corpus = ["The quick brown fox jumps over the lazy dog",
          "The lazy dog is brown",
          "The fox is brown",
          "Hello there good man!",
          "It is quite windy in London",
          "How is the weather today man?",
          ]
normalized_corpus = [normalize_for_bm(txt) for txt in corpus]

# fitting the whole corpus
yeabm = YeaBM25(epsilon=0.25)
yeabm.fit(normalized_corpus)

# fit and then update 
bm_update = YeaBM25(epsilon=0.25)
bm_update.fit(normalized_corpus[:3])
bm_update.update(normalized_corpus[3:])

assert yeabm.doc_len == bm_update.doc_len
assert yeabm.average_idf == bm_update.average_idf
assert yeabm.idf == bm_update.idf
assert yeabm.get_scores(['fox', 'jump']) == bm_update.get_scores(['fox', 'jump'])).all()
```

This work is inspired(and uses some code and ideas) by this great package - https://github.com/dorianbrown/rank_bm25/tree/master.
The main focus is creating document and query vectors (supports sparse vectors). Then using the vectors with your favourite Vector DB.

How to get the document and query vectors: 
```python
# recommended approach for large corpus, returns iterator. Each element is list[float]
# returns generator object
yeabm.iter_document_vectors()
# use it 
for vector in yeabm.iter_document_vectors():
    # dostuff could be put in DB. 
    dostuff(vector)

query = ...
yeabm.encode_query(query)
```

Why would you want to do that? Essentially the BM25 score formula is a sum, so it is a perfect candidate for one of the metrics any DB
supports - inner product.
```python
# 
bm_index.get_scores(['quick', 'fox'])
# ~ [1.30, 0.0, 0.72, 0.0, 0.0, 0.0]

# you get the same scores like so:
yeabm.get_embeddings() @ np.asarray(yeabm.encode_query(['fox', 'quick']))
# ~ [1.30, 0.0, 0.72, 0.0, 0.0, 0.0]
```
Of course you would like to leave the last calculation to the Vector DB.

One more opinionated implementation is that words that are found in more than half of the corpus would not have idf of 0. It would be small 
but still positive. For example in other implementations:

```python
from rank_bm25 import BM25Okapi
okapi = BM25Okapi(normalized_corpus)
okapi.get_scores(['brown']) 
# [0. 0. 0. 0. 0. 0.]
# where 
yeabm.get_scores(['brown'])
#[0.18 0.28 0.33  0. 0. 0.]
# this is helpful if the user is looking for a term that is abundant in the corpus and would still get somewhat useful results
# where with BM25Okapi you would get essentially random results (or no results).
```