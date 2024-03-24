# yeabm25
Yet Another BM25 algorithm implementation with helpful implementation of:
1. functionallity to update the index with .update() method. 
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
bm_index = YeaBM25(epsilon=1)
bm_index.fit(normalized_corpus)

# fit and then update 
bm_update = YeaBM25(epsilon=1)
bm_update.fit(normalized_corpus[:3])
bm_update.update(normalized_corpus[3:])

print(bm_index.doc_len == bm_update.doc_len)
print(bm_index.average_idf == bm_update.average_idf)
print(bm_index.idf == bm_update.idf)
print((bm_index.get_scores(['fox', 'jump']) == bm_update.get_scores(['fox', 'jump'])).all())
```

This work is inspired(and uses some code and ideas) by this great package - https://github.com/dorianbrown/rank_bm25/tree/master.

#### Todo tests ;) 
