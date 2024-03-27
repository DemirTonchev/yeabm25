import math
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import TypeAlias, Optional

# type alias just for readability
Vector1d: TypeAlias = np.ndarray | list[float]


# experimental
# this is created with possible compatibility with haystack>2.0
@dataclass
class BMDocument:
    content: list[str]
    meta: dict = field(default_factory=dict)
    embedding: Optional[Vector1d] = None

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]

    def __iter__(self):
        for word in self.content:
            yield word

    def __repr__(self):
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content}'" if len(self.content) < 100 else f"content: '{self.content[:100]}...'"
            )
        if len(self.meta) > 0:
            fields.append(f"meta: {self.meta}")
        if self.embedding is not None:
            fields.append(f"embedding: vector of size {len(self.embedding)}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}({fields_str})"


class YeaBM25:

    """
    The class representing the Okapi-BM25 variant of the algorithm. Okapi-BM25 is a ranking function widely used as the OLD(before transformers went
    brrrrrrrrrrrrr) state of the art for Information Retrieval tasks. BM25 is a variation of the TF-IDF model.
    This implementation applies negative idf correction from here: https://arxiv.org/pdf/1602.03606.pdf#page=4. Which also sets a floor a floor on the
    idf values to eps * average_idf.


    Attributes
    ----------
    doc_freqs : list[dict[str, int]]
        Word Frequency per document. [{'cat': 1}, {'foo': 5}] means  the first document contains the term 'hi' 1 time and
        the second containts the term foo 5 times.

    word_df : dict[str, int]
        Word Document Frequency - number of documents in the corpus that contains the term.

    idf : dict[str, float]
        Inverse Document Frequency per term.

    doc_len : list[int]
        Number of terms per document. So [3, 6] means the first document contains 3 terms and second 6 terms.

    corpus_size : int
        Number of documents in the corpus.

    avgdl : float
        Average number of terms for documents in the corpus.

    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        Parameters
        ----------
        k1 : float
            Controls term frequency component.
            k = 0 ignore tf, typical values are 1.2-2.

        b : float
            Lenght normalization.
            b=1 full document lenght normalization
            b=0 no document lenght normalization
            Lenght normalization component:
            B = (1 - b) + b * doc_len/avgdl

        epsilon : float
            Correction factor
        """
        if not (k1 >= 0):
            raise ValueError(f"k1 must be positive ()>= 0), input is {k1}")
        if not (0 <= b <= 1):
            raise ValueError(f"b must be between 0 and 1, input is {b}")
        if not (epsilon > 0):
            raise ValueError(f"epsilon should be small positive number, input is {epsilon}")

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        # learned params
        self.doc_freqs = []
        self.doc_len = []  # number of words in each document
        self.avgdl: float = 0.
        # word -> number of documents with word {'cat': 50} means 'cat' is in 50 documents from the corpus
        self.word_df = defaultdict(int)
        self.idf: dict = {}
        self.average_idf: float = 0
        self.corpus_size: int = 0

    def _process_doc(self, document: list[str]) -> None:
        """Gets a documents, which is represented as a list of words(strings).
        Appends frequencies of words per document and builds the word document frequencies. Wodf df is used for idf.
        """
        frequencies = Counter(document)
        self.doc_freqs.append(frequencies)

        for word in frequencies:
            self.word_df[word] += 1

    def _calc_idf(self):

        # idf can be negative if word is contained in more than half of documents
        negative_idfs = set()
        self.idf = {}
        for word, freq in self.word_df.items():
            # https://arxiv.org/pdf/1602.03606.pdf#page=4
            # if idf is negative we set as 0, for correct running mean calculation. This is not explicit in the linked paper
            # but the implemenation is chosen as such, because IDF should not be negative.
            idf_value = max(math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5), 0)

            # dont add idf value to the average if 0
            if idf_value > 0:
                self.idf[word] = idf_value
                self.average_idf = self.average_idf + (idf_value - self.average_idf) / (len(self.idf))

            else:
                negative_idfs.add(word)

        # eps correction
        eps_correction = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps_correction

    def fit(self, corpus: list[list[str]] | list[BMDocument]):
        """Fits the index given a corpus of documents. In the case of bm25 a document is list of words(strings)
        """

        self.corpus_size: int = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        for document in corpus:
            self._process_doc(document)

        self._calc_idf()

        return self

    def update(self, corpus: list[list[str]] | list[BMDocument]):
        self.corpus_size += len(corpus)
        self.doc_len += [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size

        for document in corpus:
            self._process_doc(document)

        self._calc_idf()

        return self

    # sklearn like api
    @property
    def features_(self):
        return [word for word in self.idf]

    def get_scores(self, query: list[str]):
        score = np.zeros(self.corpus_size)
        doc_len = np.asarray(self.doc_len)
        for q in set(query):
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs])
            score += self.idf.get(q, 0) * (q_freq * (self.k1 + 1) /
                                           (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_top_n(self, query: list[str], n=5):
        """Returns the indexes of the top n documents and scores in sorted order.
        """
        scores = self.get_scores(query)
        isort = np.argsort(scores)[-n:][::-1]
        return {i: s for i, s in zip(isort, scores[isort].round(2))}

    def document_self_scores(self, idx):
        scores = {}
        for q, q_freq in self.doc_freqs[idx].items():
            s = self.idf.get(q, 0) * (
                q_freq * (self.k1 + 1) /
                (q_freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl))
            )
            scores[q] = s
        return scores

    def document_vector(self, idx):
        feats = self.features_
        scores = self.document_self_scores(idx)
        return [scores[f] if f in scores else 0. for f in feats]

    def iter_document_vectors(self):
        """Iterate over the vectors, for example if you to store them in a Vector DB.
        """
        for idx in range(len(self.doc_freqs)):
            yield self.document_vector(idx)

    def get_embeddings(self) -> np.ndarray:
        """This should be used only if the index is relatively small.
        """
        matrix = np.zeros(shape=(len(self.doc_len), len(self.idf)))  # numdocs rows and num words cols
        doc_len = np.asarray(self.doc_len)
        for idx, word in enumerate(self.features_):
            word_freq = np.asarray([doc.get(word, 0) for doc in self.doc_freqs])
            word_vector = self.idf.get(word, 0) * (word_freq * (self.k1 + 1) /
                                                   (word_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
            matrix[:, idx] = word_vector
        return matrix

    def state_dict(self):
        return {
            'k1': self.k1,
            'b': self.b,
            'epsilon': self.epsilon,
            'avgdl': self.avgdl,
            'idf': self.idf,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        # ugly but effetive for now
        bm_index = cls()
        bm_index.__dict__.update(**state_dict)
        return bm_index

    def serialize(self, fout: Path | str, method: str = 'json', **kwargs):
        try:
            _serializers_registry[method](self.state_dict(), fout, **kwargs)
        except KeyError:
            raise

    @classmethod
    def deserialize(cls, fin, method: str = 'json'):
        loaded_state_dict = _deserializers_registry[method](fin)
        return cls.from_state_dict(loaded_state_dict)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k1={self.k1}, b={self.b}, epsilon={self.epsilon})"


def _serialize_json(obj, fout: Path | str, **kwargs):
    import json
    with open(fout, 'w') as f:
        json.dump(obj, f, **kwargs)


def _deserialize_json(fin: Path | str):
    import json
    with open(fin) as f:
        return json.load(f)


_serializers_registry = {'json': _serialize_json, }
_deserializers_registry = {'json': _deserialize_json, }


def register_serializer(method, serialize_fn):
    _serializers_registry[method] = serialize_fn


def register_deserializer(method, deserialize_fn):
    _deserializers_registry[method] = deserialize_fn
