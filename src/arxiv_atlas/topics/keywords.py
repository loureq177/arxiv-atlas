"""Describe each topic by the words that set it apart from the other topics (c-TF-IDF).

Plain TF-IDF scores words per document. Here every topic is treated as one long
document made of all its papers, so a word scores high when it is frequent in this
topic and rare across the rest.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import normalize

# Words found in almost every abstract. c-TF-IDF already down-weights them; listing them
# keeps them out of the few topics where they would still win. "model" is not here:
# "language model" and "diffusion model" are exactly the phrases a topic should show.
_ABSTRACT_WORDS = frozenset(
    [
        "paper",
        "propose",
        "proposed",
        "approach",
        "method",
        "methods",
        "results",
        "based",
        "using",
        "show",
        "novel",
    ]
)

# Inline maths ($O(n \log n)$) and LaTeX commands (\emph) would surface as "mathbb" etc.
_LATEX = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+")


def topic_keywords(
    texts: Sequence[str], labels: NDArray[np.int_], top_n: int = 10, min_df: int = 5
) -> dict[int, list[str]]:
    """The `top_n` most distinctive words and word pairs of every topic; -1 is ignored."""
    clustered = labels >= 0
    topic_ids = np.unique(labels[clustered])
    if topic_ids.size == 0:
        return {}
    vectorizer = CountVectorizer(
        stop_words=sorted(_ABSTRACT_WORDS | ENGLISH_STOP_WORDS),
        preprocessor=_without_latex,
        ngram_range=(1, 2),
        min_df=min(min_df, int(clustered.sum())),
    )
    counts = vectorizer.fit_transform([t for t, c in zip(texts, clustered, strict=True) if c])

    # Row i of `membership` selects the papers of topic_ids[i]; the product sums their counts.
    rows = np.searchsorted(topic_ids, labels[clustered])
    membership = sparse.csr_matrix(
        (np.ones(rows.size), (rows, np.arange(rows.size))), shape=(topic_ids.size, rows.size)
    )
    per_topic = sparse.csr_matrix(membership @ counts)
    term_frequency = normalize(per_topic, norm="l1", axis=1)
    term_totals = np.asarray(per_topic.sum(axis=0)).ravel()
    avg_words_per_topic = per_topic.sum() / topic_ids.size
    idf = np.log(1 + avg_words_per_topic / term_totals)
    scores = sparse.csr_matrix(term_frequency.multiply(idf)).toarray()

    vocabulary = vectorizer.get_feature_names_out()
    best = np.argsort(-scores, axis=1)[:, :top_n]
    return {
        int(topic): [str(vocabulary[j]) for j in best[i] if scores[i, j] > 0]
        for i, topic in enumerate(topic_ids)
    }


def _without_latex(text: str) -> str:
    return _LATEX.sub(" ", text).lower()
