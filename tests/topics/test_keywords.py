import numpy as np

from arxiv_atlas.topics.keywords import topic_keywords

TEXTS = [
    "graph neural networks for molecules",
    "message passing graph neural networks",
    "graph neural networks on citation graphs",
    "diffusion models for image generation",
    "denoising diffusion models for images",
    "latent diffusion models for image editing",
    "we propose a novel method",  # no topic
]
LABELS = np.array([0, 0, 0, 1, 1, 1, -1])


def test_keywords_set_topics_apart():
    keywords = topic_keywords(TEXTS, LABELS, top_n=3, min_df=1)
    assert set(keywords) == {0, 1}
    assert "graph" in keywords[0]
    assert "diffusion" in keywords[1]
    assert not {"diffusion", "image"} & set(keywords[0])


def test_papers_without_topic_and_boilerplate_are_ignored():
    keywords = topic_keywords(TEXTS, LABELS, top_n=20, min_df=1)
    words = {w for ws in keywords.values() for w in ws}
    assert not {"propose", "novel", "method"} & words


def test_latex_is_dropped():
    texts = [r"bounds in $\mathbb{R}^n$ with \emph{sharp} rates"] * 2
    keywords = topic_keywords(texts, np.array([0, 0]), min_df=1)
    assert not {"mathbb", "emph"} & set(keywords[0])
    assert "sharp" in keywords[0]


def test_no_topics_means_no_keywords():
    assert topic_keywords(["a b c"], np.array([-1])) == {}
