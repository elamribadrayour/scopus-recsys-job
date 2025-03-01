"""
This module contains the helper functions for the project.
"""

import numpy
from sentence_transformers import SentenceTransformer


def get_embeddings(data: list[str], embedding_model: str) -> numpy.ndarray:
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        convert_to_numpy=True,
        show_progress_bar=False,
        sentences=data,
    )
    return embeddings
