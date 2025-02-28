"""
This module contains the helper functions for the project.
"""

import pandas
from sentence_transformers import SentenceTransformer


def get_model_dimensions(embedding_model: str) -> int:
    model = SentenceTransformer(embedding_model)
    return int(model.get_sentence_embedding_dimension())  # type: ignore


def get_embeddings(
    data: pandas.DataFrame, embedding_model: str, column: str
) -> pandas.DataFrame:
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        convert_to_numpy=True,
        show_progress_bar=False,
        sentences=data[column].tolist(),
    )
    data["embedding"] = [embedding.tolist() for embedding in embeddings]
    return data[[column, "embedding"]]
