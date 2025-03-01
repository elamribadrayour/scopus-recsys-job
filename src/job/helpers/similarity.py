"""
Similarity functions
"""

import numpy
import pandas
from loguru import logger
from sentence_transformers import SentenceTransformer, util


def get_embeddings(data: list[str], embedding_model: str) -> numpy.ndarray:
    model = SentenceTransformer(embedding_model)
    return model.encode(
        convert_to_numpy=True,
        show_progress_bar=False,
        sentences=data,
    )


def get_clusters(embeddings: numpy.ndarray, data: list[str]) -> pandas.DataFrame:
    clusters = util.community_detection(
        embeddings=embeddings, min_community_size=1, threshold=0.75
    )

    output = list()
    for i, cluster in enumerate(clusters):
        name = data[cluster[0]]
        if len(cluster) > 10:
            logger.info(f"cluster {i} -- name: {name} -- size: {len(cluster)}")
        values = [data[j] for j in cluster]
        output.append(
            {
                "group_id": str(i),
                "group_name": name,
                "values": values,
            }
        )
    df = pandas.DataFrame(data=output)
    df = df.explode(column="values")
    df.rename(columns={"values": "value"}, inplace=True)
    return df


def get_similarity(data: list[str], embedding_model: str) -> pandas.DataFrame:
    logger.info(f"computing embeddings for {len(data)} items")
    embeddings = get_embeddings(data=data, embedding_model=embedding_model)

    logger.info(f"computing clusters for {len(embeddings)} items")
    clusters = get_clusters(embeddings=embeddings, data=data)
    return clusters
