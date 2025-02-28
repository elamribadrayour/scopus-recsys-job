"""
This script is used to predict the datasets, algorithms and application of a given abstract.
"""

from typing import Annotated

import numpy
from tqdm import tqdm
from loguru import logger
from typer import Typer, Argument

import helpers.db
import helpers.embed
import helpers.download
import helpers.classify

app = Typer(name="scopus-recsys")


@app.command(name="init")
def init(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
):
    logger.info(f"initializing database at {data_path}")
    conn = helpers.db.get_db(data_path)
    data = helpers.download.get_all_papers()
    logger.info(f"number of papers to process: {len(data)}")

    helpers.db.set_table(conn=conn, table_name="paper")
    helpers.db.set_data(conn=conn, table_name="paper", data=data)
    logger.info("database initialized")


@app.command(name="classify")
def classify(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
    openai_api_key: Annotated[str, Argument(envvar="OPENAI_API_KEY")],
    batch_size: Annotated[int, Argument(envvar="BATCH_SIZE")] = 10,
):
    conn = helpers.db.get_db(data_path=data_path)
    helpers.db.set_table(conn=conn, table_name="classification")
    data = helpers.download.get_data_to_classify(db=conn)
    logger.info(f"number of papers to classify: {len(data)}")

    if len(data) == 0:
        logger.info("no data to classify")
        return

    batches = numpy.array_split(data, len(data) // batch_size)
    for batch in tqdm(batches, total=len(batches)):
        data = helpers.classify.get_classifications(
            data=batch,  # type: ignore
            openai_api_key=openai_api_key,
        )
        helpers.db.set_data(conn=conn, table_name="classification", data=data)


@app.command(name="embed-algorithm")
def embed_algorithm(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
    embedding_model: Annotated[str, Argument(envvar="EMBEDDING_MODEL")],
):
    conn = helpers.db.get_db(data_path)
    model_dimensions = helpers.embed.get_model_dimensions(
        embedding_model=embedding_model
    )
    helpers.db.set_table(
        conn=conn,
        table_name="algorithm-embedding",
        args={"model_dimension": model_dimensions},
    )
    data = helpers.download.get_algorithms_to_embed(db=conn)
    logger.info(f"number of algorithms to embed: {len(data)}")

    if len(data) == 0:
        logger.info("no algorithms to embed")
        return

    data = helpers.embed.get_embeddings(
        data=data, embedding_model=embedding_model, column="algorithm"
    )
    helpers.db.set_data(conn=conn, table_name="algorithm-embedding", data=data)
    helpers.db.optimize_index(conn=conn, table_name="algorithm-embedding")
    logger.info("algorithms embedded")


@app.command(name="embed-application")
def embed_application(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
    embedding_model: Annotated[str, Argument(envvar="EMBEDDING_MODEL")],
):
    conn = helpers.db.get_db(data_path)
    model_dimensions = helpers.embed.get_model_dimensions(
        embedding_model=embedding_model
    )
    helpers.db.set_table(
        conn=conn,
        table_name="application-embedding",
        args={"model_dimension": model_dimensions},
    )
    data = helpers.download.get_applications_to_embed(db=conn)
    logger.info(f"number of applications to embed: {len(data)}")

    if len(data) == 0:
        logger.info("no applications to embed")
        return

    data = helpers.embed.get_embeddings(
        data=data, embedding_model=embedding_model, column="application"
    )
    helpers.db.set_data(conn=conn, table_name="application-embedding", data=data)
    helpers.db.optimize_index(conn=conn, table_name="application-embedding")
    logger.info("applications embedded")


if __name__ == "__main__":
    app()
