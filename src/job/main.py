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
import helpers.similarity

app = Typer(name="scopus-recsys")


@app.command(name="init")
def init(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
):
    logger.info("initializing database")
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
    logger.info("classifying papers")
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


@app.command(name="similarity-algorithm")
def similarity_algorithm(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
    embedding_model: Annotated[str, Argument(envvar="EMBEDDING_MODEL")],
):
    logger.info("calculating algorithm similarity")
    conn = helpers.db.get_db(data_path)
    helpers.db.set_table(
        conn=conn,
        table_name="algorithm-similarity",
    )

    data = helpers.download.get_algorithms_to_embed(db=conn)
    logger.info(f"number of algorithms to embed: {len(data)}")

    if len(data) == 0:
        logger.info("no algorithms to embed")
        return

    output = helpers.similarity.get_similarity(
        data=data, embedding_model=embedding_model
    )
    helpers.db.set_data(conn=conn, table_name="algorithm-similarity", data=output)
    helpers.db.optimize_index(conn=conn, table_name="algorithm-similarity")


@app.command(name="similarity-application")
def similarity_application(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
    embedding_model: Annotated[str, Argument(envvar="EMBEDDING_MODEL")],
):
    logger.info("calculating application similarity")
    conn = helpers.db.get_db(data_path)
    helpers.db.set_table(
        conn=conn,
        table_name="application-similarity",
    )

    data = helpers.download.get_applications_to_embed(db=conn)
    logger.info(f"number of applications to embed: {len(data)}")

    if len(data) == 0:
        logger.info("no applications to embed")
        return

    output = helpers.similarity.get_similarity(
        data=data, embedding_model=embedding_model
    )
    helpers.db.set_data(conn=conn, table_name="application-similarity", data=output)
    helpers.db.optimize_index(conn=conn, table_name="application-similarity")


@app.command(name="algorithm-application-link")
def algorithm_application_link(
    data_path: Annotated[str, Argument(envvar="DATA_PATH")],
):
    logger.info("calculating algorithm applications links")
    conn = helpers.db.get_db(data_path)
    helpers.db.set_table(
        conn=conn,
        table_name="algorithm-application-link",
    )

    helpers.db.set_data_from_query(
        conn=conn,
        table_name="algorithm-application-link",
    )


if __name__ == "__main__":
    app()
