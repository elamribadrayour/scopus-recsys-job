"""
This module contains the helper functions for the database.
"""

import os

import duckdb
import pandas
from loguru import logger


def get_db(data_path: str) -> duckdb.DuckDBPyConnection:
    db_path = os.path.join(data_path, "db.duckdb")
    conn = duckdb.connect(database=db_path)
    conn.execute(open("sql/init/db.sql").read())
    return conn


def set_table(conn: duckdb.DuckDBPyConnection, table_name: str, args: dict = dict()):
    query = open(f"sql/init/{table_name}.sql").read().format(**args)
    conn.execute(query)


def set_data(conn: duckdb.DuckDBPyConnection, table_name: str, data: pandas.DataFrame):
    table_name = table_name.replace("-", "_")
    conn.register("temp_df", data)
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
    conn.unregister("temp_df")
    conn.commit()

    nb_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # type: ignore
    logger.info(f"set data: {table_name} size={nb_rows}")


def set_data_from_query(conn: duckdb.DuckDBPyConnection, table_name: str):
    query = open(f"sql/populate/{table_name}.sql").read()
    conn.execute(query=query)
    conn.commit()

    table_name = table_name.replace("-", "_")
    nb_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # type: ignore
    logger.info(f"set data: {table_name} size={nb_rows}")


def optimize_index(conn: duckdb.DuckDBPyConnection, table_name: str):
    query = open(f"sql/optimize/{table_name}.sql").read()
    conn.execute(query)
    logger.info(f"optimized index: {table_name}")
