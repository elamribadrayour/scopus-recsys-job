"""
This module contains the functions to download the data from the API.
"""

import duckdb
import pandas


def get_all_papers() -> pandas.DataFrame:
    data = pandas.read_excel("RS_Scopus.xlsx")
    data["DOI"] = data["DOI"].str.strip().replace('""', "")
    data = data[data["DOI"] != ""]
    data = data[data["DOI"].notna()]
    data = data[["DOI", "Title", "Abstract"]]
    data.drop_duplicates(subset=["DOI"], inplace=True)
    return data


def get_data_to_classify(db: duckdb.DuckDBPyConnection) -> pandas.DataFrame:
    query = """
    WITH classified_dois AS (
        SELECT doi FROM classification
    )
    SELECT p.*
    FROM paper p
    LIMIT 10
    """
    return db.execute(query=query).fetch_df()


def get_algorithms_to_embed(db: duckdb.DuckDBPyConnection) -> list[str]:
    query = """
    SELECT DISTINCT UNNEST(algorithms) AS algorithm
    FROM classification
    """
    algorithms = db.execute(query=query).fetchall()
    return list(set([row[0] for row in algorithms]))


def get_applications_to_embed(db: duckdb.DuckDBPyConnection) -> list[str]:
    query = """
    SELECT DISTINCT application
    FROM classification
    """
    output = db.execute(query=query).fetchall()
    return list(set([row[0] for row in output]))
