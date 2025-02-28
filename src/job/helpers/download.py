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
    LEFT JOIN classified_dois cd ON p.doi = cd.doi
    WHERE cd.doi IS NULL
    """
    return db.execute(query=query).fetch_df()


def get_algorithms_to_embed(db: duckdb.DuckDBPyConnection) -> pandas.DataFrame:
    query = """
    SELECT algorithms
    FROM classification
    """
    algorithms = db.execute(query=query).fetchall()
    algorithms = [row[0] for row in algorithms]
    algorithms = [item for sublist in algorithms for item in sublist]
    algorithms = list(set(algorithms))

    query = """
    SELECT algorithm
    FROM algorithm_embedding
    """
    algorithms_embedded = db.execute(query=query).fetchall()
    algorithms_embedded = [row[0] for row in algorithms_embedded]
    algorithms_embedded = list(set(algorithms_embedded))

    algorithms = [
        algorithm for algorithm in algorithms if algorithm not in algorithms_embedded
    ]
    return pandas.DataFrame(list(set(algorithms)), columns=["algorithm"])


def get_applications_to_embed(db: duckdb.DuckDBPyConnection) -> pandas.DataFrame:
    query = """
    WITH application_processed AS (
        SELECT application
        FROM application_embedding
    ),

    all_applications AS (
        SELECT DISTINCT application
        FROM classification
    )

    SELECT
        a.application
    FROM all_applications a
    LEFT JOIN application_processed ap ON a.application = ap.application
    WHERE ap.application IS NULL
    """
    output = db.execute(query=query).fetchall()
    return pandas.DataFrame(
        list(set([row[0] for row in output])), columns=["application"]
    )
