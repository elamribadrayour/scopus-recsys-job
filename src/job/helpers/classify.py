"""
This module contains the functions to predict the datasets, algorithms and application of a given abstract.
"""

import pandas
from openai import OpenAI

from models.output import Output


def get_prompt(text: str) -> str:
    return f"""
    Analyze the following abstract and extract the following details ONLY if the abstract is related to recommendation systems:

    1. datasets: List any datasets referenced or used.
    2. algorithms: List any algorithms or methods mentioned (e.g., collaborative filtering, matrix decomposition, etc.).
    3. application: Extract the primary practical application context in which the recommendation system is applied. This should reflect the real-world domain (for example, "medical", "cancer diagnosis", "ecommerce", "entertainment", "social media", or "financial services"). If multiple application domains are present, select the one that is most prominently featured or central to the study.

    If the abstract is not related to recommendation systems, please return an empty JSON object (i.e {{}}).

    Please provide the output in JSON format with the keys: datasets, algorithms, application.

    Abstract:
    {text}
    """


def get_classification(prompt: str, openai_api_key: str) -> dict | None:
    client = OpenAI(api_key=openai_api_key)
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=Output,
    )
    output = response.choices[0].message.parsed
    return output.model_dump() if output else None


def get_classifications(
    data: pandas.DataFrame, openai_api_key: str
) -> pandas.DataFrame:
    data = data.copy()
    data["prompt"] = data["Abstract"].apply(lambda x: get_prompt(text=x))
    data["prediction"] = data["prompt"].apply(
        lambda x: get_classification(prompt=x, openai_api_key=openai_api_key)
    )
    data = data[data["prediction"].notna()]
    data["datasets"] = data["prediction"].apply(
        lambda x: [str(y).lower() for y in x["datasets"]]
    )
    data["algorithms"] = data["prediction"].apply(
        lambda x: [str(y).lower() for y in x["algorithms"]]
    )
    data["application"] = data["prediction"].apply(
        lambda x: str(x["application"]).lower()
    )
    return data[["DOI", "datasets", "algorithms", "application"]]
