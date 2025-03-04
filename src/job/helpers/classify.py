"""
This module contains the functions to predict the datasets, algorithms and application of a given abstract.
"""

import pandas
import ollama
from ollama import chat
from loguru import logger


from models.output import Output


def init(model: str, ollama_host: str) -> None:
    ollama.pull(model=model)


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


def get_classification(prompt: str, model: str, ollama_host: str) -> dict | None:
    response = chat(
        model=model,
        format=Output.model_json_schema(),
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    output = None
    try:
        output = Output.model_validate_json(response.message.content)
    except Exception as e:
        logger.error(f"error validating JSON: {e}")
        return None
    return output.model_dump() if output else None


def get_classifications(
    data: pandas.DataFrame, model: str, ollama_host: str
) -> pandas.DataFrame:
    data = data.copy()
    data["prompt"] = data["abstract"].apply(lambda x: get_prompt(text=x))
    data["prediction"] = data["prompt"].apply(
        lambda x: get_classification(prompt=x, model=model, ollama_host=ollama_host)
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
    return data[["doi", "datasets", "algorithms", "application"]]
