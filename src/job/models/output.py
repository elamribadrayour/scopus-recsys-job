"""
This module contains the models used in the project.
"""

from pydantic import BaseModel


class Output(BaseModel):
    datasets: list[str]
    algorithms: list[str]
    application: str  # Primary application field (e.g., cancer, medical, ecommerce)
