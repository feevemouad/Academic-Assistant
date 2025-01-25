from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class Classification(BaseModel):
    category: Literal["InTopic", "OutOfTopic", "Interaction"] = Field(
        default="OutOfTopic",
        description="The classification category of the input"
    )

