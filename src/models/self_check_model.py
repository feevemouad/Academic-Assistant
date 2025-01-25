from langchain_core.pydantic_v1 import BaseModel, Field

class SelfCheckResult(BaseModel):
    block: bool = Field(
        description="Whether to block the message (True) or let it pass (False)"
    )
