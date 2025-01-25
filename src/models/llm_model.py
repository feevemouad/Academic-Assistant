from pydantic import BaseModel, Field
from typing import Optional

class LLMModelConfig(BaseModel):
    provider: str = Field(..., description="The provider of the LLM (e.g., 'ollama', 'openai', 'huggingface', 'groq', 'together').")
    model_name: Optional[str] = Field(None, description="The name of the model to use (e.g., 'llama3.1', 'gpt-4'). Required for all providers except Hugging Face.")
    repo_id: Optional[str] = Field(None, description="The repository ID for Hugging Face models (e.g., 'mistralai/Mistral-7B-v0.1'). Required only for Hugging Face.")
    api_key: Optional[str] = Field(None, description="The API key for the provider (if required).")
    endpoint: Optional[str] = Field(None, description="The endpoint URL for the provider (if required).")
    
    # Add validation to ensure either `model_name` or `repo_id` is provided based on the provider
    def __init__(self, **data):
        super().__init__(**data)
        if self.provider == "huggingface":
            if not self.repo_id:
                raise ValueError("`repo_id` is required for Hugging Face models.")
        else:
            if not self.model_name:
                raise ValueError("`model_name` is required for non-Hugging Face providers.")
