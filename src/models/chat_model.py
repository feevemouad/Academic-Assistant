from pydantic import BaseModel
from .llm_model import LLMModelConfig
from typing import List, Tuple

class ChatRequest(BaseModel):
    user_input: str
    conversation_history: List[Tuple[str, str]]
    llm_model : LLMModelConfig # {"provider": "ollama", "model_name": "llama3.1")}
