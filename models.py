"""הגדרת המודלים של OpenAI במקום אחד."""
import os

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

import config


def setup_models() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("חסר OPENAI_API_KEY בקובץ .env")
    Settings.embed_model = OpenAIEmbedding(model=config.EMBED_MODEL)
    Settings.llm = OpenAI(model=config.LLM_MODEL, temperature=0)
