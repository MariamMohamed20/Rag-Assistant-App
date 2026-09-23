"""
App settings, loaded from environment variables / .env file.
Never hard-code these values elsewhere in the codebase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where the persisted Chroma vector store lives (produced by the notebook, Phase 2.7)
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHROMA_COLLECTION_NAME: str = "lecture_notes"

    # Must match the model used to build the embeddings in the notebook
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Local Ollama LLM
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_HOST: str = "http://localhost:11434"

    # Retrieval
    TOP_K: int = 4

    # CORS - the frontend's origin (Streamlit default port)
    CORS_ORIGINS: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
