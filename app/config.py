from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_path: str = "./data/chroma"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
