import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "PipeGenie"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "pipegenie-super-secret-key-change-in-prod"

    # MongoDB
    MONGODB_URL: str #= "mongodb://localhost:27017"
    MONGODB_DB: str = "pipegenie"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL: int = 3600  # 1 hour cache

    # GitHub
    GITHUB_TOKEN: str = ""
    GITHUB_WEBHOOK_SECRET: str = "pipegenie-webhook-secret"
    REPO_WRITEBACK_ENABLED: bool = True
    AUTO_OPEN_PR: bool = True
    PIPEGENIE_BOT_NAME: str = "PipeGenie Bot"
    PIPEGENIE_BOT_EMAIL: str = "pipegenie-bot@users.noreply.github.com"

    # AI / Mistral
    MISTRAL_API_KEY: str = ""          # For Mistral API
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    USE_OLLAMA: bool = True            # True = local Ollama, False = Mistral API
    LLM_MODEL: str = "mistral"         # Ollama model name

    # MilvusDB
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Docker
    DOCKER_NETWORK: str = "pipegenie-net"

    # Risk thresholds
    RISK_LOW_THRESHOLD: float = 0.3
    RISK_HIGH_THRESHOLD: float = 0.7

    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
