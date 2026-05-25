from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "ML Serving Platform"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # AWS — only required for /load-model (S3 checkpoint download)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "ml-models-bucket"
    MODEL_KEY: str = "models/classifier.pt"


settings = Settings()