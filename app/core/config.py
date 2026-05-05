from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ML Serving Platform"
    VERSION: str = "1.0.0"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "ml-models-bucket"
    MODEL_KEY: str = "models/classifier.pt"
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()