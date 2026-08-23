from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RedGuard AI"
    version: str = "0.3.0"

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    min_image_width: int = 32
    min_image_height: int = 32

    model_config = SettingsConfigDict(
        env_prefix="REDGUARD_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
