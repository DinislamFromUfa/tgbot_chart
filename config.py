from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: SecretStr
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


config = Settings()