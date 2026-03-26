from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    xtts_model_path: str = ""
    xtts_device: str = "cuda"
    xtts_use_deepspeed: bool = True

    voices_dir: str = "./voices"
    default_voice: str = "default"
    default_language: str = "en"

    ws_host: str = "0.0.0.0"
    ws_port: int = 8766

    audio_device_index: int = -1


settings = Settings()
