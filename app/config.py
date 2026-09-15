import os


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload cap (PDFs)
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    @staticmethod
    def validate():
        if not Config.SECRET_KEY:
            raise RuntimeError(
                "FLASK_SECRET_KEY is not set. Copy .env.example to .env and "
                "set a random secret key before running the app."
            )
