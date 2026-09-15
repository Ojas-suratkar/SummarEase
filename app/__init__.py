from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

# Load .env before anything reads os.environ (config, services).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .config import Config  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.validate()

    # Make sure the folder generated audio files get written to exists.
    (Path(app.static_folder) / "audio").mkdir(parents=True, exist_ok=True)

    from .routes import bp as routes_bp

    app.register_blueprint(routes_bp)

    return app
