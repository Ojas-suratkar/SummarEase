import os


def _normalize_database_url(url: str) -> str:
    """Render/Heroku-style hosts hand back `postgres://`, but SQLAlchemy's
    psycopg2 dialect wants `postgresql://` -- normalize so DATABASE_URL
    can be pasted in verbatim from whichever host you end up on."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB upload cap -- PDFs are small, but audio/video need the headroom
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    # DATABASE_URL is the de-facto standard env var name every major host
    # (Render, Railway, Heroku, Fly's Postgres add-on) injects automatically
    # once you attach a Postgres database -- set it and the app is on
    # Postgres with zero code changes; leave it unset for local dev and it
    # falls back to a single SQLite file under instance/.
    _database_url = os.environ.get("DATABASE_URL", "").strip()
    if _database_url:
        SQLALCHEMY_DATABASE_URI = _normalize_database_url(_database_url)
    else:
        _instance_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance")
        SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(_instance_dir, "app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # Cookie hardening -- meaningful once this is actually reachable over
    # the network instead of just localhost. SESSION_COOKIE_SECURE is left
    # off unless PREFERRED_URL_SCHEME/behind-HTTPS is confirmed, since a
    # `Secure` cookie silently refuses to be set at all over plain HTTP
    # (which local dev and an unconfigured host both still are).
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("FORCE_HTTPS", "0") == "1"

    # An upper bound on how long a signed session cookie is accepted at
    # all. The finer-grained idle timeout and server-side revocation
    # live in services/session_guard.py; this is the backstop for a
    # cookie that somehow outlives its row -- a session that is never
    # allowed to expire isn't a convenience, it's an unlocked door on
    # whatever computer it was left on.
    from datetime import timedelta

    PERMANENT_SESSION_LIFETIME = timedelta(days=90)
    SESSION_REFRESH_EACH_REQUEST = False

    @staticmethod
    def validate():
        if not Config.SECRET_KEY:
            raise RuntimeError(
                "FLASK_SECRET_KEY is not set. Copy .env.example to .env and "
                "set a random secret key before running the app."
            )
