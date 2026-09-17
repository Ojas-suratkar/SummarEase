import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

# Load .env before anything reads os.environ (config, services).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .config import Config  # noqa: E402
from .extensions import db, login_manager  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.validate()

    # Make sure the folder generated audio files get written to exists.
    (Path(app.static_folder) / "audio").mkdir(parents=True, exist_ok=True)
    # Make sure the instance folder (default SQLite location) exists.
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def _load_user(user_id: str):
        return User.query.get(int(user_id))

    from .auth import bp as auth_bp
    from .routes import bp as routes_bp
    from .routes_v2 import bp as v2_bp
    from .routes_main import bp as main_bp
    from .routes_v3 import bp as matters_bp

    app.register_blueprint(routes_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(v2_bp)
    app.register_blueprint(matters_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()
        # create_all() won't add columns to tables that already exist, and
        # there is real data in this database. See app/migrations.py.
        from .migrations import run_migrations

        applied = run_migrations()
        if applied:
            logging.getLogger(__name__).info("Applied %d schema migration(s)", len(applied))

    _register_session_guard(app)

    from .services import app_context

    app_context.bind(app)

    # Start the autonomous watchlist's background polling thread. Under
    # the debug reloader, Flask re-execs the whole process behind
    # WERKZEUG_RUN_MAIN so create_app() effectively runs in two separate
    # processes; only start the thread in the one that's actually serving
    # requests (or when the reloader isn't in play at all).
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from .services import watchlist

        watchlist.start_scheduler()

    _register_error_handlers(app)

    return app


def _register_session_guard(app: Flask) -> None:
    """Check session validity on every request.

    This runs before the view, so an expired or revoked session can't
    reach a handler and read data it no longer has the right to. Static
    files and the session heartbeat are exempt: the heartbeat exists to
    report remaining time, and letting it slide the window forward would
    mean an open tab never idled out at all -- the exact bug this is
    meant to fix.
    """
    from .services import session_guard

    exempt = ("/static/", "/api/session/heartbeat", "/login", "/signup", "/save")

    @app.before_request
    def _enforce_session():
        try:
            expired = session_guard.enforce(exempt_paths=exempt)
        except Exception:  # pragma: no cover
            # Never let session bookkeeping take the whole app down.
            logging.getLogger(__name__).exception("Session guard failed")
            return None
        if expired:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Your session timed out. Log in again."}), 401
            return redirect(url_for("auth.login", next=request.path, timeout="1"))
        return None

    @app.context_processor
    def _inject_session_info():
        """Makes the idle countdown available to every template without
        each route having to remember to pass it."""
        try:
            return {"session_seconds_remaining": session_guard.seconds_remaining()}
        except Exception:
            return {"session_seconds_remaining": None}


def _register_error_handlers(app: Flask) -> None:
    """Catch anything that slips past a route's own error handling so the
    user always sees a plain, friendly message instead of a raw traceback
    or a blank connection-reset page -- 'the app should never just fail'
    applies here too, not only to Gemini calls."""
    logger = logging.getLogger(__name__)

    @app.errorhandler(Exception)
    def _handle_unexpected_error(exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.path)
        if request.path.startswith("/api/"):
            return jsonify({"error": "Something went wrong on this end. Please try again."}), 500
        return render_template("error.html", message=str(exc)), 500

    @app.errorhandler(404)
    def _handle_not_found(exc):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found."}), 404
        return render_template("error.html", message="That page doesn't exist."), 404
