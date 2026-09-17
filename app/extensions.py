"""Shared Flask extension instances, created here (not in __init__.py) so
that service/model modules can import `db` without a circular import back
to the app factory. Standard Flask-SQLAlchemy/Flask-Login pattern."""
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "info"
