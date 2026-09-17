"""Account creation, login/logout, and the profile page -- the "session
management, deletion, creation, profile accessible anywhere" piece.
Plain Flask-Login + werkzeug password hashing (no third-party auth
service): a signup writes a User row with a salted+hashed password
(never plaintext, never logged), login sets a signed session cookie via
Flask-Login, and every other route in the app requires that session
(see `@login_required` in routes.py) so all data is scoped to whoever is
logged in -- the actual meaning of "profile that can be accessed
anywhere": log in from any device once this is deployed, and it's your
data, not a shared local file."""
from __future__ import annotations

import re

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from .extensions import db
from .models import Annotation, Draft, Flashcard, HistoryEntry, User, Watch
from .services import session_guard

bp = Blueprint("auth", __name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("routes.index"))

    if request.method == "GET":
        return render_template("auth/signup.html")

    email = request.form.get("email", "").strip().lower()
    display_name = request.form.get("display_name", "").strip() or email.split("@")[0]
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")

    error = None
    if not _EMAIL_RE.match(email):
        error = "Enter a valid email address."
    elif len(password) < 8:
        error = "Use a password of at least 8 characters."
    elif password != confirm:
        error = "Passwords don't match."
    elif User.query.filter_by(email=email).first() is not None:
        error = "An account with that email already exists -- log in instead."

    if error:
        return render_template("auth/signup.html", error=error, email=email, display_name=display_name)

    user = User(email=email, display_name=display_name, api_token=User.new_api_token())
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=False)
    session_guard.start_session(user.id, remember=False)
    flash("Welcome to SummarEase.", "success")
    return redirect(url_for("routes.index"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("routes.index"))

    if request.method == "GET":
        return render_template("auth/login.html", next=request.args.get("next", ""))

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    next_url = request.form.get("next", "")

    remember = request.form.get("remember") == "1"

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        return render_template("auth/login.html", error="Incorrect email or password.", email=email, next=next_url)

    # `remember` used to be hardcoded True, which quietly gave every
    # login a year-long cookie on every machine. It's now the user's
    # decision, and either way the session is tracked server-side so it
    # can be listed and revoked. See services/session_guard.py.
    login_user(user, remember=remember)
    session_guard.start_session(user.id, remember=remember)

    if next_url and next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("routes.index"))


@bp.post("/logout")
@login_required
def logout():
    session_guard.end_current_session()
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "update_details":
            display_name = request.form.get("display_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            if not _EMAIL_RE.match(email):
                error = "Enter a valid email address."
            elif email != current_user.email and User.query.filter_by(email=email).first() is not None:
                error = "Another account already uses that email."
            else:
                current_user.display_name = display_name or current_user.display_name
                current_user.email = email
                db.session.commit()
                success = "Profile updated."

        elif action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")
            if not current_user.check_password(current_pw):
                error = "Current password is incorrect."
            elif len(new_pw) < 8:
                error = "New password must be at least 8 characters."
            elif new_pw != confirm_pw:
                error = "New passwords don't match."
            else:
                current_user.set_password(new_pw)
                db.session.commit()
                success = "Password changed."

        elif action == "regenerate_token":
            current_user.api_token = User.new_api_token()
            db.session.commit()
            success = "New save-from-anywhere link generated below -- update your bookmarklet."

    stats = {
        "entry_count": HistoryEntry.query.filter_by(user_id=current_user.id).count(),
        "watch_count": Watch.query.filter_by(user_id=current_user.id).count(),
        "flashcard_count": Flashcard.query.filter_by(user_id=current_user.id).count(),
        "annotation_count": Annotation.query.filter_by(user_id=current_user.id).count(),
        "draft_count": Draft.query.filter_by(user_id=current_user.id).count(),
    }

    return render_template(
        "auth/profile.html",
        error=error,
        success=success,
        stats=stats,
        bookmarklet_url=url_for("routes.bookmarklet_save", _external=True),
    )
