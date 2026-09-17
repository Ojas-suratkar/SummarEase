"""Lets background threads (the watchlist poller, the async job runner,
auto-link-on-save) use `db.session` the same way request-handling code
does. Flask-SQLAlchemy's session is bound to `current_app`, which only
exists inside a request or an explicit application context -- a plain
`threading.Thread` has neither by default, so without this, any
database access from a background thread raises "working outside of
application context"."""
from __future__ import annotations

from typing import Callable

_app = None


def bind(app) -> None:
    global _app
    _app = app


def run(fn: Callable, *args, **kwargs):
    """Call `fn(*args, **kwargs)` inside the app's context. Falls back to
    calling it directly if `bind()` was never called (e.g. a unit test
    that exercises a service function in isolation) -- callers that need
    real database access still need the DB set up some other way in
    that case, but this at least doesn't crash on the context lookup
    itself."""
    if _app is None:
        return fn(*args, **kwargs)
    with _app.app_context():
        return fn(*args, **kwargs)
