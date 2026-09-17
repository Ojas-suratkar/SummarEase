"""
Shared Gemini API client + resilience layer.

Every service that talks to Gemini -- summarization, embeddings, RAG
question-answering, claim extraction, synthesis, and the newer structured-
output features -- goes through this one module rather than each
maintaining its own retry/fallback logic. Uses Google's current
`google-genai` SDK (the `google-generativeai` package this app originally
used is deprecated and doesn't support the newer "AQ."-prefixed API keys
Google AI Studio now issues by default).

Resilience design
------------------
A single Gemini call can fail for reasons that fall into three very
different buckets, and treating them the same way is how naive retry
logic makes things worse, not better:

1. **Transient** (momentary overload, rate limit) -- worth retrying with
   backoff, which `call_with_retry` already did.
2. **This specific model is having a bad day** -- worth trying a
   *different* model, not just the same one again. `generate_content_resilient`
   adds a fallback chain across several real, current Gemini models
   (`FALLBACK_MODELS`) so a Flash-tier outage doesn't take the whole app
   down with it.
3. **Sustained outage** -- if every model in the chain is failing
   *repeatedly*, retrying the full ladder on every single request just
   makes every page slow to fail instead of failing fast. A small circuit
   breaker (`_CircuitBreaker`) trips after a run of consecutive full
   failures and short-circuits new calls for a cool-down window, raising
   `GeminiUnavailableError` immediately so callers (see summarizer.py's
   offline TextRank fallback) can degrade gracefully instead of hanging.

On top of that, a lightweight token-bucket rate limiter
(`_RateLimiter`) paces outgoing requests so bursts of concurrent calls
(chunked YouTube summarization, briefing/compare's parallel source
extraction, topic-cluster labeling) don't collectively trip Google's own
per-minute rate limit in the first place -- prevention, not just recovery.

None of this can eliminate Google's actual quota limits, which are
enforced server-side -- what it does is make sure hitting one doesn't
mean a broken page.
"""
from __future__ import annotations

import collections
import os
import random
import threading
import time

from google import genai

_lock = threading.Lock()
_client: genai.Client | None = None

# gemini-3.5-flash / gemini-3.5-flash-lite are the current Flash tiers --
# Google'''s API returned explicit 404s telling this app to move off both
# gemini-2.5-flash and gemini-2.5-flash-lite onto these, so they'''re
# confirmed live rather than guessed (the 2.5 tier is no longer available
# to new users at all, so it isn'''t kept as a fallback -- it would just
# fail every time and waste a retry).
MODEL_NAME = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]

# Matryoshka-style embedding model: a vector truncated to a shorter
# prefix (see embeddings.py) is still a valid, useful embedding of that
# length, which is what lets this app keep stored vectors small.
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_FALLBACK_MODELS = ["gemini-embedding-001", "gemini-embedding-2-preview"]


class GeminiNotConfiguredError(RuntimeError):
    """Raised when GEMINI_API_KEY isn't set."""


class GeminiUnavailableError(RuntimeError):
    """Raised when every model in the fallback chain failed (or the
    circuit breaker is open from a recent run of failures). Distinct from
    GeminiNotConfiguredError so callers can tell "not set up" apart from
    "set up correctly, but Google's having an outage" -- the latter is
    what should trigger an offline/degraded fallback, the former should
    surface the setup instructions."""


def get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiNotConfiguredError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your own key from https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=api_key)
        return _client


# ---------------------------------------------------------------------------
# Transient-error retry (unchanged behavior, still used standalone by
# anything that wants plain retry without the fallback-chain machinery).
# ---------------------------------------------------------------------------

_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500",
    "DEADLINE_EXCEEDED", "overloaded",
)


def _is_transient(exc: Exception) -> bool:
    return any(marker in str(exc) for marker in _TRANSIENT_MARKERS)


def call_with_retry(fn, *args, retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Call `fn(*args, **kwargs)`, retrying with exponential backoff (plus
    a little jitter) when the failure looks transient, and failing fast on
    anything that looks like a real error (bad key, bad request)."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- message-sniffed below, re-raised if not transient
            last_exc = exc
            if not _is_transient(exc) or attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.5))
    raise last_exc  # pragma: no cover -- unreachable; the loop always returns or raises


# ---------------------------------------------------------------------------
# Rate limiter -- a plain token bucket. Caps both how many Gemini calls run
# at once and how many run per rolling minute, so this app's own
# concurrency (ThreadPoolExecutor chunk summarization, parallel briefing
# extraction, topic labeling) can't burst past what the API tier allows.
# ---------------------------------------------------------------------------


class _RateLimiter:
    def __init__(self, max_concurrent: int = 4, max_per_minute: int = 50):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_per_minute = max_per_minute
        self._call_times: collections.deque = collections.deque()
        self._window_lock = threading.Lock()

    def _wait_for_window(self) -> None:
        while True:
            with self._window_lock:
                now = time.monotonic()
                while self._call_times and now - self._call_times[0] > 60:
                    self._call_times.popleft()
                if len(self._call_times) < self._max_per_minute:
                    self._call_times.append(now)
                    return
                sleep_for = 60 - (now - self._call_times[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))

    def acquire(self):
        self._wait_for_window()
        self._semaphore.acquire()

    def release(self):
        self._semaphore.release()


_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# Circuit breaker -- trips after a run of consecutive *full* failures
# (every model in the fallback chain, on the same request) and, while
# open, fails new requests immediately rather than making them sit through
# the whole retry ladder during a known outage.
# ---------------------------------------------------------------------------


class _CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self._cooldown_seconds:
                # Half-open: let the next call through as a probe.
                self._opened_at = None
                self._consecutive_failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold and self._opened_at is None:
                self._opened_at = time.monotonic()


_circuit_breaker = _CircuitBreaker()


# ---------------------------------------------------------------------------
# The resilient call itself -- rate limiting + circuit breaker + a
# retried, multi-model fallback chain, all in one place so every Gemini
# text-generation call site in this app can share it.
# ---------------------------------------------------------------------------


def generate_content_resilient(contents, config=None, models: list[str] | None = None, retries_per_model: int = 2):
    """The resilient replacement for a bare
    `client.models.generate_content(model=..., contents=..., config=...)`
    call. Tries each model in `models` (default: FALLBACK_MODELS) in
    order, retrying transient failures within each model before moving
    to the next. Raises GeminiNotConfiguredError immediately if there's
    no API key (retrying that is pointless), or GeminiUnavailableError if
    every model failed or the circuit breaker is currently open."""
    client = get_client()  # raises GeminiNotConfiguredError early, before touching the breaker

    if _circuit_breaker.is_open():
        raise GeminiUnavailableError(
            "Gemini has been failing repeatedly in the last few seconds -- "
            "backing off briefly instead of retrying immediately."
        )

    models = models or FALLBACK_MODELS
    last_exc: Exception | None = None

    for model in models:
        _rate_limiter.acquire()
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            response = call_with_retry(client.models.generate_content, retries=retries_per_model, **kwargs)
            _circuit_breaker.record_success()
            return response
        except Exception as exc:  # noqa: BLE001 -- try the next model in the chain
            last_exc = exc
        finally:
            _rate_limiter.release()

    _circuit_breaker.record_failure()
    raise GeminiUnavailableError(
        f"Gemini is unavailable right now (tried {len(models)} model(s)): {last_exc}"
    ) from last_exc


def get_health_status() -> dict:
    """Snapshot of the resilience layer's current state, for the ops
    dashboard (routes.py's /dashboard) -- whether the circuit breaker is
    open, how many calls have gone out in the current rate-limit window,
    and whether an API key is configured at all."""
    configured = bool(os.environ.get("GEMINI_API_KEY"))
    with _circuit_breaker._lock:
        breaker_open = _circuit_breaker._opened_at is not None
        consecutive_failures = _circuit_breaker._consecutive_failures
    with _rate_limiter._window_lock:
        calls_in_window = len(_rate_limiter._call_times)
    return {
        "configured": configured,
        "circuit_breaker_open": breaker_open,
        "consecutive_failures": consecutive_failures,
        "calls_last_minute": calls_in_window,
        "max_calls_per_minute": _rate_limiter._max_per_minute,
        "fallback_models": FALLBACK_MODELS,
    }


def embed_content_resilient(contents, config=None, models: list[str] | None = None):
    """Same idea as generate_content_resilient, for the embeddings
    endpoint -- tries gemini-embedding-001, then a fallback embedding
    model, before giving up."""
    client = get_client()
    models = models or EMBEDDING_FALLBACK_MODELS
    last_exc: Exception | None = None

    for model in models:
        _rate_limiter.acquire()
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            result = call_with_retry(client.models.embed_content, retries=2, **kwargs)
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        finally:
            _rate_limiter.release()

    raise GeminiUnavailableError(f"Embeddings are unavailable right now: {last_exc}") from last_exc
