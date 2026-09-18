# SummarEase

A workspace for source material you intend to keep. Documents, recordings,
video, web pages and images go in; summaries come out; the originals stay,
playable and searchable, and the collection can be read across as a whole.

Built with Flask, SQLAlchemy and vanilla JavaScript. No build step.

---

## What this is, and what it isn't

The summarising is done by a language model, and a chat assistant does that
job perfectly well. That is not the reason this exists and it is not presented
as one. For a one-off summary, use a chat assistant — it is quicker and free.

What this does instead is keep the material. The original file is stored and
replayable rather than discarded once its text has been extracted, so a
recording made months ago can still be listened to. Everything saved stays
searchable, and the analysis runs across the whole collection rather than over
whatever was last pasted into a prompt.

A substantial part of the application needs no AI service at all. Those parts
are algorithms operating on stored text: fast, free, deterministic, and
working when an API key is absent or an upstream service is down.

---

## The algorithms

`app/core/` contains the work this project exists to demonstrate. Each module
implements a published algorithm directly, depends on nothing beyond the
standard library and NumPy, and carries its own test suite runnable on its own:

```bash
python app/core/provenance.py      # or any other module
```

| Module | Implements | Tests |
|---|---|---|
| `text_kit` | Porter stemming (1980), sentence boundary detection, syllable estimation, bounded edit distance | via dependants |
| `provenance` | Smith-Waterman local alignment with affine gaps (Gotoh), over IDF-weighted n-gram candidate retrieval | 26 |
| `temporal` | A grammar for date expressions: absolute, partial, relative, ranges, seasons, quarters | 115 |
| `quantities` | Quantity extraction and unit conversion across an affine conversion graph | 120 |
| `acoustics` | STFT, spectral peak picking, combinatorial hashing, offset histogramming | 18 |
| `crdt` | Vector clocks, observed-remove sets, last-writer-wins registers | 48 |
| `rules` | Tokenizer, recursive-descent parser and evaluator for a small expression language | 115 |
| `reading` | Optimal-recognition-point calculation, cloze-deletion question generation | 56 |
| `ledger` | SHA-256 hash chain and Merkle tree with membership proofs | 25 |

Roughly 11,000 lines, 523 passing checks. The CRDT suite includes a randomised
convergence property test: 200 seeds × 400 operations × 3–5 replicas, delivered
in shuffled order with duplicates, asserting every replica converges identically.

Three of these are worth singling out:

**Provenance.** Every sentence of a summary is aligned against the source it came
from, returning character offsets rather than a description. Sentences nothing
supports are flagged. A model asked to cite itself frequently invents the
citation; this computes it. Measured at 87 ms for a ten-sentence summary against
a 26,000-word source, including index construction.

**Quantities.** Figures are extracted across the collection and normalised through
a conversion graph whose edges carry affine transforms, so temperature works
through the same machinery as length. Currencies are deliberately isolated with
no inter-currency edges: exchange rates are market prices, not definitions, so
"$12M" and "€12M" are two claims rather than a contradiction.

**Rules.** User-authored conditions are executed server-side, which rules out
`eval`. The language has a hand-written tokenizer and recursive-descent parser;
attribute access is grammatically inexpressible because `.` is not a token. A
static analyser rejects catastrophic-backtracking patterns, and scanning is
chunked so the timeout is actually enforceable against CPython's uninterruptible
regex engine.

---

## Reliability

Every model call passes through `services/gemini_client.py`: retry with backoff,
a multi-model fallback chain, a circuit breaker, and a rate limiter. When every
model is unavailable, text sources fall back to local extractive summarisation
(TextRank implemented in-house over TF-IDF with NumPy power iteration), so a
bad API day degrades output rather than breaking the page.

Model identifiers are pinned in one place and are current as of the last update;
providers retire them without much notice, so that constant is the first thing
to check if summarising starts returning 404s.

---

## Accounts and data

Real accounts: salted password hashes via Werkzeug, Flask-Login sessions, and
every query in the application scoped by `user_id` without exception.

Sessions are bounded. An hour of inactivity ends one, with an in-page countdown
beforehand; "keep me signed in" extends that to thirty days; nothing survives
ninety days. Every session is a row, so `Security & devices` lists them and can
revoke any of them, taking effect on that device's next request.

Originals are content-addressed by SHA-256 under `instance/sources/` and served
only through an ownership-checked route — the request supplies an id, never a
path. Deleting an entry removes its search index, links, annotations, shares and
stored files; export produces a full JSON backup that imports back.

Optional client-side encryption for private notes uses AES-256-GCM with PBKDF2
at 600,000 iterations via WebCrypto. The passphrase never leaves the browser and
the server holds ciphertext it has no key for — which also means a forgotten
passphrase is unrecoverable, stated plainly in the interface rather than hidden.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env               # then set GEMINI_API_KEY and FLASK_SECRET_KEY
python run.py                      # http://localhost:5000
```

Leave `DATABASE_URL` unset for local use and SQLite is created automatically at
`instance/app.db`. Set it and the same code runs against Postgres unchanged.
Schema changes are applied additively at startup by `app/migrations.py`, which
only ever adds columns — it never drops or rewrites anything.

Recording from the microphone or camera needs `localhost` or HTTPS, which is a
browser requirement rather than an application one.

## Deploying

Attach Postgres (most hosts set `DATABASE_URL` for you), set `FLASK_SECRET_KEY`
and `GEMINI_API_KEY`, and set `FORCE_HTTPS=1` once TLS terminates in front of the
app so the session cookie gets its `Secure` flag. Deploy the `Dockerfile` as-is
or use the included `Procfile`. `create_app()` calls `db.create_all()` on first
boot, so there is no manual migration step for a fresh database.

## Tests

```bash
python tests/test_app.py           # end-to-end: routes, data, permissions
python app/core/<module>.py        # each algorithm module, standalone
```

`tests/test_app.py` runs the whole application against a throwaway database
with no AI key configured, which is what verifies the claim above that the
local analysis works without one. Neither suite needs pytest or any other
test dependency.

## Layout

```
app/
  core/          algorithms, no framework or network dependencies
  services/      pipelines, storage, the model client, session handling
  templates/     Jinja templates
  static/        CSS and vanilla JS, no build step
  routes*.py     the HTTP surface
```

Navigation is four entries — Add, Library, Analyse, Components. Anything not on
that path is reachable from Components, which lists the standalone pieces and
what each one implements.
