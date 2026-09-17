"""
provenance -- "show me where this came from."

The single biggest reason people distrust an AI summary is that they
cannot tell which parts are real. A model can write a fluent, confident
sentence about a number that appears nowhere in the document, and
nothing about the sentence looks different from the nine true ones
around it. Asking the model itself for citations does not fix that: a
system that can hallucinate a fact can hallucinate the footnote under
it just as easily.

So this module does not ask. Given a summary and the source text it was
supposedly derived from, it computes -- with a classical string
algorithm, offline, deterministically -- the exact character span of the
source that each summary sentence best aligns to, plus a confidence in
that alignment. The UI can then make every sentence clickable: click it,
jump to the evidence. A sentence we cannot align to anything is flagged
as unsupported, which is exactly the signal a reader needs.

The guarantee this buys is narrow but real, and worth stating plainly:
a *high* score means "these words demonstrably came from here." A *low*
score means "we could not find this in the source," which is strong
evidence of a hallucination but not proof -- a heavily abstractive
sentence that fuses three paragraphs correctly will also score low.
Lexical alignment cannot see meaning. We report a span and a number and
let the reader look; we never claim a sentence is false.

Why two stages
--------------
The obvious implementation -- Smith-Waterman between the sentence and
the whole document -- is O(n*m) and dies on a 50,000-word source. So:

  Stage 1 (cheap, recall-oriented): an inverted index over the source's
  token positions, queried with the sentence's rare unigrams, bigrams
  and trigrams, IDF-weighted so that "of the" cannot outvote a matching
  proper noun. This nominates a handful of windows -- a few hundred
  tokens total instead of fifty thousand.

  Stage 2 (expensive, precision-oriented): a real Smith-Waterman local
  alignment, with affine gaps, against each nominated window. This is
  what produces the actual span, and it is what makes the result robust
  to the things summaries actually do -- reorder clauses, drop
  modifiers, swap an inflected form for its root.

Everything here is stdlib + numpy. No model, no network, no API key.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

try:  # package import -- the normal case, inside the Flask app
    from .text_kit import STOPWORDS, levenshtein, porter_stem, split_sentences, tokenize
except ImportError:  # pragma: no cover - direct execution for the self-test
    from text_kit import STOPWORDS, levenshtein, porter_stem, split_sentences, tokenize


__all__ = ["align_summary", "align_sentence", "coverage_report"]


# ---------------------------------------------------------------------------
# Tuning constants
#
# These are the knobs that decide what "supported" means, so they live at
# the top with their reasoning rather than buried as magic numbers.
# ---------------------------------------------------------------------------

STRONG_THRESHOLD = 0.60
"""At or above this, the sentence is close to lifted from the source --
in practice a verbatim or lightly-edited quote. Safe to present as "this
is where it says that"."""

DEFAULT_MIN_SCORE = 0.28
"""Below this we return no span at all. Chosen empirically: a genuine
paraphrase of a source sentence lands around 0.35-0.55 because it keeps
the content words and loses the function words, while an unrelated
sentence drifts to 0.1-0.2 on incidental stopword collisions. Pointing a
user at a wrong span is worse than pointing them at nothing, so the
threshold sits above the noise floor, not at it."""

# Smith-Waterman scoring. Match is weighted per-token by the token's IDF
# (see _SourceIndex.weight_for) so that matching "quarterly" counts for
# much more than matching "the"; these are the base magnitudes.
MATCH_SCORE = 2.0
NEAR_MATCH_SCORE = 0.9      # same word, one typo / OCR slip apart
MISMATCH_SCORE = -0.9
MISMATCH_WEIGHT_FLOOR = 0.35
"""A mismatch costs MISMATCH_SCORE scaled by the summary token's own
weight, floored at MISMATCH_WEIGHT_FLOOR of full price. Charging a flat
penalty was the first thing tried and it was badly wrong: failing to
match "the" is not evidence of anything, but under a flat penalty it
cost exactly as much as matching "Gibraltar" earned, so any paraphrase
that reworded its function words got torn into fragments. The floor
stops the opposite failure -- a penalty of effectively zero would let an
alignment wander through a desert of stopwords for free."""

# Gaps are deliberately asymmetric, and this is the most important
# modelling decision in the file. Skipping *source* tokens is what
# summarisation IS -- "the board, after some debate, approved" becomes
# "the board approved" -- so a run of skipped source words must stay
# cheap or every real summary sentence would be torn into fragments.
# Skipping *summary* tokens is the opposite signal: words in the summary
# with no counterpart in the source are precisely what we are hunting
# for, so those cost more. Both are gentler than a textbook
# biological-sequence setting, because summaries are *expected* to be
# lossy in a way that two homologous genes are not.
SOURCE_GAP_OPEN = -1.0
SOURCE_GAP_EXTEND = -0.08
SUMMARY_GAP_OPEN = -1.0
SUMMARY_GAP_EXTEND = -0.35

MAX_CANDIDATES = 4
"""How many windows survive stage 1. Four is enough to survive a source
that repeats a phrase in three places; going higher costs linearly in
stage-2 time and, in testing, never changed the winner."""

MAX_WINDOW_TOKENS = 240
"""Hard ceiling on a candidate window's width, which is what bounds the
DP. Stage 2 is O(sentence x window), so without a ceiling a pathological
"sentence" -- a run-on, a bulleted list flattened into one line, a page
of OCR with no full stops -- would quietly turn a 60ms request into a
multi-second one. A span wider than this is not a useful highlight for a
human to read anyway, so the cap costs nothing we want."""

LONG_SENTENCE_TOKENS = 60
"""Past this length a sentence has so many n-grams that stage 1's vote is
decisive, and the runner-up windows are never the winner. We drop to two
candidates there, which is where the ceiling above actually binds."""

ALIGNMENT_SCORE_WEIGHT = 0.65
"""The reported confidence blends two views of the same evidence.

The first is *alignment coverage*: how much of the sentence's weighted
content the Smith-Waterman traceback actually paired up, in order. Order
is strong evidence -- five words in sequence is not a coincidence -- so
it carries most of the weight.

The second is *span overlap*: how much of the sentence's content simply
appears somewhere in the located passage, order ignored. This exists
because of one specific, common failure. A summary writes "as quota
managers had believed" where the source says "than the management quotas
had assumed". The evidence is plainly there, but the two words are
transposed, so a strictly monotonic alignment cannot claim both, and a
pure-alignment score reads that faithful paraphrase as unsupported.
Order-free overlap sees it. Keeping the weight at a third stops it from
rescuing genuine bag-of-words coincidences, which is the failure mode it
would otherwise reintroduce.
"""

SPAN_CONTEXT_RATIO = 0.5
"""Overlap is measured over the aligned span plus this fraction of the
sentence's length in tokens on each side. A summary sentence routinely
picks up a clause from just outside the span the alignment settled on;
looking a little wider is what lets those count."""

_NEG = -1e9  # stands in for -inf; keeps the DP in plain float arithmetic


# ---------------------------------------------------------------------------
# Tokenisation *with character offsets*
#
# This is the fiddly part of the whole module and the reason it cannot
# just call text_kit.tokenize() and be done.
#
# text_kit.normalize() applies NFKD, flattens curly quotes and collapses
# whitespace. All three change string *length*, so any offset computed
# against normalized text is wrong when sliced back out of the original.
# A highlight that is four characters off looks broken to a user, and it
# gets worse the further into the document you go.
#
# So we scan the ORIGINAL string, and fold each matched span in
# isolation. The span boundaries then still refer to the real text, and
# the folded form is what we compare on.
# ---------------------------------------------------------------------------

# Unicode-aware "word-ish run": letters/digits in any script, optionally
# joined by internal apostrophes. Underscore is excluded to match
# text_kit's [a-z0-9]+ behaviour.
_RAW_SPAN_RE = re.compile(r"[^\W_]+(?:['‘’][^\W_]+)*", re.UNICODE)


def _fold_span(raw: str) -> list[str]:
    """Fold one raw span to the token(s) text_kit.tokenize would produce.

    Kept separate from the scanner because the fast path below skips it
    entirely for ordinary ASCII words, which are ~95% of any English
    document and would otherwise dominate index-build time."""
    return tokenize(raw)


def _tokens_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokenise `text`, returning parallel lists of tokens and their
    (start, end) character offsets *into `text` as given*.

    Offsets are half-open and always slice back to the original
    substring -- that is the entire contract of this function, and every
    span this module hands to the UI ultimately comes from here.
    """
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    if not text:
        return tokens, offsets

    for match in _RAW_SPAN_RE.finditer(text):
        raw = match.group()
        start = match.start()

        # Fast path: a plain ASCII alphanumeric run folds to exactly
        # itself, lowercased. No NFKD work, no apostrophe splitting, no
        # second regex pass. This is worth roughly 4x on a large source.
        if raw.isascii() and raw.isalnum():
            tokens.append(raw.lower())
            offsets.append((start, match.end()))
            continue

        parts = _fold_span(raw)
        if not parts:
            # Scripts text_kit's [a-z0-9]+ does not cover (CJK, Cyrillic,
            # Devanagari...) fold to nothing. They carry no anchor for us,
            # but they must not shift anyone else's offsets, so we simply
            # skip them rather than emitting a placeholder token.
            continue
        if len(parts) == 1:
            tokens.append(parts[0])
            offsets.append((start, match.end()))
            continue

        # A span that folds to several tokens ("rock'n'roll", a ligature
        # that decomposes). Walk them through the lowercased raw span in
        # order so each still gets a true offset; if a folded token is
        # unrecognisable in the raw text (an NFKD expansion, say), fall
        # back to attributing the whole span, which is imprecise by a few
        # characters but never wrong about *which* words are involved.
        lowered = raw.lower()
        cursor = 0
        for part in parts:
            found = lowered.find(part, cursor)
            if found < 0:
                tokens.append(part)
                offsets.append((start, match.end()))
                continue
            tokens.append(part)
            offsets.append((start + found, start + found + len(part)))
            cursor = found + len(part)

    return tokens, offsets


# ---------------------------------------------------------------------------
# Stage 1: the source index
# ---------------------------------------------------------------------------


class _SourceIndex:
    """Everything stage 1 needs about one source document, computed once.

    Building this is O(total tokens) and costs tens of milliseconds on a
    20k-word document, which is fine once per document but wasteful once
    per sentence -- hence align_summary threads a single instance through
    all its sentences, and a tiny module-level cache serves repeated
    align_sentence() calls from the same request.
    """

    __slots__ = (
        "source_text", "tokens", "offsets", "stems", "n",
        "_idf", "_unigram_postings", "_ngram_postings", "_weight_cache",
    )

    def __init__(self, source_text: str) -> None:
        self.source_text = source_text
        self.tokens, self.offsets = _tokens_with_offsets(source_text)
        self.stems = [porter_stem(t) for t in self.tokens]
        self.n = len(self.stems)

        # --- IDF over the document's own term frequencies ---------------
        # There is no corpus here, only one document, so "rare" has to
        # mean "rare *in this source*". That is the right notion anyway:
        # a word that appears once in the document is a near-unique
        # address into it, whatever its frequency in English at large.
        counts: dict[str, int] = defaultdict(int)
        for stem in self.stems:
            counts[stem] += 1

        total = max(self.n, 1)
        ceiling = math.log(total + 1.0) or 1.0
        self._idf = {
            stem: math.log((total + 1.0) / (count + 1.0)) / ceiling
            for stem, count in counts.items()
        }
        self._weight_cache: dict[str, float] = {}

        # --- postings -----------------------------------------------------
        # Unigram postings for common words are long and useless (every
        # occurrence of "the" would vote for every window), so we refuse
        # to index a term past a frequency cap. The cap scales with the
        # document: in a 200-word note a word appearing 5 times may still
        # be distinctive, in a 50k-word report it is furniture.
        cap = max(40, int(total * 0.004))
        self._unigram_postings: dict[str, list[int]] = defaultdict(list)
        for position, stem in enumerate(self.stems):
            if stem in STOPWORDS or counts[stem] > cap:
                continue
            self._unigram_postings[stem].append(position)

        # Bigrams and trigrams are the real anchors: even when every
        # individual word is common, "approved the merger" is not. We
        # index them by position of their first token.
        self._ngram_postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        stems = self.stems
        for position in range(self.n - 1):
            self._ngram_postings[(stems[position], stems[position + 1])].append(position)
        for position in range(self.n - 2):
            self._ngram_postings[
                (stems[position], stems[position + 1], stems[position + 2])
            ].append(position)

    # -- scoring helpers ----------------------------------------------------

    def idf(self, stem: str) -> float:
        """Normalised 0..1 rarity of a stem. A stem absent from the source
        scores 1.0: we have never seen it here, so if it turns up in a
        summary it is maximally informative (usually about the sentence
        *not* being supported)."""
        return self._idf.get(stem, 1.0)

    def weight_for(self, token: str, stem: str) -> float:
        """Per-token match weight used by Smith-Waterman.

        Two adjustments on top of raw IDF. Stopwords are halved on top of
        their already-low IDF, because a run of matching function words
        is the classic way a bad alignment fakes a good score. And every
        weight is floored at 0.1 rather than zero: function words still
        carry real evidence about *word order*, which is most of what
        distinguishes a true span from a bag-of-words coincidence.
        """
        cached = self._weight_cache.get(token)
        if cached is not None:
            return cached
        weight = self.idf(stem)
        if token in STOPWORDS:
            weight *= 0.5
        weight = min(1.0, max(0.1, weight))
        self._weight_cache[token] = weight
        return weight

    # -- candidate retrieval ------------------------------------------------

    def candidate_windows(self, sentence_stems: list[str]) -> list[tuple[int, int]]:
        """Nominate a few token windows of the source that plausibly
        contain this sentence's evidence.

        Voting scheme: every shared n-gram drops IDF-weighted votes into
        the bucket its source position falls in, and into the neighbouring
        bucket so a match that straddles a boundary is not split in half.
        Longer n-grams are worth more per token, because contiguity is
        itself evidence -- three words in a row is a far stronger signal
        than the same three words scattered across a paragraph.
        """
        if not sentence_stems or self.n == 0:
            return []

        sentence_length = len(sentence_stems)
        # A window wide enough to hold the sentence plus the material a
        # summary typically compresses out of it, with margin -- capped so
        # that stage 2's cost stays bounded however long the sentence is.
        window_span = min(MAX_WINDOW_TOKENS, max(20, sentence_length * 2 + 12))
        stride = max(6, window_span // 3)
        limit = MAX_CANDIDATES if sentence_length <= LONG_SENTENCE_TOKENS else 2

        votes: dict[int, float] = defaultdict(float)

        def cast(positions: list[int], weight: float) -> None:
            for position in positions:
                bucket = position // stride
                votes[bucket] += weight
                votes[bucket + 1] += weight * 0.5
                if bucket:
                    votes[bucket - 1] += weight * 0.5

        for stem in set(sentence_stems):
            postings = self._unigram_postings.get(stem)
            if postings:
                cast(postings, self.idf(stem))

        for n, multiplier in ((2, 2.0), (3, 3.0)):
            if sentence_length < n:
                continue
            seen: set[tuple[str, ...]] = set()
            for i in range(sentence_length - n + 1):
                gram = tuple(sentence_stems[i : i + n])
                if gram in seen:
                    continue
                seen.add(gram)
                postings = self._ngram_postings.get(gram)
                if not postings or len(postings) > 200:
                    # A phrase repeated 200+ times is boilerplate, not an
                    # anchor; indexing its votes would just smear them.
                    continue
                cast(postings, multiplier * sum(self.idf(s) for s in gram))

        if not votes:
            return []

        ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen: list[int] = []
        for bucket, _score in ranked:
            # Adjacent buckets produce near-identical windows; spending a
            # candidate slot on one is pure waste.
            if any(abs(bucket - taken) <= 1 for taken in chosen):
                continue
            chosen.append(bucket)
            if len(chosen) >= limit:
                break

        windows: list[tuple[int, int]] = []
        for bucket in chosen:
            start = max(0, bucket * stride - stride)
            end = min(self.n, bucket * stride + 2 * stride)
            if end > start:
                windows.append((start, end))
        return windows


# A very small cache so that calling align_sentence() in a loop over a
# ten-sentence summary does not rebuild the index ten times. Keyed by
# hash but verified by identity of content, because a hash collision here
# would silently align against the wrong document.
_INDEX_CACHE: dict[int, _SourceIndex] = {}
_INDEX_CACHE_LIMIT = 4


def _get_index(source_text: str) -> _SourceIndex:
    key = hash(source_text)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached.source_text == source_text:
        return cached
    index = _SourceIndex(source_text)
    if len(_INDEX_CACHE) >= _INDEX_CACHE_LIMIT:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = index
    return index


# ---------------------------------------------------------------------------
# Stage 2: Smith-Waterman local alignment
# ---------------------------------------------------------------------------


def _near_match(a: str, b: str) -> bool:
    """True for two stems that are one edit apart -- an OCR slip, a
    British/American spelling, a typo the model introduced.

    Guarded hard before the edit distance is computed, because
    levenshtein() over every token pair in every window would dominate
    the runtime. Requiring a shared first character and near-equal length
    rejects almost all pairs in one comparison each.
    """
    if len(a) < 4 or len(b) < 4:
        return False
    if a[0] != b[0]:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    return levenshtein(a, b, max_distance=1) <= 1


def _substitution_matrix(
    sentence_stems: list[str],
    weights: list[float],
    window_stems: list[str],
) -> list[list[float]]:
    """The (sentence x window) score table fed to the DP.

    Built with numpy because it is a genuine outer comparison -- encoding
    both stem sequences as integer ids turns "do these two words match"
    into one vectorised equality over the whole table, instead of
    len(sentence) * len(window) Python-level string compares.
    """
    vocabulary: dict[str, int] = {}
    sentence_ids = np.fromiter(
        (vocabulary.setdefault(s, len(vocabulary)) for s in sentence_stems),
        dtype=np.int32,
        count=len(sentence_stems),
    )
    window_ids = np.fromiter(
        (vocabulary.setdefault(s, len(vocabulary)) for s in window_stems),
        dtype=np.int32,
        count=len(window_stems),
    )

    equal = sentence_ids[:, None] == window_ids[None, :]
    weight_array = np.asarray(weights, dtype=np.float64)
    match_scores = weight_array * MATCH_SCORE
    mismatch_scores = MISMATCH_SCORE * (
        MISMATCH_WEIGHT_FLOOR + (1.0 - MISMATCH_WEIGHT_FLOOR) * weight_array
    )
    matrix = np.where(equal, match_scores[:, None], mismatch_scores[:, None])

    # Near-matches are rare enough to handle row by row, and only against
    # the window's *distinct* stems.
    distinct = sorted(set(window_stems))
    for i, stem in enumerate(sentence_stems):
        near = [other for other in distinct if other != stem and _near_match(stem, other)]
        if not near:
            continue
        near_ids = np.asarray([vocabulary[other] for other in near], dtype=np.int32)
        mask = np.isin(window_ids, near_ids) & ~equal[i]
        if mask.any():
            matrix[i, mask] = NEAR_MATCH_SCORE * weights[i]

    return matrix.tolist()


def _smith_waterman(
    substitution: list[list[float]], rows: int, columns: int
) -> tuple[float, float, int, int] | None:
    """Smith-Waterman local alignment with affine gap penalties.

    Returns (best score, matched value, first aligned column, last
    aligned column) with 0-based column indices into the window, or None
    when nothing aligned.

    "Matched value" is the sum of the *positive* substitution scores the
    traceback actually consumed -- the evidence, stripped of the gap and
    mismatch penalties that were needed to decide where the span stops.
    The raw score is the right thing to compare candidate windows on; the
    matched value is the right thing to report a confidence from, because
    a user asking "how much of this sentence is really in there" is
    asking about the evidence, not about the bookkeeping.

    Three matrices, the standard Gotoh formulation:
      H -- best local alignment ending at (i, j) in any state
      E -- best one ending in a gap that consumed source tokens
      F -- best one ending in a gap that consumed summary tokens
    Clamping H at zero is what makes it *local*: an alignment that has
    gone badly is abandoned and restarted rather than dragged along, so
    we find the best matching region rather than the best match over the
    whole window.

    The columns we return come only from diagonal (aligned-pair) steps,
    never from gap steps, so the reported span always begins and ends on
    a word that actually matched. Without that, a trailing cheap
    source-gap could stretch the highlight ten words past the evidence.
    """
    H = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    E = [[_NEG] * (columns + 1) for _ in range(rows + 1)]
    F = [[_NEG] * (columns + 1) for _ in range(rows + 1)]

    best = 0.0
    best_i = best_j = 0

    for i in range(1, rows + 1):
        current_h, previous_h = H[i], H[i - 1]
        current_e = E[i]
        current_f, previous_f = F[i], F[i - 1]
        row_scores = substitution[i - 1]
        for j in range(1, columns + 1):
            e = current_h[j - 1] + SOURCE_GAP_OPEN
            extended = current_e[j - 1] + SOURCE_GAP_EXTEND
            if extended > e:
                e = extended
            current_e[j] = e

            f = previous_h[j] + SUMMARY_GAP_OPEN
            extended = previous_f[j] + SUMMARY_GAP_EXTEND
            if extended > f:
                f = extended
            current_f[j] = f

            h = previous_h[j - 1] + row_scores[j - 1]
            if e > h:
                h = e
            if f > h:
                h = f
            if h < 0.0:
                h = 0.0
            current_h[j] = h

            if h > best:
                best, best_i, best_j = h, i, j

    if best <= 0.0:
        return None

    # --- traceback -------------------------------------------------------
    # No pointer matrix: we re-derive which predecessor produced each cell
    # by comparing against the recurrence. That costs a few extra float
    # comparisons per traceback step -- a rounding error next to the DP
    # itself -- and saves allocating a fourth matrix per candidate.
    i, j = best_i, best_j
    state = 0  # 0 = H, 1 = E (source gap), 2 = F (summary gap)
    first_column = last_column = -1
    matched_value = 0.0
    epsilon = 1e-9

    while i > 0 and j > 0:
        if state == 0:
            value = H[i][j]
            if value <= 0.0:
                break
            pair_score = substitution[i - 1][j - 1]
            if abs(value - (H[i - 1][j - 1] + pair_score)) < epsilon:
                if pair_score > 0.0:
                    matched_value += pair_score
                    if last_column < 0:
                        last_column = j - 1
                    first_column = j - 1
                i -= 1
                j -= 1
            elif abs(value - E[i][j]) < epsilon:
                state = 1
            else:
                state = 2
        elif state == 1:
            if abs(E[i][j] - (H[i][j - 1] + SOURCE_GAP_OPEN)) < epsilon:
                state = 0
            j -= 1
        else:
            if abs(F[i][j] - (H[i - 1][j] + SUMMARY_GAP_OPEN)) < epsilon:
                state = 0
            i -= 1

    if first_column < 0 or last_column < 0:
        return None
    return best, matched_value, first_column, last_column


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _span_overlap(
    index: _SourceIndex,
    sentence_stems: list[str],
    weights: list[float],
    total_weight: float,
    token_start: int,
    token_end: int,
) -> float:
    """Fraction of the sentence's weighted content that appears anywhere
    in source tokens [token_start, token_end), ignoring order.

    Deliberately a *multiset* intersection rather than a set one: a
    sentence that says "hundred" twice should only be credited twice if
    the passage says it twice, otherwise repetition becomes a cheap way
    to inflate the number.
    """
    if total_weight <= 0.0 or token_end <= token_start:
        return 0.0

    available: dict[str, int] = defaultdict(int)
    for stem in index.stems[token_start:token_end]:
        available[stem] += 1

    found = 0.0
    for stem, weight in zip(sentence_stems, weights):
        remaining = available.get(stem, 0)
        if remaining:
            available[stem] = remaining - 1
            found += weight
    return min(1.0, found / total_weight)


def _unmatched(sentence: str, sentence_index: int) -> dict:
    """The shape we return when there is nothing to point at. Kept in one
    place so every "no evidence" path is identical -- the UI branches on
    start == -1 and must never see a half-populated variant."""
    return {
        "sentence": sentence,
        "sentence_index": sentence_index,
        "start": -1,
        "end": -1,
        "score": 0.0,
        "matched_text": "",
        "support": "none",
    }


def _classify(score: float, min_score: float) -> str:
    if score >= STRONG_THRESHOLD:
        return "strong"
    if score >= min_score:
        return "partial"
    return "none"


def _align_against_index(
    sentence: str,
    sentence_index: int,
    index: _SourceIndex,
    min_score: float,
) -> dict:
    """The real worker. Both public entry points funnel through here so
    that a single sentence and a sentence inside a summary can never
    disagree about their own provenance."""
    if index.n == 0:
        return _unmatched(sentence, sentence_index)

    sentence_tokens, _ = _tokens_with_offsets(sentence)
    if not sentence_tokens:
        return _unmatched(sentence, sentence_index)

    sentence_stems = [porter_stem(t) for t in sentence_tokens]

    # A sentence made only of function words ("And so it was.") has no
    # content to anchor on, and any span we returned would be an artefact
    # of stopword collisions. Refusing to answer is the honest result.
    if not any(token not in STOPWORDS and len(token) > 1 for token in sentence_tokens):
        return _unmatched(sentence, sentence_index)

    windows = index.candidate_windows(sentence_stems)
    if not windows:
        return _unmatched(sentence, sentence_index)

    weights = [
        index.weight_for(token, stem)
        for token, stem in zip(sentence_tokens, sentence_stems)
    ]
    # Perfect score: every summary token matched at full weight. Dividing
    # by this is what puts a 6-word sentence and a 40-word sentence on the
    # same 0..1 scale, and it is why a sentence stuffed with words absent
    # from the source cannot score well -- those words inflate the
    # denominator while contributing nothing to the numerator.
    best_possible = MATCH_SCORE * sum(weights)
    if best_possible <= 0.0:
        return _unmatched(sentence, sentence_index)

    rows = len(sentence_stems)
    total_weight = sum(weights)
    context = max(3, int(rows * SPAN_CONTEXT_RATIO))
    best_score = -1.0
    best_span: tuple[int, int] | None = None

    for window_start, window_end in windows:
        window_stems = index.stems[window_start:window_end]
        columns = len(window_stems)
        if columns == 0:
            continue
        substitution = _substitution_matrix(sentence_stems, weights, window_stems)
        outcome = _smith_waterman(substitution, rows, columns)
        if outcome is None:
            continue
        _raw, matched_value, first_column, last_column = outcome

        token_start = window_start + first_column
        token_end = window_start + last_column
        alignment_coverage = matched_value / best_possible
        overlap = _span_overlap(
            index,
            sentence_stems,
            weights,
            total_weight,
            max(0, token_start - context),
            min(index.n, token_end + 1 + context),
        )
        score = (
            ALIGNMENT_SCORE_WEIGHT * alignment_coverage
            + (1.0 - ALIGNMENT_SCORE_WEIGHT) * overlap
        )
        if score > best_score:
            best_score = score
            best_span = (token_start, token_end)

    if best_span is None:
        return _unmatched(sentence, sentence_index)

    token_start, token_end = best_span
    score = max(0.0, min(1.0, best_score))
    support = _classify(score, min_score)
    if support == "none":
        return _unmatched(sentence, sentence_index)

    start = index.offsets[token_start][0]
    end = index.offsets[token_end][1]

    # Pull in sentence-final punctuation that abuts the last matched word.
    # Purely cosmetic, but a highlight that stops one character short of
    # the full stop looks like a bug to the person reading it.
    if end < len(index.source_text) and index.source_text[end] in ".!?":
        end += 1

    return {
        "sentence": sentence,
        "sentence_index": sentence_index,
        "start": start,
        "end": end,
        "score": round(score, 4),
        "matched_text": index.source_text[start:end],
        "support": support,
    }


def align_summary(
    summary: str, source_text: str, *, min_score: float = DEFAULT_MIN_SCORE
) -> list[dict]:
    """Locate each sentence of `summary` in `source_text`.

    Returns one record per summary sentence, in order, each carrying the
    character span of the source it was derived from (or -1/-1 and
    support "none" when we could not find one). `start`/`end` index into
    `source_text` exactly as passed in, so the caller can slice or
    highlight without re-deriving anything.

    The index over the source is built once and shared across every
    sentence, which is why aligning a whole summary costs barely more
    than aligning its first sentence.
    """
    sentences = split_sentences(summary)
    if not sentences:
        return []
    if not source_text or not source_text.strip():
        return [_unmatched(sentence, i) for i, sentence in enumerate(sentences)]

    index = _get_index(source_text)
    return [
        _align_against_index(sentence, i, index, min_score)
        for i, sentence in enumerate(sentences)
    ]


def align_sentence(
    sentence: str, source_text: str, *, min_score: float = DEFAULT_MIN_SCORE
) -> dict:
    """Locate a single sentence in `source_text`.

    Same record shape as align_summary, with sentence_index 0. Useful for
    the interactive case -- a user edits one line of a summary and we
    re-check just that line.

    The sentence is used as given (not re-split), so a caller can pass a
    fragment, a bullet point, or a claim a user typed themselves.
    """
    if not sentence or not sentence.strip():
        return _unmatched(sentence or "", 0)
    if not source_text or not source_text.strip():
        return _unmatched(sentence, 0)
    return _align_against_index(sentence, 0, _get_index(source_text), min_score)


def coverage_report(summary: str, source_text: str) -> dict:
    """Roll the per-sentence alignments up into one document-level verdict.

    `coverage` is the fraction of summary sentences we could tie to
    *something* in the source. It is the number worth surfacing at the top
    of a summary -- "9 of 10 sentences traced to the source" tells a
    reader how much scrutiny the thing deserves before they read a word
    of it -- and `unsupported_sentences` tells them exactly which ones to
    read twice.

    An empty summary reports coverage 0.0 rather than dividing by zero or
    claiming a vacuous 1.0: we have verified nothing, and the number
    should say so.
    """
    alignments = align_summary(summary, source_text)
    total = len(alignments)
    strong = sum(1 for a in alignments if a["support"] == "strong")
    partial = sum(1 for a in alignments if a["support"] == "partial")
    unsupported = [a["sentence"] for a in alignments if a["support"] == "none"]

    return {
        "sentences": total,
        "strong": strong,
        "partial": partial,
        "unsupported": len(unsupported),
        "coverage": round((strong + partial) / total, 4) if total else 0.0,
        "unsupported_sentences": unsupported,
    }


# ---------------------------------------------------------------------------
# Self-test
#
# Run directly:  python provenance.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import random
    import time

    _passes = 0
    _failures = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global _passes, _failures
        if condition:
            _passes += 1
            print(f"PASS  {name}")
        else:
            _failures += 1
            print(f"FAIL  {name}" + (f"  --  {detail}" if detail else ""))

    SOURCE = (
        "The Helios Project began in March 2019 as a joint venture between "
        "the Maritime Institute and two regional universities. Its stated "
        "goal was to map the seasonal migration of bluefin tuna across the "
        "eastern Atlantic using satellite tags rather than vessel sightings. "
        "Over the first eighteen months the team tagged four hundred and "
        "twelve individual fish, of which three hundred and one transmitted "
        "usable data for longer than ninety days. The resulting dataset "
        "overturned a long-standing assumption: the population does not "
        "split into two discrete stocks at the Strait of Gibraltar, but "
        "mixes far more freely than the management quotas had assumed. "
        "Funding was renewed in 2022 for a further five years, with an "
        "expanded remit covering swordfish and albacore. Dr. Alina Reyes, "
        "who directed the tagging programme from its second year, has "
        "argued that the quota framework must be rewritten before the next "
        "assessment cycle in 2027. Critics within the fishing industry "
        "counter that the sample remains too small to justify redrawing "
        "boundaries that have stood since 1998."
    )

    print("-" * 68)
    print("provenance.py self-test")
    print("-" * 68)

    # (a) verbatim sentence -> high score, offsets slice back exactly
    verbatim = (
        "Funding was renewed in 2022 for a further five years, with an "
        "expanded remit covering swordfish and albacore."
    )
    result = align_sentence(verbatim, SOURCE)
    sliced = SOURCE[result["start"]:result["end"]]
    check(
        "(a) verbatim sentence scores strongly",
        result["support"] == "strong" and result["score"] >= 0.9,
        f"score={result['score']} support={result['support']}",
    )
    check(
        "(a) verbatim offsets slice back to the exact source text",
        sliced == verbatim and result["matched_text"] == verbatim,
        f"got {sliced!r}",
    )

    # A second verbatim case, mid-document, to prove offsets do not drift.
    verbatim2 = (
        "Over the first eighteen months the team tagged four hundred and "
        "twelve individual fish, of which three hundred and one transmitted "
        "usable data for longer than ninety days."
    )
    r2 = align_sentence(verbatim2, SOURCE)
    check(
        "(a) offsets do not drift mid-document",
        SOURCE[r2["start"]:r2["end"]] == verbatim2 and r2["score"] >= 0.9,
        f"score={r2['score']} got={SOURCE[r2['start']:r2['end']]!r}",
    )

    # (b) paraphrase -> aligns to roughly the right region
    paraphrase = (
        "Researchers tagged over four hundred tuna, and about three hundred "
        "of the tags kept transmitting for more than ninety days."
    )
    expected_start = SOURCE.index("Over the first eighteen months")
    expected_end = SOURCE.index("usable data for longer than ninety days.") + len(
        "usable data for longer than ninety days."
    )
    r3 = align_sentence(paraphrase, SOURCE)
    overlaps = (
        r3["start"] >= 0
        and r3["start"] < expected_end
        and r3["end"] > expected_start
    )
    check(
        "(b) paraphrase lands in the right region",
        overlaps and r3["support"] in ("strong", "partial"),
        f"span=({r3['start']},{r3['end']}) expected~({expected_start},{expected_end}) "
        f"score={r3['score']} text={r3['matched_text']!r}",
    )

    paraphrase2 = (
        "The study found that the fish do not separate into two distinct "
        "stocks at Gibraltar as quota managers had believed."
    )
    r4 = align_sentence(paraphrase2, SOURCE)
    check(
        "(b) second paraphrase lands on the Gibraltar claim",
        r4["support"] in ("strong", "partial") and "Gibraltar" in r4["matched_text"],
        f"score={r4['score']} text={r4['matched_text']!r}",
    )

    # (c) content that is simply not in the source
    absent = (
        "The committee voted unanimously to relocate the headquarters to "
        "Reykjavik after a lengthy debate about municipal parking permits."
    )
    r5 = align_sentence(absent, SOURCE)
    check(
        "(c) absent content returns support 'none'",
        r5["support"] == "none" and r5["start"] == -1 and r5["matched_text"] == "",
        f"score={r5['score']} support={r5['support']} text={r5['matched_text']!r}",
    )

    # multi-sentence summary + coverage report
    summary = (
        f"{verbatim} {paraphrase} {absent} "
        "Dr. Alina Reyes has argued the quota framework must be rewritten "
        "before 2027."
    )
    aligned = align_summary(summary, SOURCE)
    check(
        "align_summary returns one record per sentence, in order",
        len(aligned) == 4 and [a["sentence_index"] for a in aligned] == [0, 1, 2, 3],
        f"got {len(aligned)} records",
    )
    check(
        "align_summary records all have the full key set",
        all(
            set(a) == {
                "sentence", "sentence_index", "start", "end",
                "score", "matched_text", "support",
            }
            for a in aligned
        ),
    )
    check(
        "'Dr.' abbreviation did not split into its own sentence",
        any("Reyes" in a["sentence"] for a in aligned),
    )
    report = coverage_report(summary, SOURCE)
    check(
        "coverage_report counts the unsupported sentence",
        report["sentences"] == 4
        and report["unsupported"] == 1
        and abs(report["coverage"] - 0.75) < 1e-6
        and len(report["unsupported_sentences"]) == 1,
        str(report),
    )

    # --- edge cases -------------------------------------------------------
    check("empty summary -> []", align_summary("", SOURCE) == [])
    check(
        "empty source -> all unmatched",
        all(a["support"] == "none" for a in align_summary(summary, "")),
    )
    check("both empty -> []", align_summary("", "") == [])
    check(
        "coverage_report on empty summary does not divide by zero",
        coverage_report("", SOURCE)["coverage"] == 0.0,
    )
    check(
        "summary longer than source is handled",
        len(align_summary(SOURCE + " " + SOURCE, "Bluefin tuna migrate.")) > 0,
    )
    check(
        "function-word-only sentence is refused",
        align_sentence("And so it was that they did.", SOURCE)["support"] == "none",
    )
    check(
        "whitespace-only sentence is refused",
        align_sentence("   ", SOURCE)["support"] == "none",
    )
    check(
        "punctuation-only sentence is refused",
        align_sentence("!?!", SOURCE)["support"] == "none",
    )

    # unicode: accents, curly quotes and CJK must not corrupt offsets
    uni_source = (
        "Le café was opened in 1912 by Émile Vaudreuil, a Montréal printer "
        "who had grown tired of the trade. 東京 offices followed in 1968. "
        "The company’s archive was donated to the city in 2004."
    )
    uni_sentence = "The company’s archive was donated to the city in 2004."
    r6 = align_sentence(uni_sentence, uni_source)
    check(
        "unicode source: offsets slice back correctly",
        r6["start"] >= 0
        and uni_source[r6["start"]:r6["end"]] == r6["matched_text"]
        and "archive was donated to the city in 2004" in r6["matched_text"],
        f"span=({r6['start']},{r6['end']}) text={r6['matched_text']!r}",
    )
    r7 = align_sentence("Émile Vaudreuil was a printer from Montréal.", uni_source)
    check(
        "unicode source: accented names still anchor",
        r7["support"] in ("strong", "partial")
        and "Vaudreuil" in uni_source[r7["start"]:r7["end"]],
        f"score={r7['score']} text={r7['matched_text']!r}",
    )

    # repeated calls must agree exactly -- this is a feature whose whole
    # value is that a user can re-check it and get the same answer.
    repeated = [align_sentence(paraphrase, SOURCE) for _ in range(3)]
    check("results are deterministic across calls", all(r == repeated[0] for r in repeated))

    # negative controls: things that share topic or function words with the
    # source but make a claim it does not support must not be rescued by the
    # order-free overlap term.
    negatives = [
        "It was the case that they had not been able to do so, and that was that.",
        "The project tagged nine thousand sharks in the Pacific and proved the "
        "stocks were entirely separate.",
        "Parking permits in the municipal district were debated at length by the "
        "relocation committee.",
    ]
    negative_scores = [align_sentence(n, SOURCE)["score"] for n in negatives]
    check(
        "negative controls stay below the support threshold",
        all(score < DEFAULT_MIN_SCORE for score in negative_scores),
        f"scores={[round(s, 3) for s in negative_scores]}",
    )

    # every returned span must be self-consistent
    check(
        "matched_text always equals source_text[start:end]",
        all(
            a["matched_text"] == SOURCE[a["start"]:a["end"]]
            for a in aligned
            if a["start"] >= 0
        ),
    )

    # (d) speed on a large synthetic document
    random.seed(20240917)
    vocabulary = [
        "harbour", "sediment", "quota", "acoustic", "buoy", "calibration",
        "salinity", "transponder", "biomass", "thermocline", "trawler",
        "recruitment", "spawning", "isotope", "otolith", "bycatch",
        "longline", "survey", "anomaly", "gradient", "plankton", "larval",
        "moratorium", "hatchery", "estuary", "benthic", "pelagic", "fishery",
    ]
    connectors = [
        "the", "of", "in", "and", "was", "were", "that", "which", "a", "to",
        "for", "with", "by", "from", "at", "on",
    ]

    def make_sentence() -> str:
        words = []
        for _ in range(random.randint(12, 26)):
            words.append(
                random.choice(vocabulary)
                if random.random() < 0.45
                else random.choice(connectors)
            )
        words[0] = words[0].capitalize()
        return " ".join(words) + "."

    big_sentences = [make_sentence() for _ in range(1400)]
    # Plant ten real, findable sentences at spread-out positions.
    planted = [
        "The Kestrel survey recorded an unprecedented thermocline anomaly "
        "of eleven degrees near the Faroe shelf in August 2021.",
        "Otolith microchemistry from the Skagerrak samples indicated three "
        "distinct natal origins rather than the single origin assumed.",
        "Bycatch of juvenile haddock fell by thirty-one percent after the "
        "square-mesh panel was made mandatory in 2017.",
        "A moratorium on longline gear inside the Rockall box took effect "
        "in January 2015 and was lifted four years later.",
        "The acoustic transponder array at Ullapool logged sixty-two "
        "thousand detections during the spring spawning run.",
        "Larval drift modelling suggested the estuary hatchery contributes "
        "less than five percent of recruitment in the outer firth.",
        "Sediment cores taken off Shetland preserved a clear isotope "
        "signature of the 1976 collapse in pelagic biomass.",
        "Calibration of the Simrad echosounder drifted by nine decibels "
        "over the course of the eighteen-day cruise.",
        "The benthic trawl survey was suspended in 2020 and resumed with a "
        "reduced station list the following autumn.",
        "Salinity at the mouth of the Clyde rose sharply after the "
        "unusually dry summer of 2018, according to buoy records.",
    ]
    for offset, sentence in enumerate(planted):
        big_sentences.insert(60 + offset * 130, sentence)

    big_source = " ".join(big_sentences)
    word_total = len(big_source.split())

    big_summary = " ".join(planted)

    _INDEX_CACHE.clear()
    t0 = time.perf_counter()
    big_results = align_summary(big_summary, big_source)
    elapsed = time.perf_counter() - t0

    check(
        "(d) large document: 10-sentence summary under 1.0s",
        elapsed < 1.0,
        f"took {elapsed:.3f}s on {word_total} words",
    )
    located = sum(
        1
        for record, original in zip(big_results, planted)
        if record["start"] >= 0 and original.rstrip(".") in big_source[
            max(0, record["start"] - 5):record["end"] + 5
        ]
    )
    check(
        "(d) large document: all ten planted sentences located exactly",
        located == 10,
        f"located {located}/10",
    )

    # A pathological "sentence" -- a flattened bullet list, or OCR with no
    # full stops -- must not blow the time budget.
    run_on = " ".join(random.choice(vocabulary + connectors) for _ in range(300))
    _INDEX_CACHE.clear()
    _get_index(big_source)
    t0 = time.perf_counter()
    align_sentence(run_on, big_source)
    run_on_elapsed = time.perf_counter() - t0
    check(
        "(d) 300-token run-on sentence stays bounded",
        run_on_elapsed < 0.5,
        f"took {run_on_elapsed:.3f}s",
    )

    # Timing with a cold index, reported for the record.
    _INDEX_CACHE.clear()
    t0 = time.perf_counter()
    _SourceIndex(big_source)
    index_time = time.perf_counter() - t0

    print("-" * 68)
    print(f"large document: {word_total:,} words, {len(big_sentences)} sentences")
    print(f"  index build            : {index_time * 1000:7.1f} ms")
    print(f"  align 10-sentence summary: {elapsed * 1000:7.1f} ms (index included)")
    print("-" * 68)
    print(f"{_passes} passed, {_failures} failed")
    raise SystemExit(1 if _failures else 0)
