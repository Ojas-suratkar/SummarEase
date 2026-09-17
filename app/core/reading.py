"""
reading -- the pure-Python engine behind SummarEase's RSVP speed reader.

RSVP (Rapid Serial Visual Presentation) flashes one word at a time at a
fixed point on screen. Ordinary reading spends a large share of its time
on saccades -- the ballistic eye jumps between fixations -- plus the
regressions back to words you half-missed. Hold the words still and the
eye stops travelling, and most readers gain something like 1.5-2.5x on
familiar material with comprehension that holds up.

"Holds up" is the load-bearing phrase, and it is why this module does
more than emit words on a timer:

  * prepare_words()           -- where the eye should land, and for how long
  * pacing_plan()             -- turning a wpm target into real milliseconds
  * comprehension_questions() -- did you actually take it in?
  * adapt_wpm()               -- move the speed based on that answer
  * reading_fitness()         -- is any of this getting better over time?

Nothing here calls an AI model, or the network, or a database. Every
function is deterministic: same input, same output, testable by hand.
That is deliberate. A speed-reading tool that lets you tell yourself you
read at 900 wpm is a toy. One that measures you, and moves the dial for
you, is an instrument -- but only if the measurement is something you
can inspect. See comprehension_questions() for an honest account of what
these questions can and cannot detect.
"""
from __future__ import annotations

import math
import random
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

try:  # package import
    from .text_kit import (
        STOPWORDS,
        content_tokens,
        count_syllables,
        levenshtein,
        normalize,
        porter_stem,
        split_sentences,
        tokenize,
    )
except ImportError:  # standalone / self-test import
    from text_kit import (  # type: ignore[no-redef]
        STOPWORDS,
        content_tokens,
        count_syllables,
        levenshtein,
        normalize,
        porter_stem,
        split_sentences,
        tokenize,
    )

__all__ = [
    "prepare_words",
    "pacing_plan",
    "comprehension_questions",
    "adapt_wpm",
    "reading_fitness",
    "DEFAULT_WPM",
    "WPM_FLOOR",
    "WPM_CEILING",
]

DEFAULT_WPM = 300
WPM_FLOOR = 150
WPM_CEILING = 900

# Two frames at 60Hz. Anything shorter is not a word the eye saw, it is a
# flicker, so no per-word slot is ever allowed below this.
_MIN_FRAME_MS = 34

# Mirrors text_kit's private abbreviation table. Duplicated rather than
# imported so this module does not depend on another module's privates.
_ABBREV = frozenset(
    """mr mrs ms dr prof sr jr st vs etc eg ie fig al inc ltd co corp dept est
    approx no vol ed pp jan feb mar apr jun jul aug sep sept oct nov dec us uk
    eu un ca cf dept fl ft lb oz hr min sec""".split()
)

_TOKEN_RE = re.compile(r"\S+")
_PARAGRAPH_GAP_RE = re.compile(r"\n[ \t\r]*\n")
_ACRONYM_RE = re.compile(r"^(?:[A-Za-z]\.){1,}[A-Za-z]?$")
_HAS_DIGIT_RE = re.compile(r"\d")
_LETTER_RE = re.compile(r"[A-Za-z0-9]")

# Length-preserving typographic fold. text_kit.normalize() is the right
# tool for scoring, but it collapses whitespace and runs NFKD, both of
# which move character offsets -- and prepare_words() promises offsets
# that index back into the caller's original string. Every substitution
# here is one character for one character.
_FOLD = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)

_CLOSERS = "\"')]}»”’"
_OPENERS = "\"'([{«“‘"
_MINOR_PAUSE_CHARS = ",;:-"


# ---------------------------------------------------------------------------
# Word preparation
# ---------------------------------------------------------------------------


def _is_sentence_end(token: str) -> bool:
    """True when this whitespace-delimited token genuinely closes a
    sentence, rather than merely ending in a dot."""
    core = token.rstrip(_CLOSERS)
    if not core or core[-1] not in ".!?":
        return False

    stripped = core.rstrip(".!?").lstrip(_OPENERS)
    if not stripped:
        return False

    # "U.S." / "e.g." written as one token -- an acronym, not an ending.
    if _ACRONYM_RE.match(core.lstrip(_OPENERS)):
        return False

    last = stripped.lower().rstrip(".")
    if last in _ABBREV:
        return False
    # "J." in "J. R. R. Tolkien" -- a bare initial.
    if len(last) == 1 and last.isalpha():
        return False
    # A bare "1." or "2." is almost always a numbered-list marker.
    if last.isdigit() and core.endswith("."):
        return False
    return True


def _minor_pause(token: str) -> bool:
    """Comma, semicolon, colon or dash: a clause boundary, not a full stop."""
    core = token.rstrip(_CLOSERS)
    return bool(core) and core[-1] in _MINOR_PAUSE_CHARS and not core[-1].isalnum()


def orp_index(word: str) -> int:
    """The Optimal Recognition Point: which character of `word` the reader
    should be fixating.

    THE RULE, and why it is this rule
    ---------------------------------
    In normal reading the eye does not land in the middle of a word. It
    lands slightly left of centre -- the "preferred viewing location",
    about 30-35% of the way in -- because the useful visual field is
    asymmetric: you resolve roughly 4 characters to the left of fixation
    and 8-10 to the right. Fixate the true centre of a long word and its
    opening letters fall outside the span that identifies it, and word
    onset is where most of the identifying information lives.

    So the pivot is proportional (~32% in) but *compressed*, because the
    useful leftward span is fixed at about 4 characters: past that, moving
    the pivot further right buys nothing and starts pushing the word's
    beginning out of view. That gives the bands:

        length  1      -> 0     the only character there is
        length  2-5    -> 1     ~20-50%; second character
        length  6-9    -> 2     ~22-33%
        length 10-13   -> 3     ~23-30%
        length 14+     -> 4     capped: the leftward span runs out

    Refinements on top of the bands:

    1. Punctuation is not fixated. The bands are computed on the word's
       alphanumeric core and the returned index is shifted back over any
       leading quote or bracket, so `"quickly` pivots on the same letter
       as `quickly`. Without this, quoted dialogue jitters left and right
       around the fixation line, which is exactly the eye movement RSVP
       exists to remove.
    2. The pivot never lands on a non-letter inside the word. In
       `well-known` the band puts index 3 on the letter `l`; in `re-entry`
       it would land on the hyphen, so it steps forward to the next
       letter. A coloured hyphen is a pivot the eye cannot lock onto.
    3. The index is clamped inside the token, which matters for one- and
       two-character tokens and for tokens that are pure punctuation.

    Getting this wrong is the single biggest quality gap between RSVP
    implementations. Centre-pivoting (`len // 2`) feels fine on short
    words and falls apart on long ones, which are precisely the words
    that need the help.
    """
    if not word:
        return 0

    lead = 0
    while lead < len(word) and word[lead] in _OPENERS:
        lead += 1

    trail = len(word)
    while trail > lead and word[trail - 1] in _CLOSERS:
        trail -= 1

    core = word[lead:trail]
    if not core:
        return 0

    n = len(core)
    if n == 1:
        base = 0
    elif n <= 5:
        base = 1
    elif n <= 9:
        base = 2
    elif n <= 13:
        base = 3
    else:
        base = 4

    # Refinement 2: never pivot on an interior non-letter.
    idx = base
    while idx < n and not _LETTER_RE.match(core[idx]):
        idx += 1
    if idx >= n:
        idx = base  # all punctuation; fall back to the band

    return max(0, min(len(word) - 1, lead + idx))


def _delay_multiplier(
    token: str,
    *,
    sentence_end: bool,
    paragraph_end: bool,
    minor_pause: bool,
) -> float:
    """Relative display time for one token, 1.0 being an average word.

    The constants are not arbitrary. Two families of evidence set them:

    LEXICAL COST (how long the word itself takes to identify)
      * Length. Gaze duration rises roughly linearly with word length past
        the ~5-character mean of English prose -- about 15-20 ms per extra
        character against a ~250 ms baseline fixation, so ~0.05x per
        character. Capped at +0.45 because the curve flattens: a 20-letter
        word is not four times a 10-letter word, it is decomposed into
        morphemes and read in parts.
      * Syllables. Phonological encoding load beyond two syllables, worth
        less than raw length because the two correlate; +0.06 each,
        capped at +0.30.
      * Digits. A numeral has no lexical entry to recognise -- "1,482" is
        decoded character by character -- so it gets a flat +0.35.
      * Acronyms (all-caps, 3+ chars) are read letter-wise too: +0.20.
      * Very long words (12+) get a further +0.20 on top of the length
        term, standing in for the rarity that makes them long.
      * Short function words go the other way. "the", "of", "and" are
        identified parafoveally in normal reading and are frequently
        skipped outright; giving them a full slot is wasted time, so they
        drop to 0.85.

    STRUCTURAL PAUSE (how long the reader needs *after* the word)
      Comprehension is not word recognition. Clause and sentence
      boundaries are where the reader integrates what they just took in --
      the wrap-up effect, visible as a real spike in fixation time at
      sentence-final words. RSVP removes the natural opportunity to pause,
      so it has to be put back deliberately, or sentences run together
      and comprehension collapses well before the reader notices.

        minor (, ; : -)   1.5x   clause boundary
        sentence (. ! ?)  2.0x   integrate the proposition
        paragraph         2.5x   integrate the topic, reset

      The pause is added to the lexical cost rather than multiplied by it
      (`1.0 + (pause - 1.0)`), so a plain sentence-ending word lands on
      exactly 2.0x while a long one gets its extra time on top instead of
      compounding into a jarring three-second stall.
    """
    core = token.strip(_OPENERS + _CLOSERS + ".,;:!?")
    letters = re.sub(r"[^A-Za-z']", "", core)
    length = len(core) if core else len(token)

    multiplier = 1.0

    # --- lexical cost -----------------------------------------------------
    if length > 5:
        multiplier += min(0.45, (length - 5) * 0.05)
    if length >= 12:
        multiplier += 0.20

    if letters:
        syllables = count_syllables(letters)
        if syllables > 2:
            multiplier += min(0.30, (syllables - 2) * 0.06)

    if _HAS_DIGIT_RE.search(core):
        multiplier += 0.35
    elif len(core) >= 3 and core.isupper():
        multiplier += 0.20

    # --- structural pause -------------------------------------------------
    pause = 1.0
    if paragraph_end:
        pause = 2.5
    elif sentence_end:
        pause = 2.0
    elif minor_pause:
        pause = 1.5

    # The short-function-word discount applies only to words in the middle
    # of a clause. "The deficit did not recover on days off." ends on a
    # stopword, but the reader still needs the full wrap-up pause there --
    # the time is for integrating the sentence, not for reading "off".
    if pause == 1.0 and length <= 3 and letters and letters.lower() in STOPWORDS:
        multiplier = 0.85

    multiplier += pause - 1.0

    # Clamp: below 0.8 a word flickers, above 3.5 the reader's attention
    # wanders and the rhythm that makes RSVP work is broken.
    return round(max(0.8, min(3.5, multiplier)), 3)


def prepare_words(text: str) -> list[dict]:
    """Turn a document into the RSVP stream the client renders.

    Each entry is::

        {"word": str,              # exactly as it appears, punctuation kept
         "orp": int,               # character index to pin and colour
         "delay_multiplier": float,# relative display time, 1.0 = average
         "is_sentence_end": bool,
         "is_paragraph_end": bool,
         "index": int,
         "char_offset": int}       # offset into `text`, for highlighting

    Punctuation stays attached to its word. Stripping it would be simpler
    to render but it destroys the pacing signal -- the comma *is* the
    instruction to pause -- and it hides sentence structure from a reader
    who has no line breaks left to infer it from.
    """
    if not text:
        return []

    folded = text.translate(_FOLD)
    matches = list(_TOKEN_RE.finditer(folded))
    if not matches:
        return []

    words: list[dict] = []
    for i, match in enumerate(matches):
        token = match.group(0)

        # Paragraph end: a blank line in the gap that follows, or the end
        # of the document.
        if i + 1 < len(matches):
            gap = folded[match.end() : matches[i + 1].start()]
            paragraph_end = bool(_PARAGRAPH_GAP_RE.search(gap))
        else:
            paragraph_end = True

        sentence_end = _is_sentence_end(token)
        # A paragraph break closes a sentence even without punctuation --
        # headings and list items usually have none, and running them into
        # the next paragraph is the fastest way to lose the reader.
        if paragraph_end:
            sentence_end = True

        minor = (not sentence_end) and _minor_pause(token)

        words.append(
            {
                "word": token,
                "orp": orp_index(token),
                "delay_multiplier": _delay_multiplier(
                    token,
                    sentence_end=sentence_end,
                    paragraph_end=paragraph_end,
                    minor_pause=minor,
                ),
                "is_sentence_end": sentence_end,
                "is_paragraph_end": paragraph_end,
                "index": i,
                "char_offset": match.start(),
            }
        )

    return words


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


def pacing_plan(words: list[dict], wpm: int) -> dict:
    """Convert relative multipliers into a millisecond schedule.

    The multipliers are *normalised to a mean of 1.0* before being scaled,
    so a document played at 400 wpm really does finish in
    `len(words) / 400` minutes. Time is redistributed across the stream,
    never added to it. Without this the dial lies: multipliers average
    around 1.15-1.25 on ordinary prose, so "400 wpm" would silently play
    at ~330 and every downstream number -- estimated finish, effective
    wpm, the adaptation loop -- would inherit the error.

    Rounding carry: the residual from each `round()` is carried into the
    next slot, so cumulative drift stays under a millisecond instead of
    accumulating to whole seconds over a long document.

    Returns ``{"total_ms": int, "per_word_ms": [int],
    "estimated_minutes": float}``.
    """
    if not words:
        return {"total_ms": 0, "per_word_ms": [], "estimated_minutes": 0.0}

    wpm = max(50, min(1500, int(wpm)))
    base_ms = 60000.0 / wpm

    multipliers = [float(w.get("delay_multiplier", 1.0) or 1.0) for w in words]
    mean = sum(multipliers) / len(multipliers)
    if mean <= 0:
        mean = 1.0

    per_word_ms: list[int] = []
    carry = 0.0
    for multiplier in multipliers:
        exact = base_ms * (multiplier / mean) + carry
        slot = int(round(exact))
        if slot < _MIN_FRAME_MS:
            slot = _MIN_FRAME_MS
        carry = exact - slot
        per_word_ms.append(slot)

    total_ms = sum(per_word_ms)
    return {
        "total_ms": total_ms,
        "per_word_ms": per_word_ms,
        "estimated_minutes": round(total_ms / 60000.0, 2),
    }


# ---------------------------------------------------------------------------
# Comprehension questions
#
# Everything below is classical NLP: term weighting, morphological
# matching, edit distance. No model, no API call, no training data.
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"([$£€]?)"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(%|percent|million|billion|trillion|thousand|kg|km|mph|bn|m)?"
    r"(?![\w])",
    re.IGNORECASE,
)


def _cap_class(word: str) -> str:
    if len(word) >= 2 and word.isupper():
        return "upper"
    if word[:1].isupper():
        return "title"
    return "lower"


def _suffix_class(word: str) -> str:
    """A crude, purely morphological stand-in for part of speech.

    It is wrong often enough that you would not build a parser on it, but
    it is right often enough for the only job it has here: keeping a
    distractor from being ruled out on grammar alone. If the sentence is
    "the committee ______ the proposal", offering "quickly" as an option
    is offering nothing -- the reader eliminates it without recalling a
    single thing about the document.
    """
    w = word.lower()
    if w.endswith("ing"):
        return "ing"
    if w.endswith("ed"):
        return "ed"
    if w.endswith("ly"):
        return "ly"
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism")):
        return "nominal"
    if w.endswith(("ous", "ive", "ful", "able", "ible", "al", "ic", "est")):
        return "adjectival"
    if w.endswith(("er", "or")):
        return "agent"
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return "plural"
    return "bare"


def _salience_map(sentences: list[str]) -> dict[str, float]:
    """TF-IDF over the document, with sentences as the pseudo-documents.

    A word that shows up in every sentence is the document's subject and
    therefore a terrible thing to blank out -- the reader fills it from
    the title. A word that shows up in one or two sentences carries the
    specifics, which is what we want to test. That is exactly the shape
    of inverse document frequency, so we borrow it, using sentences as
    the unit rather than a corpus we do not have.
    """
    n = max(1, len(sentences))
    term_freq: dict[str, int] = {}
    sentence_freq: dict[str, int] = {}
    proper: set[str] = set()

    for sentence in sentences:
        seen: set[str] = set()
        raw_words = re.findall(r"[A-Za-z][A-Za-z'\-]*", sentence)
        for position, raw in enumerate(raw_words):
            lower = raw.lower()
            term_freq[lower] = term_freq.get(lower, 0) + 1
            if lower not in seen:
                seen.add(lower)
                sentence_freq[lower] = sentence_freq.get(lower, 0) + 1
            # Capitalised anywhere but sentence-initial: likely a name.
            if position > 0 and raw[:1].isupper() and not raw.isupper():
                proper.add(lower)

    salience: dict[str, float] = {}
    for term, tf in term_freq.items():
        if term in STOPWORDS or len(term) < 3:
            continue
        sf = sentence_freq.get(term, 1)
        idf = math.log(1.0 + (n / sf))
        score = (1.0 + math.log(tf)) * idf
        score *= 1.0 + 0.03 * min(len(term), 14)  # longer -> more specific
        if term in proper:
            score *= 1.35  # names are the facts people are meant to retain
        salience[term] = score
    return salience


def _distractor_pool(
    answer: str,
    salience: dict[str, float],
    *,
    exclude: set[str],
    wanted: int = 3,
    strict: bool = False,
) -> list[str]:
    """Find plausible wrong answers for `answer`, drawn from the document.

    WHY THIS IS THE HARD PART. A multiple-choice question is only a
    measurement if the wrong options are live. Distractors pulled from a
    generic word list, or generated by length alone, leak the answer
    through surface cues -- the longest option, the only capitalised one,
    the only plural, the only one that could grammatically fit. A reader
    who took in nothing scores 75%, and the adaptation loop then happily
    pushes them to 900 wpm.

    So candidates must clear four filters, relaxed in order only if too
    few survive:

      1. Same capitalisation class -- a lone capitalised option in a list
         of lowercase ones is a free answer.
      2. Same morphological class -- see _suffix_class; keeps every option
         grammatical in the gap.
      3. Similar length and syllable count -- removes the "pick the long
         one" heuristic.
      4. Edit distance > 2 from the answer and a different Porter stem --
         "policies" against "policy" is not a distractor, it is a typo,
         and it makes the question unfair in the opposite direction.

    Survivors are ranked by salience, so the wrong answers are themselves
    important words from the same document. The reader cannot fall back on
    topic plausibility; they have to remember which of four on-topic,
    same-shaped words was actually in that sentence.
    """
    answer_lower = answer.lower()
    answer_stem = porter_stem(answer_lower)
    answer_cap = _cap_class(answer)
    answer_suffix = _suffix_class(answer)
    answer_syllables = count_syllables(answer_lower)

    blocked = {answer_lower} | {e.lower() for e in exclude}

    candidates: list[tuple[float, str]] = []
    for term, score in salience.items():
        if term in blocked or term in STOPWORDS or len(term) < 3:
            continue
        if porter_stem(term) == answer_stem:
            continue
        if levenshtein(term, answer_lower, max_distance=2) <= 2:
            continue
        candidates.append((score, term))

    candidates.sort(key=lambda pair: (-pair[0], pair[1]))

    def _shaped(term: str) -> str:
        """Present the distractor with the answer's capitalisation so case
        is never a tell."""
        if answer_cap == "upper":
            return term.upper()
        if answer_cap == "title":
            return term.capitalize()
        return term

    tiers: list[Any] = [
        # tier 1: same morphology, tight length and syllable match
        lambda t: (
            _suffix_class(t) == answer_suffix
            and abs(len(t) - len(answer_lower)) <= 2
            and abs(count_syllables(t) - answer_syllables) <= 1
        ),
        # tier 2: same morphology, looser length
        lambda t: (
            _suffix_class(t) == answer_suffix
            and abs(len(t) - len(answer_lower)) <= 4
        ),
        # tier 3: length only
        lambda t: abs(len(t) - len(answer_lower)) <= 3,
        # tier 4: anything salient left
        lambda t: True,
    ]

    # `strict` stops at the two morphology-preserving tiers. Sentence
    # mutation needs that: dropping a length-matched but grammatically
    # wrong word into a real sentence produces a word salad that the
    # reader rejects on syntax, not on memory, which is the exact failure
    # this whole function exists to avoid. In a cloze the gap hides the
    # grammar, so the looser tiers are safe there.
    if strict:
        tiers = tiers[:2]

    picked: list[str] = []
    taken: set[str] = set()
    for accepts in tiers:
        for _score, term in candidates:
            if len(picked) >= wanted:
                break
            if term in taken:
                continue
            if accepts(term):
                taken.add(term)
                picked.append(_shaped(term))
        if len(picked) >= wanted:
            break

    return picked[:wanted]


def _format_number(value: float, decimals: int, grouped: bool) -> str:
    if decimals > 0:
        text = f"{value:,.{decimals}f}" if grouped else f"{value:.{decimals}f}"
    else:
        text = f"{round(value):,d}" if grouped else f"{int(round(value))}"
    return text


def _numeric_distractors(
    prefix: str, digits: str, unit: str, text: str, *, wanted: int = 3
) -> list[str]:
    """Perturb a figure without changing its shape.

    "$1,200" must not be offered against "3" and "eleven thousand" -- the
    formatting alone would answer the question. Every distractor keeps the
    currency symbol, the thousands grouping, the decimal places and the
    unit, and differs only in value. Perturbations are multiplicative so
    they stay in the same order of magnitude, with additive fallbacks for
    small integers where scaling collapses (0.5 * 2 and 1.5 * 2 both round
    to plausible but adjacent values). Any candidate that happens to
    appear elsewhere in the document is discarded -- otherwise there are
    two defensible answers.
    """
    grouped = "," in digits
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    try:
        value = float(digits.replace(",", ""))
    except ValueError:
        return []

    unit_suffix = f" {unit}" if unit and unit not in {"%"} else (unit or "")

    def render(v: float) -> str:
        return f"{prefix}{_format_number(v, decimals, grouped)}{unit_suffix}"

    answer = render(value)
    lowered = text.lower()

    candidates: list[float] = []
    for factor in (1.5, 0.6, 2.0, 0.75, 1.25, 0.4, 3.0):
        candidates.append(value * factor)
    step = max(1.0, abs(value) * 0.1)
    for delta in (step, -step, 2 * step, 5 * step):
        candidates.append(value + delta)

    out: list[str] = []
    seen = {answer.lower()}
    for candidate in candidates:
        if len(out) >= wanted:
            break
        if candidate < 0:
            continue
        rendered = render(candidate)
        key = rendered.lower()
        if key in seen:
            continue
        # Must not be a figure the document actually states.
        bare = _format_number(candidate, decimals, grouped)
        if bare.lower() in lowered:
            continue
        seen.add(key)
        out.append(rendered)
    return out


def _fragment(sentence: str, words: int = 9) -> str:
    parts = sentence.split()
    if len(parts) <= words:
        return sentence.rstrip(".!?")
    return " ".join(parts[:words]).rstrip(",;:") + " …"


def _mutate_sentence(
    sentence: str, salience: dict[str, float], rng: random.Random, forbidden: set[str]
) -> str | None:
    """Turn a real sentence into one that was never in the document.

    This is recognition-memory lure construction: the wrong options are
    sentences the reader half-remembers, altered in one specific place.
    A distractor that is obviously alien -- different topic, different
    register, different length -- tests nothing, because the reader picks
    the familiar-sounding one and is right for the wrong reason. Altering
    a single salient content word, a negation, or a figure means every
    option sounds like the document and only precise recall separates
    them.

    Returns None if no mutation produced a string the document does not
    actually contain.
    """
    tokens = sentence.split()
    content_positions = [
        i
        for i, tok in enumerate(tokens)
        if re.sub(r"[^A-Za-z]", "", tok).lower() in salience
        and len(re.sub(r"[^A-Za-z]", "", tok)) >= 4
    ]

    strategies: list[str] = []
    if content_positions:
        strategies.append("swap")
    if _NUMBER_RE.search(sentence):
        strategies.append("number")
    if re.search(r"\b(not|never|no longer|cannot)\b", sentence, re.IGNORECASE):
        strategies.append("undo_negation")
    elif re.search(r"\b(is|are|was|were|has|have|can|will|does|do)\b", sentence):
        strategies.append("negate")
    # Word transposition is deliberately NOT a strategy. It produces
    # "errors medication rose 34%", which any reader discards on syntax
    # without recalling a thing. Every mutation here has to leave a
    # sentence that could plausibly have been written.
    interior_positions = [i for i in content_positions if i > 0]

    rng.shuffle(strategies)

    for strategy in strategies:
        mutated: str | None = None

        if strategy == "swap":
            candidates = list(interior_positions or content_positions)
            rng.shuffle(candidates)
            for position in candidates[:6]:
                original = tokens[position]
                bare = re.sub(r"[^A-Za-z]", "", original)
                replacements = _distractor_pool(
                    bare,
                    salience,
                    exclude=set(sentence.lower().split()),
                    wanted=3,
                    strict=True,
                )
                if replacements:
                    choice = rng.choice(replacements)
                    copy = list(tokens)
                    copy[position] = original.replace(bare, choice, 1)
                    mutated = " ".join(copy)
                    break

        elif strategy == "number":
            match = _NUMBER_RE.search(sentence)
            if match:
                alternatives = _numeric_distractors(
                    match.group(1), match.group(2), match.group(3) or "", sentence, wanted=2
                )
                if alternatives:
                    mutated = (
                        sentence[: match.start()]
                        + rng.choice(alternatives)
                        + sentence[match.end() :]
                    )

        elif strategy == "negate":
            mutated = re.sub(
                r"\b(is|are|was|were|has|have|can|will|does|do)\b",
                lambda m: m.group(1) + " not",
                sentence,
                count=1,
            )

        elif strategy == "undo_negation":
            mutated = re.sub(r"\s*\b(not|never)\b", "", sentence, count=1)

        if mutated and mutated != sentence:
            key = re.sub(r"\s+", " ", mutated.strip().lower())
            if key not in forbidden:
                return mutated

    return None


def _build_cloze(
    sentences: list[str], salience: dict[str, float], rng: random.Random
) -> list[dict]:
    questions: list[dict] = []
    used_answers: set[str] = set()

    ranked = []
    for index, sentence in enumerate(sentences):
        words = sentence.split()
        if not (8 <= len(words) <= 34):
            continue
        score = sum(salience.get(re.sub(r"[^A-Za-z]", "", w).lower(), 0.0) for w in words)
        ranked.append((score / max(1, len(words)), index, sentence))
    ranked.sort(key=lambda t: (-t[0], t[1]))

    for _score, index, sentence in ranked:
        tokens = sentence.split()
        best: tuple[float, int, str] | None = None
        lowered_tokens = [re.sub(r"[^A-Za-z]", "", t).lower() for t in tokens]

        for position, bare in enumerate(lowered_tokens):
            if len(bare) < 4 or bare in STOPWORDS or bare in used_answers:
                continue
            # The blank must be unambiguous: if the word repeats in this
            # sentence the reader can read the answer off the other copy.
            if lowered_tokens.count(bare) > 1:
                continue
            score = salience.get(bare, 0.0)
            if score <= 0:
                continue
            if best is None or score > best[0]:
                best = (score, position, tokens[position])

        if best is None:
            continue

        _score2, position, surface = best
        answer_bare = re.sub(r"[^A-Za-z]", "", surface)
        distractors = _distractor_pool(
            answer_bare,
            salience,
            exclude=set(lowered_tokens),
            wanted=3,
        )
        if len(distractors) < 3:
            continue

        blanked = list(tokens)
        blanked[position] = surface.replace(answer_bare, "______", 1)
        used_answers.add(answer_bare.lower())

        questions.append(
            {
                "kind": "cloze",
                "prompt": "Which word filled this gap?",
                "question": "Which word filled this gap?  " + " ".join(blanked),
                "_answer": answer_bare,
                "_distractors": distractors,
                "evidence": sentence,
            }
        )
        if len(questions) >= 4:
            break

    return questions


def _build_ordering(sentences: list[str], rng: random.Random) -> list[dict]:
    if len(sentences) < 6:
        return []

    # Take one sentence from each quarter of the document so the ordering
    # is about the shape of the argument, not about two adjacent lines.
    quarter = len(sentences) / 4.0
    picks: list[tuple[int, str]] = []
    for band in range(4):
        start = int(band * quarter)
        stop = max(start + 1, int((band + 1) * quarter))
        window = [
            (i, sentences[i])
            for i in range(start, min(stop, len(sentences)))
            if 6 <= len(sentences[i].split()) <= 40
        ]
        if not window:
            return []
        picks.append(window[len(window) // 2])

    # Fragments are truncated to the same word count, so no option is
    # identifiable by being conspicuously longer or shorter.
    options = [(i, _fragment(s, 9)) for i, s in picks]
    if len({text for _i, text in options}) < 4:
        return []

    ask_last = rng.random() < 0.5
    target = max(options, key=lambda p: p[0]) if ask_last else min(options, key=lambda p: p[0])
    question = (
        "Which of these came last in the text?"
        if ask_last
        else "Which of these came first in the text?"
    )

    return [
        {
            "kind": "ordering",
            "prompt": question,
            "question": question,
            "_answer": target[1],
            "_distractors": [text for i, text in options if i != target[0]][:3],
            "evidence": sentences[target[0]],
        }
    ]


def _build_numeric(
    sentences: list[str], text: str, rng: random.Random
) -> list[dict]:
    questions: list[dict] = []
    for sentence in sentences:
        if len(questions) >= 3:
            break
        match = _NUMBER_RE.search(sentence)
        if not match:
            continue
        words = sentence.split()
        if not (6 <= len(words) <= 40):
            continue

        prefix, digits, unit = match.group(1), match.group(2), match.group(3) or ""
        answer = match.group(0).strip()
        distractors = _numeric_distractors(prefix, digits, unit, text, wanted=3)
        if len(distractors) < 3:
            continue

        blanked = sentence[: match.start()] + "______" + sentence[match.end() :]
        questions.append(
            {
                "kind": "numeric",
                "prompt": "Which figure did the text give?",
                "question": "Which figure did the text give?  " + blanked.strip(),
                "_answer": answer,
                "_distractors": distractors,
                "evidence": sentence,
            }
        )
    return questions


def _build_attribution(
    sentences: list[str], salience: dict[str, float], rng: random.Random
) -> list[dict]:
    if len(sentences) < 5:
        return []

    forbidden = {re.sub(r"\s+", " ", s.strip().lower()) for s in sentences}
    usable = [s for s in sentences if 7 <= len(s.split()) <= 28]
    if len(usable) < 4:
        return []

    questions: list[dict] = []
    for offset in range(min(3, len(usable))):
        target = usable[(offset * 2 + 1) % len(usable)]
        pool = [s for s in usable if s != target]
        rng.shuffle(pool)

        distractors: list[str] = []
        for source in pool:
            if len(distractors) >= 3:
                break
            # Match the target's length so no option stands out.
            if abs(len(source.split()) - len(target.split())) > 8:
                continue
            mutated = _mutate_sentence(source, salience, rng, forbidden)
            if mutated and mutated not in distractors:
                distractors.append(mutated)

        if len(distractors) < 3:
            continue

        question = "Which of these sentences actually appeared in the text?"
        questions.append(
            {
                "kind": "attribution",
                "prompt": question,
                "question": question,
                "_answer": target,
                "_distractors": distractors,
                "evidence": target,
            }
        )
        break  # one per document: they are long to read and slow to answer

    return questions


def _document_sentences(text: str) -> list[str]:
    """Sentences, split paragraph by paragraph.

    split_sentences() works on punctuation, and a heading or a list item
    usually has none -- so "Night Shift Work and the Circadian Clock" would
    glue itself onto the first real sentence and both end up in the same
    question. Splitting each blank-line-separated block separately keeps
    layout boundaries as sentence boundaries, which is also how
    prepare_words() treats them.
    """
    out: list[str] = []
    for block in re.split(r"\n[ \t\r]*\n", text or ""):
        block = block.strip()
        if not block:
            continue
        pieces = [s.strip() for s in split_sentences(block) if s.strip()]
        if pieces:
            out.extend(pieces)
        else:
            folded = normalize(block)
            if folded:
                out.append(folded)
    return out


def comprehension_questions(
    text: str, *, count: int = 4, seed: int | None = None
) -> list[dict]:
    """Generate multiple-choice comprehension questions from `text`.

    NO AI IS INVOLVED. Four classical techniques, in order of how much
    signal they carry:

    * **cloze** -- blank the highest-TF-IDF content word of a salient
      sentence; distractors are other salient words from the same
      document matched on capitalisation, morphology, length and syllable
      count (see _distractor_pool).
    * **numeric** -- blank a stated figure; distractors are perturbations
      that keep the currency symbol, grouping, decimals and unit.
    * **ordering** -- which of four truncated fragments came first (or
      last); tests whether the reader built a structure or just a bag of
      words.
    * **attribution** -- which sentence really appeared; distractors are
      real sentences from elsewhere in the document with one content
      word, figure or polarity altered.

    Returns a list of::

        {"question": str, "options": [str], "answer_index": int,
         "kind": str, "evidence": str}

    Deterministic: the same `text` and `seed` always produce the same
    quiz, options included. With `seed=None` the seed is derived from the
    text itself, so a document's quiz is stable across requests without
    the caller having to store anything.

    WHAT THIS CANNOT DO, honestly
    -----------------------------
    These questions test *recall of stated surface content*: which word,
    which figure, which order, which sentence. They do not test
    inference, synthesis, or whether you understood why any of it
    matters, because measuring that without a model is beyond classical
    NLP and pretending otherwise would be the same self-deception the
    feature exists to prevent.

    Three more limits worth naming:

    1. Four items is a noisy estimate. A single score of 0.5 is two coin
       flips away from 0.75. That is why adapt_wpm() damps hard and uses
       a dead band -- it is designed for a noisy sensor.
    2. A reader can sometimes reconstruct a cloze answer from syntax and
       topic alone without having read that sentence. The distractor
       filters shrink this but do not eliminate it.
    3. Salience is TF-IDF over sentences, which tracks *distinctiveness*,
       not *importance*. An unusual aside can outrank the thesis.

    So treat the number as a rough, honest, downward-biased indicator of
    attention -- which is still far better than asking someone whether
    they felt like they understood it.
    """
    if not text or not text.strip():
        return []

    count = max(1, int(count))
    if seed is None:
        # A stable, platform-independent seed derived from the content.
        seed = int.from_bytes(normalize(text)[:512].encode("utf-8", "ignore"), "big", signed=False) % (2**31)
    rng = random.Random(seed)

    sentences = _document_sentences(text)
    if len(sentences) < 2:
        return []

    salience = _salience_map(sentences)
    if not salience:
        return []

    by_kind: dict[str, list[dict]] = {
        "cloze": _build_cloze(sentences, salience, rng),
        "numeric": _build_numeric(sentences, text, rng),
        "ordering": _build_ordering(sentences, rng),
        "attribution": _build_attribution(sentences, salience, rng),
    }

    # Interleave kinds so a 4-question quiz is varied rather than four
    # clozes from four adjacent paragraphs.
    order = ["cloze", "numeric", "ordering", "attribution"]
    drafts: list[dict] = []
    round_index = 0
    while len(drafts) < count and any(by_kind[k] for k in order):
        progressed = False
        for kind in order:
            if len(drafts) >= count:
                break
            bucket = by_kind[kind]
            if round_index < len(bucket):
                drafts.append(bucket[round_index])
                progressed = True
        if not progressed:
            break
        round_index += 1

    questions: list[dict] = []
    for draft in drafts[:count]:
        answer = draft["_answer"]
        options = [answer] + [d for d in draft["_distractors"] if d != answer]
        # De-duplicate while preserving order; a repeated option means two
        # correct answers, which silently breaks the score.
        deduped: list[str] = []
        seen_lower: set[str] = set()
        for option in options:
            key = option.strip().lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            deduped.append(option)
        if len(deduped) < 4:
            continue

        deduped = deduped[:4]
        rng.shuffle(deduped)
        questions.append(
            {
                "question": draft["question"],
                "options": deduped,
                "answer_index": deduped.index(answer),
                "kind": draft["kind"],
                "evidence": draft["evidence"],
            }
        )

    return questions


# ---------------------------------------------------------------------------
# Speed adaptation
# ---------------------------------------------------------------------------


def adapt_wpm(
    current_wpm: int,
    comprehension: float,
    *,
    floor: int = WPM_FLOOR,
    ceiling: int = WPM_CEILING,
) -> dict:
    """Move the reading speed based on measured comprehension.

    THE CONTROL LAW: a dead-band proportional controller with asymmetric
    gain, rate limiting and output quantisation. Four ideas, each doing a
    specific job.

    1. DEAD BAND. Comprehension between 0.60 and 0.85 produces no change
       at all. The sensor is a four-item quiz -- one item is worth 0.25 --
       so small score movements are noise, not signal. A controller with
       no dead band chases that noise and the speed hunts up and down
       forever, which feels arbitrary and destroys the reader's trust in
       the number.

    2. PROPORTIONAL RESPONSE. Outside the band, the correction is
       proportional to how far outside::

           error = comprehension - 0.85   (above the band, positive)
           error = comprehension - 0.60   (below the band, negative)
           change = current_wpm * gain * error

       A reader at 0.95 gets a bigger push than one at 0.86; a reader at
       0.2 gets pulled back hard. Scaling by current_wpm keeps the step a
       constant *percentage*, so the loop behaves the same at 200 wpm and
       at 700.

    3. ASYMMETRIC GAIN. Up-gain 0.45, down-gain 1.30 -- backing off is
       nearly three times as aggressive as pushing forward. This is the
       same reasoning as TCP's additive-increase/multiplicative-decrease:
       the two errors are not equally costly. Reading 40 wpm slower than
       you could costs you a few seconds. Reading 100 wpm faster than you
       can costs you the document, plus the false belief that you read it.

    4. RATE LIMIT AND QUANTISATION. Any single change is capped at 12% of
       current speed and 60 wpm absolute, then rounded to the nearest 5,
       and changes under 5 wpm are dropped to zero. The cap stops one
       unlucky quiz from throwing the speed across its whole range; the
       rounding gives the controller a stable resting point instead of an
       endless trickle of one-wpm adjustments.

    Together these guarantee convergence rather than oscillation: each
    step moves toward the band by a bounded amount, and once inside the
    band the controller stops entirely.

    Returns ``{"new_wpm": int, "change": int, "reason": str}``.
    """
    try:
        current = int(current_wpm)
    except (TypeError, ValueError):
        current = DEFAULT_WPM
    floor = int(floor)
    ceiling = int(ceiling)
    if floor > ceiling:
        floor, ceiling = ceiling, floor
    current = max(floor, min(ceiling, current))

    try:
        score = float(comprehension)
    except (TypeError, ValueError):
        return {
            "new_wpm": current,
            "change": 0,
            "reason": "No comprehension score for that session, so the speed stays where it is.",
        }
    score = max(0.0, min(1.0, score))

    upper_band, lower_band = 0.85, 0.60
    up_gain, down_gain = 0.45, 1.30

    if score >= upper_band:
        error = score - upper_band
        raw = current * up_gain * error
        direction = "up"
    elif score <= lower_band:
        error = score - lower_band
        raw = current * down_gain * error
        direction = "down"
    else:
        raw = 0.0
        direction = "hold"

    # Rate limit, then quantise.
    limit = min(60.0, current * 0.12)
    raw = max(-limit, min(limit, raw))
    change = int(round(raw / 5.0)) * 5
    if abs(change) < 5:
        change = 0

    proposed = current + change
    new_wpm = max(floor, min(ceiling, proposed))
    change = new_wpm - current
    pct = int(round(score * 100))

    if change > 0:
        reason = (
            f"You scored {pct}% on the check, which is comfortably above the "
            f"85% mark, so the speed goes up {change} to {new_wpm} wpm."
        )
    elif change < 0:
        reason = (
            f"You scored {pct}%, below the 60% floor where comprehension starts "
            f"to slip, so the speed drops {abs(change)} to {new_wpm} wpm."
        )
    elif direction == "hold":
        reason = (
            f"You scored {pct}%, inside the 60-85% band where the speed is about "
            f"right. Staying at {new_wpm} wpm."
        )
    elif direction == "up" and new_wpm >= ceiling:
        reason = (
            f"You scored {pct}%, but {ceiling} wpm is the ceiling. Past this point "
            "the gains are mostly imagined."
        )
    elif direction == "down" and new_wpm <= floor:
        reason = (
            f"You scored {pct}%, and {floor} wpm is the floor. If this keeps "
            "happening the material is the problem, not the speed."
        )
    else:
        reason = f"The adjustment rounded to nothing. Staying at {new_wpm} wpm."

    return {"new_wpm": int(new_wpm), "change": int(change), "reason": reason}


# ---------------------------------------------------------------------------
# Fitness over time
# ---------------------------------------------------------------------------


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _session_date(session: dict) -> date | None:
    for key in ("created_at", "date", "timestamp", "ended_at", "finished_at"):
        raw = session.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw.date()
        if isinstance(raw, date):
            return raw
        if isinstance(raw, (int, float)):
            try:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc).date()
            except (OSError, OverflowError, ValueError):
                continue
        if isinstance(raw, str):
            candidate = raw.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(candidate).date()
            except ValueError:
                try:
                    return datetime.strptime(raw.strip()[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue
    return None


def _slope(values: list[float]) -> float:
    """Least-squares slope of `values` against their index -- the change in
    effective wpm per session. A regression rather than last-minus-first
    because a single bad session should not flip the verdict."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def reading_fitness(sessions: list[dict]) -> dict:
    """Summarise reading sessions into a picture of whether this is working.

    The headline number is EFFECTIVE WPM: ``wpm * comprehension``. Raw wpm
    is the number people brag about and it is close to meaningless on its
    own -- 900 wpm at 30% comprehension is 270 wpm of reading plus a
    false memory. Multiplying by the measured score gives words per
    minute that actually landed, and it is the only number in the app
    that cannot be gamed by dragging the slider right.

    BEST SUSTAINED WPM is deliberately strict: the highest raw speed
    reached in a session of at least 200 words with comprehension of at
    least 0.70. A 30-word burst at 800 wpm is not a speed you have, it is
    a speed you touched.

    Each session may carry ``wpm``, ``comprehension`` (0-1), ``words_read``,
    ``duration_ms``, ``completed`` and a date under any of ``created_at`` /
    ``date`` / ``timestamp``. Missing fields degrade quietly rather than
    raising -- history is written by other code and will be uneven.
    """
    empty = {
        "session_count": 0,
        "total_words": 0,
        "total_minutes": 0.0,
        "average_wpm": 0,
        "average_comprehension": 0.0,
        "effective_wpm": 0,
        "best_sustained_wpm": 0,
        "streak_days": 0,
        "trend": "none",
        "trend_per_session": 0.0,
        "series": [],
        "assessment": (
            "No reading sessions yet. Start one and this fills in after the "
            "first comprehension check."
        ),
    }

    if not sessions:
        return empty

    cleaned: list[dict] = []
    for session in sessions:
        if not isinstance(session, dict):
            continue
        wpm = _as_float(session.get("wpm"), 0.0)
        if wpm <= 0:
            continue
        comprehension = session.get("comprehension")
        comprehension = None if comprehension is None else max(0.0, min(1.0, _as_float(comprehension)))
        cleaned.append(
            {
                "wpm": wpm,
                "comprehension": comprehension,
                "words_read": int(_as_float(session.get("words_read"), 0.0)),
                "duration_ms": _as_float(session.get("duration_ms"), 0.0),
                "completed": bool(session.get("completed")),
                "day": _session_date(session),
            }
        )

    if not cleaned:
        return empty

    # Oldest first, undated sessions keep their given order at the end.
    dated = [s for s in cleaned if s["day"] is not None]
    undated = [s for s in cleaned if s["day"] is None]
    dated.sort(key=lambda s: s["day"])
    ordered = dated + undated

    total_words = sum(s["words_read"] for s in ordered)
    total_minutes = sum(s["duration_ms"] for s in ordered) / 60000.0

    scored = [s for s in ordered if s["comprehension"] is not None]
    average_comprehension = (
        sum(s["comprehension"] for s in scored) / len(scored) if scored else 0.0
    )
    average_wpm = sum(s["wpm"] for s in ordered) / len(ordered)

    series: list[dict] = []
    for s in scored:
        series.append(
            {
                "date": s["day"].isoformat() if s["day"] else None,
                "wpm": int(round(s["wpm"])),
                "comprehension": round(s["comprehension"], 3),
                "effective_wpm": int(round(s["wpm"] * s["comprehension"])),
                "words_read": s["words_read"],
            }
        )

    effective_values = [float(point["effective_wpm"]) for point in series]
    effective_wpm = (
        int(round(sum(effective_values) / len(effective_values))) if effective_values else 0
    )

    sustained = [
        s["wpm"]
        for s in ordered
        if s["words_read"] >= 200
        and s["comprehension"] is not None
        and s["comprehension"] >= 0.70
    ]
    best_sustained = int(round(max(sustained))) if sustained else 0

    # Streak: consecutive calendar days ending at the most recent session.
    # Anchored on the last session rather than today, so it reports what
    # happened instead of quietly resetting to zero overnight.
    streak = 0
    if dated:
        days = sorted({s["day"] for s in dated}, reverse=True)
        streak = 1
        for previous, current in zip(days, days[1:]):
            if previous - current == timedelta(days=1):
                streak += 1
            else:
                break

    slope = _slope(effective_values) if len(effective_values) >= 4 else 0.0
    if len(effective_values) < 4:
        trend = "too_early"
    elif slope > 2.0:
        trend = "improving"
    elif slope < -2.0:
        trend = "declining"
    else:
        trend = "steady"

    # Plain-English assessment. No praise for numbers that do not deserve
    # it -- the point of the feature is an honest mirror.
    parts: list[str] = []
    if effective_wpm:
        parts.append(
            f"Your effective rate is about {effective_wpm} words a minute "
            f"({int(round(average_wpm))} wpm at {int(round(average_comprehension * 100))}% recall)."
        )
    else:
        parts.append(
            f"You have read at around {int(round(average_wpm))} wpm, but no "
            "comprehension checks yet, so there is no effective rate to report."
        )

    if trend == "improving":
        parts.append(f"It is climbing, roughly {abs(slope):.0f} wpm per session.")
    elif trend == "declining":
        parts.append(
            f"It has been slipping, roughly {abs(slope):.0f} wpm per session. "
            "Denser material or a tired hour will both do that."
        )
    elif trend == "steady":
        parts.append("It has been flat across recent sessions.")
    else:
        parts.append("A few more sessions and a trend becomes readable.")

    if best_sustained:
        parts.append(
            f"Your best sustained speed is {best_sustained} wpm, held over a full "
            "passage with recall intact."
        )
    else:
        parts.append(
            "No sustained speed recorded yet: that needs a session of 200 words "
            "or more with recall at 70% or better."
        )

    if streak >= 2:
        parts.append(f"{streak} days in a row.")

    return {
        "session_count": len(ordered),
        "total_words": total_words,
        "total_minutes": round(total_minutes, 2),
        "average_wpm": int(round(average_wpm)),
        "average_comprehension": round(average_comprehension, 3),
        "effective_wpm": effective_wpm,
        "best_sustained_wpm": best_sustained,
        "streak_days": streak,
        "trend": trend,
        "trend_per_session": round(slope, 2),
        "series": series,
        "assessment": " ".join(parts),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SAMPLE = """Night Shift Work and the Circadian Clock

Researchers at the Karolinska Institute followed 1,482 hospital staff over
six years, tracking sleep, alertness and error rates. The cohort included
nurses, technicians and junior doctors working rotating schedules.

Participants on permanent night rotations lost an average of 47 minutes of
sleep per 24-hour cycle compared with their day-shift colleagues. The
deficit did not recover on days off. Melatonin onset drifted later by
roughly 2.5 hours within the first fortnight of a rotation, and never fully
realigned while the rotation continued.

Error rates tell a blunter story. Documented medication errors rose 34%
among staff in the fourth consecutive night of a rotation. Reaction times
measured at the end of a night shift were comparable to those of a person
with a blood alcohol concentration of 0.05%.

The investigators recommend forward-rotating schedules, in which shifts
advance from morning to evening to night, because the human circadian
system lengthens more easily than it shortens. They also recommend a
minimum of 11 hours between consecutive shifts, and bright-light exposure
at the start of a night rotation rather than the end.

Hospitals that adopted forward rotation reported a measurable drop in
sickness absence within eighteen months. The effect was strongest among
staff over forty, whose circadian flexibility declines with age. Cost
modelling suggested the schedule change paid for itself through reduced
agency cover alone."""


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {label}"
    if detail and not condition:
        line += f"  -> {detail}"
    print(line)
    return condition


def _self_test() -> int:
    results: list[bool] = []
    print("=" * 72)
    print("reading.py self-test")
    print("=" * 72)

    # --- ORP bands --------------------------------------------------------
    print("\n-- ORP --")
    band_cases = [
        ("a", 0),
        ("I", 0),
        ("of", 1),
        ("the", 1),
        ("word", 1),
        ("quick", 1),
        ("reader", 2),
        ("sentence", 2),
        ("attention", 2),
        ("comparable", 3),
        ("measurement", 3),
        ("recommendation", 4),
        ("incomprehensible", 4),
    ]
    ok = True
    for word, expected in band_cases:
        actual = orp_index(word)
        if actual != expected:
            ok = False
            print(f"        {word!r}: expected {expected}, got {actual}")
    results.append(_check("ORP band table (1 / 2-5 / 6-9 / 10-13 / 14+)", ok))

    results.append(
        _check(
            "ORP skips leading punctuation so the pivot letter is stable",
            orp_index('"quickly') == orp_index("quickly") + 1
            and orp_index("(reader)") == orp_index("reader") + 1,
            f'"quickly -> {orp_index(chr(34) + "quickly")}, quickly -> {orp_index("quickly")}',
        )
    )
    results.append(
        _check(
            "ORP never lands on an interior non-letter",
            all(
                _LETTER_RE.match(w[orp_index(w)])
                for w in ["re-entry", "well-known", "state-of-the-art", "co-op", "x-ray"]
            ),
            str([(w, w[orp_index(w)]) for w in ["re-entry", "well-known", "x-ray"]]),
        )
    )
    results.append(
        _check(
            "ORP always inside the token",
            all(0 <= orp_index(w) < len(w) for w in ["a", "to", "---", '"x"', "1,482", "e.g."]),
        )
    )

    # --- prepare_words ----------------------------------------------------
    print("\n-- prepare_words --")
    words = prepare_words(_SAMPLE)
    results.append(_check("produces a word stream", len(words) > 150, f"got {len(words)}"))
    results.append(
        _check(
            "char_offset indexes back into the source text",
            all(
                _SAMPLE[w["char_offset"] : w["char_offset"] + len(w["word"])]
                .translate(_FOLD)
                == w["word"]
                for w in words
            ),
        )
    )
    results.append(
        _check("index is contiguous", [w["index"] for w in words] == list(range(len(words))))
    )

    sentence_ends = [w for w in words if w["is_sentence_end"]]
    results.append(
        _check(
            "sentence ends detected",
            12 <= len(sentence_ends) <= 24,
            f"got {len(sentence_ends)}",
        )
    )
    results.append(
        _check(
            "sentence-end pause actually appears (>= 2.0x)",
            all(w["delay_multiplier"] >= 1.99 for w in sentence_ends),
            str([(w["word"], w["delay_multiplier"]) for w in sentence_ends if w["delay_multiplier"] < 1.99][:3]),
        )
    )
    para_ends = [w for w in words if w["is_paragraph_end"]]
    results.append(
        _check(
            "paragraph-end pause is larger still (>= 2.5x)",
            len(para_ends) >= 5 and all(w["delay_multiplier"] >= 2.49 for w in para_ends),
            f"{len(para_ends)} paragraph ends",
        )
    )

    comma_words = [
        w
        for w in words
        if w["word"].rstrip(_CLOSERS).endswith(",") and not w["is_sentence_end"]
    ]
    results.append(
        _check(
            "comma pause is present and smaller than a full stop",
            bool(comma_words)
            and all(1.45 <= w["delay_multiplier"] < 2.0 for w in comma_words),
            str([(w["word"], w["delay_multiplier"]) for w in comma_words[:4]]),
        )
    )

    abbrev = prepare_words("Dr. Smith met Prof. Lee at 3.5 percent. Then he left.")
    results.append(
        _check(
            "abbreviations and decimals are not sentence ends",
            [w["word"] for w in abbrev if w["is_sentence_end"]] == ["percent.", "left."],
            str([w["word"] for w in abbrev if w["is_sentence_end"]]),
        )
    )

    numeral = prepare_words("The figure 1,482 appeared.")[2]
    plain = prepare_words("The figure results appeared.")[2]
    results.append(
        _check(
            "numerals get extra time over a same-length word",
            numeral["delay_multiplier"] > plain["delay_multiplier"],
            f'{numeral["word"]}={numeral["delay_multiplier"]} vs {plain["word"]}={plain["delay_multiplier"]}',
        )
    )
    results.append(_check("empty text gives an empty stream", prepare_words("") == []))

    # --- pacing -----------------------------------------------------------
    print("\n-- pacing_plan --")
    pacing_ok = True
    detail = ""
    for wpm in (150, 250, 300, 450, 600, 900):
        plan = pacing_plan(words, wpm)
        expected = len(words) * 60000.0 / wpm
        drift = abs(plan["total_ms"] - expected) / expected
        if drift > 0.01:
            pacing_ok = False
            detail = f"{wpm} wpm drifted {drift:.4%}"
        if len(plan["per_word_ms"]) != len(words):
            pacing_ok = False
            detail = "per_word_ms length mismatch"
    results.append(_check("total time matches the requested wpm within 1%", pacing_ok, detail))

    plan300 = pacing_plan(words, 300)
    end_slots = [plan300["per_word_ms"][w["index"]] for w in sentence_ends]
    mid_slots = [
        plan300["per_word_ms"][w["index"]] for w in words if not w["is_sentence_end"]
    ]
    results.append(
        _check(
            "sentence ends really get longer slots in milliseconds",
            min(end_slots) > sum(mid_slots) / len(mid_slots),
            f"min end slot {min(end_slots)}ms vs mean mid slot {sum(mid_slots) / len(mid_slots):.0f}ms",
        )
    )
    results.append(
        _check(
            "no slot is below the two-frame floor",
            min(plan300["per_word_ms"]) >= _MIN_FRAME_MS,
        )
    )
    results.append(
        _check(
            "estimated_minutes is consistent with total_ms",
            abs(plan300["estimated_minutes"] - plan300["total_ms"] / 60000.0) < 0.01,
        )
    )
    results.append(
        _check("empty word list is handled", pacing_plan([], 300)["total_ms"] == 0)
    )

    # --- comprehension questions -----------------------------------------
    print("\n-- comprehension_questions --")
    quiz = comprehension_questions(_SAMPLE, count=4, seed=7)
    results.append(_check("four questions generated for real text", len(quiz) == 4, f"got {len(quiz)}"))
    results.append(
        _check(
            "more than one kind of question",
            len({q["kind"] for q in quiz}) >= 3,
            str([q["kind"] for q in quiz]),
        )
    )
    results.append(
        _check(
            "every question has exactly four options",
            all(len(q["options"]) == 4 for q in quiz),
        )
    )
    results.append(
        _check(
            "options are never duplicated (exactly one correct answer)",
            all(len({o.strip().lower() for o in q["options"]}) == 4 for q in quiz),
            str([q["options"] for q in quiz if len({o.strip().lower() for o in q["options"]}) != 4]),
        )
    )
    results.append(
        _check(
            "answer_index is in range and selects the intended answer",
            all(0 <= q["answer_index"] < 4 for q in quiz),
        )
    )

    haystack = re.sub(r"\s+", " ", normalize(_SAMPLE)).lower()
    answer_ok = True
    answer_detail = ""
    for q in quiz:
        answer = q["options"][q["answer_index"]].strip().rstrip("…").strip()
        needle = re.sub(r"\s+", " ", normalize(answer)).lower().rstrip(".")
        if needle not in haystack:
            answer_ok = False
            answer_detail = f'{q["kind"]}: {answer!r} not found in text'
    results.append(_check("the correct answer is genuinely in the text", answer_ok, answer_detail))

    distractor_ok = True
    distractor_detail = ""
    for q in quiz:
        answer = q["options"][q["answer_index"]]
        for i, option in enumerate(q["options"]):
            if i == q["answer_index"]:
                continue
            if option.strip().lower() == answer.strip().lower():
                distractor_ok = False
                distractor_detail = f"duplicate of answer in {q['kind']}"
            if q["kind"] == "attribution":
                needle = re.sub(r"\s+", " ", normalize(option)).lower().rstrip(".")
                if needle in haystack:
                    distractor_ok = False
                    distractor_detail = f"attribution distractor is a real sentence: {option!r}"
            if q["kind"] == "numeric" and option.strip() == answer.strip():
                distractor_ok = False
                distractor_detail = "numeric distractor equals the answer"
    results.append(
        _check(
            "distractors are never duplicates of the answer, and attribution lures are not real sentences",
            distractor_ok,
            distractor_detail,
        )
    )

    cloze = [q for q in quiz if q["kind"] == "cloze"]
    shape_ok = True
    shape_detail = ""
    for q in cloze:
        lengths = [len(o) for o in q["options"]]
        if max(lengths) - min(lengths) > 6:
            shape_ok = False
            shape_detail = f"cloze options vary too much in length: {q['options']}"
        caps = {_cap_class(o) for o in q["options"]}
        if len(caps) > 1:
            shape_ok = False
            shape_detail = f"cloze options mix capitalisation: {q['options']}"
    results.append(
        _check("cloze distractors match the answer's shape (no free giveaways)", shape_ok, shape_detail)
    )

    results.append(
        _check(
            "same seed reproduces the same quiz exactly",
            comprehension_questions(_SAMPLE, count=4, seed=7) == quiz,
        )
    )
    results.append(
        _check(
            "a different seed gives a different quiz",
            comprehension_questions(_SAMPLE, count=4, seed=99) != quiz,
        )
    )
    results.append(
        _check(
            "seed=None is still deterministic for the same text",
            comprehension_questions(_SAMPLE, count=3)
            == comprehension_questions(_SAMPLE, count=3),
        )
    )
    results.append(
        _check(
            "short or empty text degrades quietly",
            comprehension_questions("") == [] and comprehension_questions("Hi.") == [],
        )
    )
    results.append(
        _check(
            "count is respected",
            len(comprehension_questions(_SAMPLE, count=2, seed=3)) == 2,
        )
    )

    # --- adapt_wpm --------------------------------------------------------
    print("\n-- adapt_wpm --")
    high = adapt_wpm(300, 0.95)
    hold = adapt_wpm(300, 0.72)
    low = adapt_wpm(300, 0.35)
    results.append(_check("high comprehension raises the speed", high["change"] > 0, str(high)))
    results.append(_check("mid-band comprehension holds", hold["change"] == 0, str(hold)))
    results.append(_check("low comprehension lowers the speed", low["change"] < 0, str(low)))
    results.append(
        _check(
            "the drop is more aggressive than the rise (asymmetric gain)",
            abs(low["change"]) > abs(high["change"]),
            f'up {high["change"]}, down {low["change"]}',
        )
    )
    results.append(
        _check(
            "ceiling respected",
            adapt_wpm(890, 1.0, ceiling=900)["new_wpm"] == 900
            and adapt_wpm(900, 1.0, ceiling=900)["change"] == 0,
        )
    )
    results.append(
        _check(
            "floor respected",
            adapt_wpm(160, 0.0, floor=150)["new_wpm"] == 150
            and adapt_wpm(150, 0.0, floor=150)["change"] == 0,
        )
    )
    results.append(
        _check(
            "single step is rate limited to 12% / 60 wpm",
            all(
                abs(adapt_wpm(w, c)["change"]) <= max(5, min(60, int(w * 0.12)) + 5)
                for w in (150, 300, 600, 900)
                for c in (0.0, 0.2, 0.5, 0.9, 1.0)
            ),
        )
    )
    results.append(
        _check(
            "a missing score changes nothing",
            adapt_wpm(400, None)["change"] == 0,  # type: ignore[arg-type]
        )
    )

    # Closed-loop simulation: comprehension falls as speed rises, so there
    # is a true equilibrium the controller has to find and then sit at.
    def simulated_comprehension(wpm: int) -> float:
        return max(0.0, min(1.0, 1.25 - wpm / 800.0))

    wpm = 700
    trace = [wpm]
    changes: list[int] = []
    for _ in range(25):
        step = adapt_wpm(wpm, simulated_comprehension(wpm))
        changes.append(step["change"])
        wpm = step["new_wpm"]
        trace.append(wpm)

    sign_flips = 0
    last_sign = 0
    for change in changes:
        sign = (change > 0) - (change < 0)
        if sign and last_sign and sign != last_sign:
            sign_flips += 1
        if sign:
            last_sign = sign
    results.append(
        _check(
            "closed loop converges without oscillating",
            sign_flips == 0 and changes[-1] == 0,
            f"flips={sign_flips}, trace={trace}",
        )
    )
    results.append(
        _check(
            "it settles inside the comprehension band",
            0.60 <= simulated_comprehension(trace[-1]) <= 0.85,
            f"settled at {trace[-1]} wpm -> {simulated_comprehension(trace[-1]):.2f}",
        )
    )

    # And from below: a slow reader with perfect recall should climb.
    wpm = 200
    up_changes = []
    for _ in range(25):
        step = adapt_wpm(wpm, simulated_comprehension(wpm))
        up_changes.append(step["change"])
        wpm = step["new_wpm"]
    results.append(
        _check(
            "it also climbs from below and stops, without overshoot flapping",
            wpm > 200
            and up_changes[-1] == 0
            and sum(1 for c in up_changes if c < 0) == 0,
            f"ended at {wpm}, changes={up_changes}",
        )
    )

    # --- reading_fitness --------------------------------------------------
    print("\n-- reading_fitness --")
    zero = reading_fitness([])
    results.append(
        _check(
            "zero sessions handled",
            zero["session_count"] == 0
            and zero["effective_wpm"] == 0
            and zero["streak_days"] == 0
            and isinstance(zero["assessment"], str)
            and zero["series"] == [],
        )
    )
    results.append(
        _check("garbage sessions handled", reading_fitness([{}, {"wpm": 0}])["session_count"] == 0)
    )

    today = date(2026, 9, 16)
    history = []
    for i in range(8):
        day = today - timedelta(days=7 - i)
        history.append(
            {
                "wpm": 260 + i * 25,
                "comprehension": 0.80 + i * 0.01,
                "words_read": 400 + i * 30,
                "duration_ms": 90000,
                "completed": True,
                "created_at": day.isoformat(),
            }
        )
    fitness = reading_fitness(history)
    results.append(_check("counts every session", fitness["session_count"] == 8))
    results.append(
        _check("total words summed", fitness["total_words"] == sum(h["words_read"] for h in history))
    )
    results.append(
        _check(
            "effective wpm is wpm x comprehension",
            fitness["series"][0]["effective_wpm"] == int(round(260 * 0.80)),
            str(fitness["series"][0]),
        )
    )
    results.append(
        _check("rising history reads as improving", fitness["trend"] == "improving", fitness["trend"])
    )
    results.append(
        _check(
            "best sustained wpm is the strict one",
            fitness["best_sustained_wpm"] == 260 + 7 * 25,
            str(fitness["best_sustained_wpm"]),
        )
    )
    results.append(_check("consecutive-day streak counted", fitness["streak_days"] == 8, str(fitness["streak_days"])))
    results.append(
        _check(
            "assessment is plain English and non-empty",
            len(fitness["assessment"]) > 40 and "!" not in fitness["assessment"],
        )
    )

    # Three consecutive days, then a month-old session: the streak counts
    # the run ending at the most recent session and stops at the gap.
    broken_streak = history[:3] + [
        dict(history[-1], created_at=(today - timedelta(days=30)).isoformat())
    ]
    results.append(
        _check(
            "a gap breaks the streak",
            reading_fitness(broken_streak)["streak_days"] == 3,
            str(reading_fitness(broken_streak)["streak_days"]),
        )
    )
    isolated = [
        {"wpm": 300, "comprehension": 0.8, "words_read": 300, "created_at": today.isoformat()},
        {"wpm": 300, "comprehension": 0.8, "words_read": 300,
         "created_at": (today - timedelta(days=10)).isoformat()},
        {"wpm": 300, "comprehension": 0.8, "words_read": 300,
         "created_at": (today - timedelta(days=11)).isoformat()},
    ]
    results.append(
        _check(
            "an isolated latest session is a streak of one",
            reading_fitness(isolated)["streak_days"] == 1,
            str(reading_fitness(isolated)["streak_days"]),
        )
    )
    results.append(
        _check(
            "short sessions do not count as sustained",
            reading_fitness(
                [{"wpm": 900, "comprehension": 0.9, "words_read": 30, "created_at": "2026-09-01"}]
            )["best_sustained_wpm"]
            == 0,
        )
    )
    results.append(
        _check(
            "sessions without a comprehension score still count words",
            reading_fitness([{"wpm": 300, "words_read": 500, "created_at": "2026-09-01"}])[
                "total_words"
            ]
            == 500,
        )
    )

    # --- summary ----------------------------------------------------------
    print("\n" + "=" * 72)
    passed = sum(1 for r in results if r)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)

    if passed == len(results):
        print("\nSample quiz (seed=7):\n")
        for i, q in enumerate(quiz, 1):
            print(f"{i}. [{q['kind']}] {q['question']}")
            for j, option in enumerate(q["options"]):
                mark = "*" if j == q["answer_index"] else " "
                print(f"   {mark} {chr(97 + j)}) {option}")
            print()

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
