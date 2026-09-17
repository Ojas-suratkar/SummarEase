"""
text_kit -- the shared language primitives every feature in app/core/
is built on.

Nothing in this package calls an AI API. Not Gemini, not OpenAI, not a
hosted embedding endpoint, not anything over the network. Every function
here is a classical algorithm implemented directly: tokenization, the
Porter stemming algorithm (Porter, 1980), sentence boundary detection,
and syllable estimation. They run in microseconds, offline, for free,
deterministically -- the same input always gives the same output, which
is what makes the features above them (search ranking, deduplication,
readability scoring) something you can actually trust and test.

That matters for a reason beyond cost: a summary from a language model
is a black box you have to take on faith, but a BM25 score or a
Flesch-Kincaid grade is a number you can recompute by hand and check.
The features built on this file are the part of the app that is *ours*.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Stopwords -- high-frequency function words that carry grammatical rather
# than topical meaning. Removing them before indexing/scoring keeps "the"
# from dominating every similarity calculation.
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves will just
also may might must shall upon said says say get got make made even much many
""".split())

# Words that end a sentence-like abbreviation rather than a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "fig", "al", "inc", "ltd", "co", "corp", "dept", "est", "approx", "no",
    "vol", "ed", "pp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "us", "uk", "eu", "un",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold to a canonical form: NFKD unicode normalization (so accented
    characters and their decomposed equivalents compare equal), curly
    quotes flattened, whitespace collapsed."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace(" ", " ")
    )
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps internal apostrophes ("don't" stays
    one token) and alphanumerics ("covid19", "3d"), drops everything else."""
    if not text:
        return []
    return _WORD_RE.findall(normalize(text).lower())


def content_tokens(text: str, *, stem: bool = True, min_length: int = 2) -> list[str]:
    """Tokens with stopwords and very short tokens removed, optionally
    stemmed. This is the standard input to indexing and similarity."""
    out = []
    for token in tokenize(text):
        if len(token) < min_length or token in STOPWORDS:
            continue
        out.append(porter_stem(token) if stem else token)
    return out


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Contiguous n-grams. Used for shingling (fingerprint.py) and phrase
    matching (lexicon.py)."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Sentence splitting
#
# A regex on /[.!?]/ alone splits "Dr. Smith" and "3.5 million" into
# fragments, which then corrupts every sentence-level score built on top
# (TextRank centrality, average sentence length, readability grade). This
# handles the common false-positive cases: known abbreviations, initials,
# decimals, and ellipses.
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"([.!?]+)(\s+|$)")


def split_sentences(text: str) -> list[str]:
    """Split into sentences, guarding against abbreviations, initials and
    decimal points."""
    text = normalize(text)
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENT_END_RE.finditer(text):
        end = match.end(1)
        candidate = text[start:end].strip()
        if not candidate:
            continue

        # Look at the token immediately before the punctuation.
        before = text[:match.start(1)]
        last_word = re.split(r"[\s(\[]", before)[-1].strip().lower().rstrip(".")

        # "Dr." / "e.g." -- an abbreviation, not a sentence end.
        if last_word in _ABBREVIATIONS:
            continue
        # "J. R. R. Tolkien" -- a single initial.
        if len(last_word) == 1 and last_word.isalpha():
            continue
        # "3.5" -- a decimal, only when a digit follows.
        if last_word and last_word[-1].isdigit() and match.end() < len(text):
            following = text[match.end():match.end() + 1]
            if following.isdigit():
                continue

        sentences.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Syllable estimation
#
# Every readability formula (Flesch, Flesch-Kincaid, SMOG, Gunning Fog)
# needs a syllable count, and English has no closed-form rule for it.
# This is the standard vowel-group heuristic with the usual corrections:
# silent terminal 'e', 'le' endings, consecutive-vowel collapsing, and a
# small exception table. It agrees with a pronunciation dictionary on
# roughly 90% of common English words -- good enough for grade-level
# scoring, which is itself only accurate to about a grade.
# ---------------------------------------------------------------------------

_VOWELS = "aeiouy"

_SYLLABLE_EXCEPTIONS = {
    "the": 1, "he": 1, "she": 1, "we": 1, "be": 1, "me": 1, "business": 2,
    "wednesday": 2, "people": 2, "every": 3, "everything": 3, "different": 3,
    "beautiful": 3, "science": 2, "area": 3, "idea": 3, "real": 1, "create": 2,
    "being": 2, "doing": 2, "going": 2, "simile": 3, "recipe": 3, "coyote": 3,
}


def count_syllables(word: str) -> int:
    """Estimated syllable count for a single word. Always >= 1 for a
    non-empty word."""
    word = re.sub(r"[^a-z]", "", (word or "").lower())
    if not word:
        return 0
    if word in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[word]
    if len(word) <= 3:
        return 1

    # Count vowel groups: consecutive vowels are one nucleus ("beat" = 1).
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in _VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel

    # Silent terminal 'e' ("make" = 1, not 2) -- but "-le" after a
    # consonant is its own syllable ("table" = 2).
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")):
        count -= 1
    elif word.endswith("le") and len(word) > 2 and word[-3] not in _VOWELS:
        pass  # "table", "little" -- the 'le' counts, already counted above
    if word.endswith("ed") and len(word) > 3 and word[-3] not in "td":
        count -= 1  # "walked" = 1, but "wanted"/"landed" = 2

    return max(1, count)


def syllables_in(text: str) -> int:
    return sum(count_syllables(w) for w in tokenize(text))


# ---------------------------------------------------------------------------
# The Porter stemming algorithm (Porter, 1980)
#
# Reduces inflected forms to a common stem so "connect", "connected",
# "connecting" and "connection" all index as one term. Implemented in
# full here rather than pulled from nltk: it's ~100 lines of well-defined
# rules, and vendoring it keeps app/core/ dependency-free and portable.
# ---------------------------------------------------------------------------


def _is_consonant(word: str, i: int) -> bool:
    char = word[i]
    if char in "aeiou":
        return False
    if char == "y":
        # 'y' is a consonant only when preceded by a vowel ("toy") --
        # at the start of a word or after a consonant it's a vowel ("sky").
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's `m`: the number of vowel-consonant sequences in the stem.
    Rules fire only above a measure threshold, which is what stops
    "tree" -> "tr"."""
    count = 0
    i = 0
    n = len(stem)
    # skip initial consonants
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        count += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return count


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem: str) -> bool:
    """consonant-vowel-consonant where the final consonant isn't w, x or y
    -- the condition for restoring a silent 'e' ("hope" -> "hop" + e)."""
    if len(stem) < 3:
        return False
    return (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
        and stem[-1] not in "wxy"
    )


_STEP2_SUFFIXES = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3_SUFFIXES = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4_SUFFIXES = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]

_stem_cache: dict[str, str] = {}


def porter_stem(word: str) -> str:
    """Porter stem of a single lowercase word. Cached -- the same handful
    of words recur constantly across a corpus."""
    if not word:
        return ""
    cached = _stem_cache.get(word)
    if cached is not None:
        return cached

    original = word
    if len(word) <= 2:
        _stem_cache[original] = word
        return word

    # --- Step 1a: plurals ---
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # --- Step 1b: past tense / gerunds ---
    step1b_applied = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _contains_vowel(word[:-2]):
        word = word[:-2]
        step1b_applied = True
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        word = word[:-3]
        step1b_applied = True

    if step1b_applied:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # --- Step 1c: terminal y -> i ---
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # --- Step 2 & 3: derivational suffixes ---
    for suffix, replacement in _STEP2_SUFFIXES:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    for suffix, replacement in _STEP3_SUFFIXES:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    # --- Step 4: strip suffixes from longer stems ---
    for suffix in _STEP4_SUFFIXES:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1:
                if suffix == "ion" and not (stem and stem[-1] in "st"):
                    break
                word = stem
            break

    # --- Step 5: tidy up terminal e and doubled l ---
    if word.endswith("e"):
        m = _measure(word[:-1])
        if m > 1 or (m == 1 and not _ends_cvc(word[:-1])):
            word = word[:-1]
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    _stem_cache[original] = word
    return word


# ---------------------------------------------------------------------------
# Small conveniences used across several modules
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    return len(tokenize(text))


def unique_ratio(text: str) -> float:
    """Type-token ratio: distinct words / total words. A crude but real
    measure of lexical variety."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def levenshtein(a: str, b: str, *, max_distance: int = 3) -> int:
    """Edit distance with early exit -- used for fuzzy search matching.
    Returns max_distance + 1 rather than the true distance once it's
    clear the strings are further apart than we care about."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]
