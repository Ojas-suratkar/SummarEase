"""
temporal -- a grammar-based temporal expression parser and timeline builder.

WHY THIS EXISTS (and why there is no model call in it)
------------------------------------------------------
The timeline feature turns a user's reading history into a chronology: every
date mentioned in every document, placed on an axis. An LLM could do that, but
it would be slow, metered, non-deterministic and impossible to unit-test. Dates
are one of the few parts of natural language with a genuinely small, closed
grammar -- there are maybe forty ways an English writer spells a date and all of
them are regular. So this is a real parser: a table of rules, each a compiled
regex plus a resolver, run over the text, with overlaps settled by a
longest-match sweep. It is deterministic, offline, free, and every decision it
makes is a line of code you can point at in a bug report.

THE OUTPUT CONTRACT
-------------------
Every expression resolves to a *closed interval* [start_date, end_date] plus a
`precision` naming the calendar unit the writer actually used. The interval is
the honest answer to "what instants could this mean?", and `precision` is the
honest answer to "how specific was the writer?". They are separate on purpose:
"12/03/2022" has day precision (the writer named one day) but a nine-month
interval (we cannot tell which day they meant). A caller that wants only
pinpoint dates filters on `start_date == end_date`; a caller that wants to
render a bar on a timeline uses the interval; a caller that wants to label the
bar uses `precision`.

AMBIGUITY DECISIONS -- all of them, in one place
------------------------------------------------
Every one of these is a judgement call. They are listed here so a reviewer can
disagree with them in one place rather than hunting through the rules.

1.  DD/MM vs MM/DD ("12/03/2022"). If exactly one reading is a real calendar
    date (25/12/2022 -- there is no month 25) we take it, confidence 0.85. If
    BOTH readings are valid we prefer MM/DD, because this product's corpus is
    overwhelmingly US-English prose -- but we lower confidence to 0.55 AND
    widen the interval to span both readings (2022-03-12 .. 2022-12-03). The
    widening is the important half: a downstream timeline then visibly shows
    the uncertainty instead of asserting a date we guessed. When day == month
    ("3/3/2024") the two readings coincide and confidence goes back up to 0.9.
2.  Dotted dates ("03.03.2024") are read as DD.MM.YYYY, the opposite default,
    because the dotted form is a European convention and almost never written
    by a US author. Same widening rule when both readings are valid.
3.  Two-digit years ("Mar '24") use a fixed pivot at 50: 00-49 -> 2000s,
    50-99 -> 1900s. A sliding pivot relative to today would make the parser
    non-deterministic across time, which is exactly the property we are trying
    to keep.
4.  Bare years ("in 1997") are the biggest false-positive risk in the whole
    file -- any four-digit number looks like a year. We require BOTH a plausible
    range (1000-2999) AND an immediately preceding temporal cue word
    ("in/since/by/during/until/circa/of/..."). "page 1997", "Room 2024" and
    "4,000 users" therefore do not match, while "in 1997" does. The reported
    span covers only the year itself, not the cue, so `text` reads "1997".
5.  Seasons are NORTHERN HEMISPHERE. "summer 1969" is June-August. There is no
    way to disambiguate from the text alone, and the corpus skews northern.
    Southern-hemisphere documents will be wrong by six months; that is a known,
    accepted limitation rather than an oversight. Winter is treated as spanning
    the year boundary: "winter 1969" is 1969-12-01 .. 1970-02-28.
6.  Quarters are CALENDAR quarters (Q1 = Jan-Mar), not fiscal. Fiscal years
    differ per company and are unknowable from prose.
7.  Centuries use the formal definition: the 20th century is 1901-2000, not
    1900-1999. Colloquial usage often means the latter; we take the strict one
    because it is the defensible one and it is documented here.
8.  Bare two-digit decades ("the 20s") pick the most recent decade that is not
    in the future relative to the reference date -- so "the 90s" is the 1990s
    and "the 20s" is the 2020s. Requires a preceding "the" or an apostrophe so
    that "in his 20s" (an age) does not match.
9.  Vague expressions ("recently", "soon", "currently") ARE emitted, but with
    confidence 0.20 and a deliberately huge interval (+-90 days). Dropping them
    loses real signal; trusting them is wrong. Callers that want only firm
    dates filter on confidence >= 0.4, which is the documented cut-off.
10. "three weeks ago" resolves to one exact day even though a human means it
    loosely. An exact day is reproducible and auditable; a fuzzed interval
    would just be a second guess layered on the first.
11. Week-long expressions ("last week") report precision "day" because the
    public precision vocabulary has no "week" value; the seven-day interval
    carries the real information.
12. Overlapping matches are resolved longest-first, so "March 2024" beats the
    "March" inside it and "from March to June 2024" beats both of its operands.
    Ties break on higher confidence, then on earlier position.

Stdlib only -- re, datetime, calendar -- plus text_kit, which is itself
dependency-free.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

try:  # package import (app.core.temporal)
    from .text_kit import normalize, split_sentences
except ImportError:  # direct execution (python temporal.py)
    from text_kit import normalize, split_sentences


__all__ = ["extract_dates", "extract_events", "build_timeline"]


# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

_MONTH_SEQUENCE = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

_MONTH_NUM: dict[str, int] = {}
for _i, _full in enumerate(_MONTH_SEQUENCE, start=1):
    _MONTH_NUM[_full] = _i
    _MONTH_NUM[_full[:3]] = _i
_MONTH_NUM["sept"] = 9

_WEEKDAY_NUM = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
    "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}

# Northern hemisphere, meteorological boundaries (whole months). See decision 5.
_SEASON_MONTHS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),  # wraps into the following year
}

_ORDINAL_NUM = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
}

# "a few" ~ 3, "several" ~ 4, "a couple" ~ 2. Arbitrary but conventional, and
# these all carry reduced confidence anyway.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
    "a couple of": 2, "a couple": 2, "a few": 3, "several": 4,
}

# Words that, sitting immediately before a number, mean the number is an
# identifier rather than a year. Guards the bare-year and dash-range rules.
_NON_TEMPORAL_PREFIXES = frozenset({
    "page", "pages", "pp", "p", "no", "nos", "number", "num", "section",
    "chapter", "figure", "fig", "table", "line", "lines", "row", "rows",
    "room", "suite", "apt", "unit", "ext", "isbn", "issn", "vol", "volume",
    "issue", "id", "code", "sku", "ref", "version", "v", "step", "item",
})

# Nouns that follow a count, not a year: "in 4000 users" is not a date.
_COUNT_NOUNS = (
    r"users?|people|persons?|customers?|clients?|items?|words?|units?|copies|"
    r"employees?|subscribers?|documents?|files?|rows?|records?|miles?|"
    r"kilometers?|km|feet|dollars?|euros?|pounds?|hours?|minutes?|seconds?"
)

# Cue words that license a bare four-digit year. See decision 4.
_YEAR_CUES = (
    r"in|since|by|during|from|until|till|through|throughout|before|after|"
    r"around|about|circa|ca|c|of|between|and|to|post|pre|early|late|"
    r"year|back in|as of|as early as|as late as|dated|est"
)


# ---------------------------------------------------------------------------
# Pattern fragments. Rules are written with <PLACEHOLDER> tokens and expanded by
# _expand() -- much easier to read (and to get right) than nesting f-string
# braces inside regex quantifiers.
# ---------------------------------------------------------------------------

_FRAGMENTS: dict[str, str] = {}
_FRAGMENTS["<M>"] = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sept(?:ember)?|Sep|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\.?"
)
_FRAGMENTS["<S>"] = r"(?:spring|summer|autumn|fall|winter)"
_FRAGMENTS["<WD>"] = (
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tues|tue|wed|thurs|thur|thu|fri|sat|sun)"
)
_FRAGMENTS["<ORD>"] = r"(?:first|second|third|fourth|1st|2nd|3rd|4th)"
_FRAGMENTS["<MOD>"] = r"(?:early|mid|middle|late)"
_FRAGMENTS["<NUM>"] = (
    r"(?:\d{1,4}|a couple of|a couple|a few|several|an|a|one|two|three|four|"
    r"five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)"
)
_FRAGMENTS["<DASH>"] = r"[-‐‑‒–—―]"
# The same characters WITHOUT the surrounding brackets, for splicing INTO a
# character class. Using <DASH> there would nest brackets and silently build
# a nonsense range instead of a dash class.
_FRAGMENTS["<DASHCH>"] = r"\-‐‑‒–—―"
_FRAGMENTS["<APOS>"] = r"['‘’ʼ]"
_FRAGMENTS["<CUE>"] = "(?:" + _YEAR_CUES + ")"
_FRAGMENTS["<COUNT>"] = "(?:" + _COUNT_NOUNS + ")"


def _expand(pattern: str) -> str:
    """Substitute <PLACEHOLDER> fragments, repeatedly, so fragments may nest."""
    for _ in range(4):
        before = pattern
        for key, value in _FRAGMENTS.items():
            pattern = pattern.replace(key, value)
        if pattern == before:
            break
    return pattern


# Range operands. Written after the base fragments so they can reuse them.
# Order matters inside the alternation: the most specific spelling first, so
# "June 2024" is consumed whole rather than as a bare "June".
_FRAGMENTS["<OP>"] = _expand(
    r"(?:<M>\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|<M>\s+(?:of\s+)?\d{4}"
    r"|<M>\s*<APOS>\d{2}"
    r"|Q[1-4]\s+\d{4}"
    r"|Q[1-4]"
    r"|<M>"
    r"|\d{4})"
)
# Operands for the bare-dash form ("1997-2003", "March-June 2024"). Deliberately
# narrower than <OP>: a hyphen between two arbitrary things is far too common in
# prose to trust.
_FRAGMENTS["<OP2>"] = _expand(r"(?:<M>\s+\d{4}|<M>|\d{4})")


def _P(pattern: str) -> re.Pattern[str]:
    return re.compile(_expand(pattern), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

_PRECISION_RANK = {
    "day": 0, "month": 1, "quarter": 2, "season": 2,
    "year": 3, "decade": 4, "century": 5,
}


def _safe_date(year: int, month: int, day: int) -> date | None:
    """date() that returns None instead of raising. Every rule funnels through
    this, so a malformed expression ("2024-02-31") is simply not a match rather
    than a 500 on a document upload."""
    if not (1 <= month <= 12) or year < 1 or year > 9999:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _month_bounds(year: int, month: int) -> tuple[date, date] | None:
    """First and last day of a calendar month."""
    if not (1 <= month <= 12) or not (1 <= year <= 9999):
        return None
    last = calendar.monthrange(year, month)[1]
    start = _safe_date(year, month, 1)
    end = _safe_date(year, month, last)
    if start is None or end is None:
        return None
    return start, end


def _year_bounds(year: int) -> tuple[date, date] | None:
    start = _safe_date(year, 1, 1)
    end = _safe_date(year, 12, 31)
    if start is None or end is None:
        return None
    return start, end


def _add_months(anchor: date, months: int) -> date:
    """Calendar-aware month arithmetic, clamping the day to the target month's
    length so "one month after 31 January" is 28/29 February rather than an
    exception."""
    index = anchor.month - 1 + months
    year = anchor.year + index // 12
    month = index % 12 + 1
    year = max(1, min(9999, year))
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month_num(token: str) -> int | None:
    key = token.strip().rstrip(".").lower()
    return _MONTH_NUM.get(key) or _MONTH_NUM.get(key[:3])


def _expand_two_digit_year(value: int) -> int:
    """'24 -> 2024, '97 -> 1997. Fixed pivot at 50 (decision 3)."""
    return 2000 + value if value <= 49 else 1900 + value


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date] | None:
    if not (1 <= quarter <= 4):
        return None
    first_month = (quarter - 1) * 3 + 1
    start = _safe_date(year, first_month, 1)
    last_month = first_month + 2
    bounds = _month_bounds(year, last_month)
    if start is None or bounds is None:
        return None
    return start, bounds[1]


def _season_bounds(year: int, season: str) -> tuple[date, date] | None:
    """Northern hemisphere (decision 5). Winter spills into the next year."""
    months = _SEASON_MONTHS.get(season.lower())
    if months is None:
        return None
    first, last = months
    if first <= last:
        start = _safe_date(year, first, 1)
        end_bounds = _month_bounds(year, last)
    else:  # winter: Dec of `year` through Feb of `year + 1`
        start = _safe_date(year, first, 1)
        end_bounds = _month_bounds(year + 1, last)
    if start is None or end_bounds is None:
        return None
    return start, end_bounds[1]


def _decade_bounds(start_year: int) -> tuple[date, date] | None:
    first = _year_bounds(start_year)
    last = _year_bounds(start_year + 9)
    if first is None or last is None:
        return None
    return first[0], last[1]


def _century_bounds(ordinal: int) -> tuple[date, date] | None:
    """20th century -> 1901-01-01 .. 2000-12-31 (decision 7)."""
    if not (1 <= ordinal <= 99):
        return None
    first = _year_bounds((ordinal - 1) * 100 + 1)
    last = _year_bounds(ordinal * 100)
    if first is None or last is None:
        return None
    return first[0], last[1]


def _apply_modifier(
    modifier: str | None, start: date, end: date, unit: str
) -> tuple[date, date]:
    """Narrow an interval for "early"/"mid"/"late".

    Months split 1-10 / 11-20 / 21-end (a rough but universally understood
    reading of "mid-February"). Years split into thirds of four months.
    Decades split 0-3 / 4-6 / 7-9. Centuries split into thirds.
    """
    if not modifier:
        return start, end
    modifier = modifier.lower()
    if modifier == "middle":
        modifier = "mid"

    if unit == "month":
        last_day = calendar.monthrange(start.year, start.month)[1]
        spans = {
            "early": (1, min(10, last_day)),
            "mid": (min(11, last_day), min(20, last_day)),
            "late": (min(21, last_day), last_day),
        }
        lo, hi = spans[modifier]
        a = _safe_date(start.year, start.month, lo)
        b = _safe_date(start.year, start.month, hi)
        return (a or start), (b or end)

    if unit == "year":
        spans = {"early": (1, 4), "mid": (5, 8), "late": (9, 12)}
        lo, hi = spans[modifier]
        a = _safe_date(start.year, lo, 1)
        bounds = _month_bounds(start.year, hi)
        return (a or start), (bounds[1] if bounds else end)

    if unit == "decade":
        # 0-3 / 4-6 / 7-9: "the early 1990s" is 1990-1993, "the mid-1990s" is
        # 1994-1996, "the late 1990s" is 1997-1999.
        spans = {"early": (0, 3), "mid": (4, 6), "late": (7, 9)}
        lo_offset, hi_offset = spans[modifier]
        lo_year, hi_year = start.year + lo_offset, start.year + hi_offset
        lo_bounds = _year_bounds(lo_year)
        hi_bounds = _year_bounds(hi_year)
        return (
            lo_bounds[0] if lo_bounds else start,
            hi_bounds[1] if hi_bounds else end,
        )

    if unit == "century":
        total = (end.year - start.year) + 1
        third = max(1, total // 3)
        if modifier == "early":
            lo_year, hi_year = start.year, start.year + third - 1
        elif modifier == "mid":
            lo_year, hi_year = start.year + third, end.year - third
        else:
            lo_year, hi_year = end.year - third + 1, end.year
        if lo_year > hi_year:
            lo_year, hi_year = hi_year, lo_year
        lo_bounds = _year_bounds(lo_year)
        hi_bounds = _year_bounds(hi_year)
        return (
            lo_bounds[0] if lo_bounds else start,
            hi_bounds[1] if hi_bounds else end,
        )

    return start, end


def _coarser(a: str, b: str) -> str:
    """The less precise of two precision labels -- a range is only as precise as
    its vaguest endpoint."""
    return a if _PRECISION_RANK.get(a, 0) >= _PRECISION_RANK.get(b, 0) else b


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _result(
    text: str,
    start: int,
    end: int,
    kind: str,
    start_date: date | None,
    end_date: date | None,
    precision: str,
    confidence: float,
) -> dict | None:
    if start_date is None or end_date is None:
        return None
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "kind": kind,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "precision": precision,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 2),
    }


_WORD_BEFORE_RE = re.compile(r"([A-Za-z.]+)\W*$")


def _preceding_word(text: str, position: int) -> str:
    """The alphabetic token immediately before `position`, lowercased and with
    a trailing period stripped ("pp." -> "pp"). Used by the identifier guard."""
    match = _WORD_BEFORE_RE.search(text[max(0, position - 24):position])
    if not match:
        return ""
    return match.group(1).strip().rstrip(".").lower()


def _blocked_by_identifier(text: str, position: int) -> bool:
    return _preceding_word(text, position) in _NON_TEMPORAL_PREFIXES


def _word_number(token: str) -> int | None:
    key = re.sub(r"\s+", " ", token.strip().lower())
    if key.isdigit():
        try:
            return int(key)
        except ValueError:
            return None
    return _NUMBER_WORDS.get(key)


# ---------------------------------------------------------------------------
# Rule handlers
#
# Each takes (match, text, reference_date) and returns a result dict or None.
# Returning None is how a rule declines a match it cannot validate -- e.g. the
# ISO rule declines "2024-13-45".
# ---------------------------------------------------------------------------


def _h_iso_day(m: re.Match[str], text: str, ref: date) -> dict | None:
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _result(text, m.start(), m.end(), "absolute",
                   _safe_date(year, month, day), _safe_date(year, month, day),
                   "day", 0.99)


def _h_iso_month(m: re.Match[str], text: str, ref: date) -> dict | None:
    bounds = _month_bounds(int(m.group(1)), int(m.group(2)))
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "month", 0.9)


def _h_slash_ymd(m: re.Match[str], text: str, ref: date) -> dict | None:
    """2024/03/03 -- year first, so unambiguous."""
    day = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return _result(text, m.start(), m.end(), "absolute", day, day, "day", 0.95)


def _ambiguous_numeric(
    m: re.Match[str],
    text: str,
    first: int,
    second: int,
    year: int,
    *,
    prefer_month_first: bool,
) -> dict | None:
    """Shared resolver for "a/b/yyyy" and "a.b.yyyy" (decisions 1 and 2).

    Builds both readings, keeps whichever are real calendar dates, and then:
      * one valid  -> that date, confidence 0.85
      * both valid and identical -> that date, confidence 0.90
      * both valid and different -> preferred reading's date as the anchor, but
        the interval is widened to cover BOTH readings and confidence drops to
        0.55, so the uncertainty is visible downstream instead of hidden.
    """
    month_first = _safe_date(year, first, second)
    day_first = _safe_date(year, second, first)

    if month_first is None and day_first is None:
        return None
    if month_first is None or day_first is None:
        only = month_first or day_first
        return _result(text, m.start(), m.end(), "absolute", only, only,
                       "day", 0.85)
    if month_first == day_first:
        return _result(text, m.start(), m.end(), "absolute", month_first,
                       month_first, "day", 0.9)

    lo, hi = sorted((month_first, day_first))
    return _result(text, m.start(), m.end(), "absolute", lo, hi, "day", 0.55)


def _h_slash_date(m: re.Match[str], text: str, ref: date) -> dict | None:
    first, second = int(m.group(1)), int(m.group(2))
    raw_year = m.group(3)
    year = int(raw_year)
    if len(raw_year) == 2:
        year = _expand_two_digit_year(year)
    return _ambiguous_numeric(m, text, first, second, year,
                              prefer_month_first=True)


def _h_dotted_date(m: re.Match[str], text: str, ref: date) -> dict | None:
    # Dotted form is European: day first (decision 2).
    day, month = int(m.group(1)), int(m.group(2))
    return _ambiguous_numeric(m, text, month, day, int(m.group(3)),
                              prefer_month_first=False)


def _h_slash_month_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    bounds = _month_bounds(int(m.group(2)), int(m.group(1)))
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "month", 0.8)


def _h_month_day_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    month = _month_num(m.group(1))
    if month is None:
        return None
    day = _safe_date(int(m.group(3)), month, int(m.group(2)))
    return _result(text, m.start(), m.end(), "absolute", day, day, "day", 0.97)


def _h_month_day_range(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"March 3-5, 2024" -- one month, two days."""
    month = _month_num(m.group(1))
    if month is None:
        return None
    year = int(m.group(4))
    start = _safe_date(year, month, int(m.group(2)))
    end = _safe_date(year, month, int(m.group(3)))
    if start is None or end is None or start > end:
        return None
    return _result(text, m.start(), m.end(), "range", start, end, "day", 0.93)


def _h_day_month_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    month = _month_num(m.group(2))
    if month is None:
        return None
    day = _safe_date(int(m.group(3)), month, int(m.group(1)))
    return _result(text, m.start(), m.end(), "absolute", day, day, "day", 0.96)


def _h_month_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    month = _month_num(m.group(1))
    if month is None:
        return None
    bounds = _month_bounds(int(m.group(2)), month)
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "month", 0.93)


def _h_month_apos_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    month = _month_num(m.group(1))
    if month is None:
        return None
    bounds = _month_bounds(_expand_two_digit_year(int(m.group(2))), month)
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "month", 0.85)


def _h_modified_month(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"mid-February", "late March 2021". Without a year we fall back to the
    reference year and drop confidence accordingly."""
    month = _month_num(m.group(2))
    if month is None:
        return None
    has_year = m.group(3) is not None
    year = int(m.group(3)) if has_year else ref.year
    bounds = _month_bounds(year, month)
    if bounds is None:
        return None
    start, end = _apply_modifier(m.group(1), bounds[0], bounds[1], "month")
    return _result(text, m.start(), m.end(), "partial", start, end, "month",
                   0.85 if has_year else 0.6)


def _h_modified_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    bounds = _year_bounds(int(m.group(2)))
    if bounds is None:
        return None
    start, end = _apply_modifier(m.group(1), bounds[0], bounds[1], "year")
    return _result(text, m.start(), m.end(), "partial", start, end, "year", 0.8)


def _h_bare_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    """Cue-licensed bare year (decision 4). The reported span is the year only,
    not the cue word, so `text` reads "1997" rather than "in 1997"."""
    year = int(m.group(1))
    if not (1000 <= year <= 2999):
        return None
    if _blocked_by_identifier(text, m.start(1)):
        return None
    bounds = _year_bounds(year)
    if bounds is None:
        return None
    return _result(text, m.start(1), m.end(1), "partial", bounds[0], bounds[1],
                   "year", 0.7)


def _h_quarter_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    year = (
        _expand_two_digit_year(int(m.group(2))) if m.group(2)
        else int(m.group(3))
    )
    bounds = _quarter_bounds(year, int(m.group(1)))
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "quarter", 0.92)


def _h_quarter_bare(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"Q1" with no year -- assume the reference year, low confidence."""
    bounds = _quarter_bounds(ref.year, int(m.group(1)))
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "quarter", 0.45)


def _h_ordinal_quarter(m: re.Match[str], text: str, ref: date) -> dict | None:
    quarter = _ORDINAL_NUM.get(m.group(1).lower())
    if quarter is None:
        return None
    year = int(m.group(2)) if m.group(2) else ref.year
    bounds = _quarter_bounds(year, quarter)
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "partial", bounds[0], bounds[1],
                   "quarter", 0.9 if m.group(2) else 0.45)


def _h_season_year(m: re.Match[str], text: str, ref: date) -> dict | None:
    year = (
        _expand_two_digit_year(int(m.group(2))) if m.group(2)
        else int(m.group(3))
    )
    bounds = _season_bounds(year, m.group(1))
    if bounds is None:
        return None
    return _result(text, m.start(), m.end(), "season", bounds[0], bounds[1],
                   "season", 0.85)


def _h_season_relative(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"last winter" / "next summer" / "this spring", resolved against ref.

    "last X" is the most recent occurrence of X that has already begun; "next X"
    is the first occurrence that has not yet begun. Winter's year-wrap is what
    makes this fiddly, so we just test the three candidate years and pick.
    """
    direction = m.group(1).lower()
    season = m.group(2).lower()
    candidates: list[tuple[date, date]] = []
    for year in (ref.year - 1, ref.year, ref.year + 1):
        bounds = _season_bounds(year, season)
        if bounds:
            candidates.append(bounds)
    if not candidates:
        return None

    if direction in ("last", "past", "previous"):
        past = [c for c in candidates if c[0] <= ref]
        chosen = past[-1] if past else candidates[0]
        # "last winter" in mid-winter should mean the previous one, not this one
        if chosen[0] <= ref <= chosen[1] and len(past) > 1:
            chosen = past[-2]
    elif direction in ("next", "coming", "upcoming", "following"):
        future = [c for c in candidates if c[0] > ref]
        chosen = future[0] if future else candidates[-1]
    else:  # "this"
        current = [c for c in candidates if c[0] <= ref <= c[1]]
        chosen = current[0] if current else candidates[-1]
    return _result(text, m.start(), m.end(), "season", chosen[0], chosen[1],
                   "season", 0.7)


def _h_decade_four(m: re.Match[str], text: str, ref: date) -> dict | None:
    bounds = _decade_bounds(int(m.group(2)))
    if bounds is None:
        return None
    start, end = _apply_modifier(m.group(1), bounds[0], bounds[1], "decade")
    return _result(text, m.start(), m.end(), "decade", start, end, "decade",
                   0.9 if not m.group(1) else 0.85)


def _h_decade_two(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"the 90s" / "'20s" -- most recent non-future decade (decision 8)."""
    value = int(m.group(2))
    base = ref.year - (ref.year % 100)
    start_year = base + (value - value % 10)
    if start_year > ref.year:
        start_year -= 100
    bounds = _decade_bounds(start_year)
    if bounds is None:
        return None
    start, end = _apply_modifier(m.group(1), bounds[0], bounds[1], "decade")
    return _result(text, m.start(), m.end(), "decade", start, end, "decade", 0.8)


def _h_century(m: re.Match[str], text: str, ref: date) -> dict | None:
    bounds = _century_bounds(int(m.group(2)))
    if bounds is None:
        return None
    start, end = _apply_modifier(m.group(1), bounds[0], bounds[1], "century")
    return _result(text, m.start(), m.end(), "decade", start, end, "century",
                   0.88)


# --- relative ---------------------------------------------------------------

_SIMPLE_RELATIVE_OFFSETS = {
    "today": 0, "tonight": 0, "yesterday": -1, "tomorrow": 1,
}


def _h_simple_relative(m: re.Match[str], text: str, ref: date) -> dict | None:
    offset = _SIMPLE_RELATIVE_OFFSETS[m.group(1).lower()]
    day = ref + timedelta(days=offset)
    return _result(text, m.start(), m.end(), "relative", day, day, "day", 0.95)


def _h_day_before_after(m: re.Match[str], text: str, ref: date) -> dict | None:
    offset = -2 if m.group(1).lower() == "before" else 2
    day = ref + timedelta(days=offset)
    return _result(text, m.start(), m.end(), "relative", day, day, "day", 0.9)


def _h_relative_weekday(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"last Tuesday" is the most recent Tuesday strictly before the reference
    day (so on a Tuesday it means a week ago, which is what people mean).
    "next Tuesday" is strictly after. "this Tuesday" is the one in the current
    Monday-start week, which may be in the past."""
    direction = m.group(1).lower()
    target = _WEEKDAY_NUM.get(m.group(2).lower())
    if target is None:
        return None
    if direction in ("last", "past", "previous"):
        delta = (ref.weekday() - target) % 7 or 7
        day = ref - timedelta(days=delta)
    elif direction in ("next", "coming", "upcoming", "following"):
        delta = (target - ref.weekday()) % 7 or 7
        day = ref + timedelta(days=delta)
    else:  # "this"
        day = ref - timedelta(days=ref.weekday()) + timedelta(days=target)
    return _result(text, m.start(), m.end(), "relative", day, day, "day", 0.8)


def _h_relative_unit(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"last week/month/quarter/year/decade" and their next/this variants.

    Each resolves to the whole calendar unit, not a point -- "last month" is all
    of the previous month. Weeks are Monday-start and report precision "day"
    (decision 11).
    """
    direction = m.group(1).lower()
    unit = m.group(2).lower()
    step = 0
    if direction in ("last", "past", "previous", "preceding"):
        step = -1
    elif direction in ("next", "coming", "upcoming", "following", "subsequent"):
        step = 1

    if unit in ("week", "fortnight"):
        width = 14 if unit == "fortnight" else 7
        monday = ref - timedelta(days=ref.weekday())
        start = monday + timedelta(days=width * step)
        return _result(text, m.start(), m.end(), "relative", start,
                       start + timedelta(days=width - 1), "day", 0.75)

    if unit == "month":
        anchor = _add_months(ref.replace(day=1), step)
        bounds = _month_bounds(anchor.year, anchor.month)
        if bounds is None:
            return None
        return _result(text, m.start(), m.end(), "relative", bounds[0],
                       bounds[1], "month", 0.85)

    if unit == "quarter":
        quarter = (ref.month - 1) // 3 + 1 + step
        year = ref.year
        while quarter < 1:
            quarter += 4
            year -= 1
        while quarter > 4:
            quarter -= 4
            year += 1
        bounds = _quarter_bounds(year, quarter)
        if bounds is None:
            return None
        return _result(text, m.start(), m.end(), "relative", bounds[0],
                       bounds[1], "quarter", 0.8)

    if unit == "year":
        bounds = _year_bounds(ref.year + step)
        if bounds is None:
            return None
        return _result(text, m.start(), m.end(), "relative", bounds[0],
                       bounds[1], "year", 0.85)

    if unit == "decade":
        base = ref.year - (ref.year % 10) + 10 * step
        bounds = _decade_bounds(base)
        if bounds is None:
            return None
        return _result(text, m.start(), m.end(), "relative", bounds[0],
                       bounds[1], "decade", 0.7)

    if unit == "century":
        ordinal = (ref.year - 1) // 100 + 1 + step
        bounds = _century_bounds(ordinal)
        if bounds is None:
            return None
        return _result(text, m.start(), m.end(), "relative", bounds[0],
                       bounds[1], "century", 0.7)

    return None


def _offset_by_unit(ref: date, amount: int, unit: str) -> date | None:
    if unit == "day":
        return ref + timedelta(days=amount)
    if unit == "week":
        return ref + timedelta(days=7 * amount)
    if unit == "fortnight":
        return ref + timedelta(days=14 * amount)
    if unit == "month":
        return _add_months(ref, amount)
    if unit == "year":
        return _add_months(ref, 12 * amount)
    if unit == "decade":
        return _add_months(ref, 120 * amount)
    return None


def _h_offset_phrase(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"three weeks ago", "two days later", "a year ago". Resolves to one exact
    day (decision 10)."""
    amount = _word_number(m.group(1))
    if amount is None or amount > 4000:
        return None
    unit = m.group(2).lower()
    tail = re.sub(r"\s+", " ", m.group(3).strip().lower())
    backwards = tail in ("ago", "earlier", "before", "previously", "prior")
    day = _offset_by_unit(ref, -amount if backwards else amount, unit)
    if day is None:
        return None
    spelled_out = not m.group(1).strip().isdigit()
    return _result(text, m.start(), m.end(), "relative", day, day, "day",
                   0.7 if spelled_out else 0.75)


def _h_in_offset(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"in two weeks" / "within three days" -- forward-looking."""
    amount = _word_number(m.group(1))
    if amount is None or amount > 4000:
        return None
    day = _offset_by_unit(ref, amount, m.group(2).lower())
    if day is None:
        return None
    return _result(text, m.start(), m.end(), "relative", day, day, "day", 0.65)


_VAGUE_WINDOWS = {
    "recently": (-90, 0), "lately": (-90, 0), "previously": (-365, 0),
    "soon": (0, 90), "shortly": (0, 90),
    "nowadays": (-30, 30), "currently": (-30, 30), "these days": (-30, 30),
}


def _h_vague(m: re.Match[str], text: str, ref: date) -> dict | None:
    """Emitted at confidence 0.20 with a wide window (decision 9)."""
    key = re.sub(r"\s+", " ", m.group(1).strip().lower())
    window = _VAGUE_WINDOWS.get(key)
    if window is None:
        return None
    return _result(text, m.start(), m.end(), "relative",
                   ref + timedelta(days=window[0]),
                   ref + timedelta(days=window[1]), "year", 0.2)


# --- ranges -----------------------------------------------------------------


def _parse_operand(
    fragment: str, ref: date, default_year: int | None = None
) -> tuple[date, date, str] | None:
    """Resolve one side of a range ("March", "June 2024", "2010", "Q3 2024").

    Runs the single-expression rules over the fragment first; if none covers it
    whole, falls back to the bare forms that only make sense inside a range --
    a lone month or quarter, which borrows the year from the other operand.
    """
    fragment = fragment.strip()
    if not fragment:
        return None

    for candidate in _scan(fragment, ref, allow_ranges=False):
        if candidate["start"] == 0 and candidate["end"] == len(fragment):
            return (
                date.fromisoformat(candidate["start_date"]),
                date.fromisoformat(candidate["end_date"]),
                candidate["precision"],
            )

    bare_year = re.fullmatch(r"\s*(\d{4})\s*", fragment)
    if bare_year:
        bounds = _year_bounds(int(bare_year.group(1)))
        return (bounds[0], bounds[1], "year") if bounds else None

    bare_month = re.fullmatch(_expand(r"\s*(<M>)\s*"), fragment, re.IGNORECASE)
    if bare_month:
        month = _month_num(bare_month.group(1))
        if month is None:
            return None
        bounds = _month_bounds(default_year or ref.year, month)
        return (bounds[0], bounds[1], "month") if bounds else None

    bare_quarter = re.fullmatch(r"\s*Q([1-4])\s*", fragment, re.IGNORECASE)
    if bare_quarter:
        bounds = _quarter_bounds(default_year or ref.year,
                                 int(bare_quarter.group(1)))
        return (bounds[0], bounds[1], "quarter") if bounds else None

    return None


def _build_range(
    m: re.Match[str], text: str, ref: date, confidence: float,
    *, require_ascending: bool = False,
) -> dict | None:
    """Resolve the right-hand operand first: it usually carries the year that
    the left-hand operand omits ("from March to June 2024")."""
    right = _parse_operand(m.group(2), ref)
    if right is None:
        return None
    left = _parse_operand(m.group(1), ref, default_year=right[0].year)
    if left is None:
        return None
    if require_ascending and left[0] > right[0]:
        return None
    return _result(text, m.start(), m.end(), "range",
                   min(left[0], right[0]), max(left[1], right[1]),
                   _coarser(left[2], right[2]), confidence)


def _h_range_from_to(m: re.Match[str], text: str, ref: date) -> dict | None:
    return _build_range(m, text, ref, 0.88)


def _h_range_between(m: re.Match[str], text: str, ref: date) -> dict | None:
    return _build_range(m, text, ref, 0.85)


def _h_range_dash(m: re.Match[str], text: str, ref: date) -> dict | None:
    """"1997-2003". Guarded against page/figure ranges ("pp. 1997-2003") and
    required to ascend, since a descending pair is almost always something
    other than a date range."""
    if _blocked_by_identifier(text, m.start()):
        return None
    return _build_range(m, text, ref, 0.82, require_ascending=True)


# ---------------------------------------------------------------------------
# The rule table
#
# Order here is documentation only -- overlap resolution is by match length,
# not by table position -- but grouping keeps it readable.
# ---------------------------------------------------------------------------

_Rule = tuple[str, re.Pattern[str], object, bool]  # name, regex, handler, is_range

_RULES: list[_Rule] = [
    # -- ranges (longest, so they win over their own operands) --
    ("range_from_to",
     _P(r"\bfrom\s+(<OP>)\s+(?:to|through|thru|until|till)\s+(<OP>)(?!\d)"),
     _h_range_from_to, True),
    ("range_between",
     _P(r"\bbetween\s+(<OP>)\s+and\s+(<OP>)(?!\d)"),
     _h_range_between, True),
    ("range_dash",
     _P(r"\b(<OP2>)\s*<DASH>\s*(<OP2>)(?!\d)"),
     _h_range_dash, True),
    ("month_day_range",
     _P(r"\b(<M>)\s+(\d{1,2})(?:st|nd|rd|th)?\s*<DASH>\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b"),
     _h_month_day_range, False),

    # -- absolute --
    ("iso_day", _P(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), _h_iso_day, False),
    ("iso_month", _P(r"\b(\d{4})-(0[1-9]|1[0-2])\b(?!-)"), _h_iso_month, False),
    ("slash_ymd", _P(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b"), _h_slash_ymd, False),
    ("slash_date", _P(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b"),
     _h_slash_date, False),
    ("dotted_date", _P(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"),
     _h_dotted_date, False),
    ("slash_month_year", _P(r"\b(0?[1-9]|1[0-2])/(\d{4})\b"),
     _h_slash_month_year, False),
    ("month_day_year",
     _P(r"\b(<M>)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b"),
     _h_month_day_year, False),
    ("day_month_year",
     _P(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(<M>)\s*,?\s*(\d{4})\b"),
     _h_day_month_year, False),

    # -- partial --
    ("month_year", _P(r"\b(<M>)\s+(?:of\s+)?(\d{4})\b"), _h_month_year, False),
    ("month_apos_year", _P(r"\b(<M>)\s*<APOS>(\d{2})\b"),
     _h_month_apos_year, False),
    ("modified_month",
     _P(r"\b(?:the\s+)?(<MOD>)[\s<DASHCH>]+(<M>)(?:\s+(?:of\s+)?(\d{4}))?\b"),
     _h_modified_month, False),
    ("modified_year", _P(r"\b(?:the\s+)?(<MOD>)[\s<DASHCH>]+(\d{4})\b(?!s)"),
     _h_modified_year, False),
    ("bare_year", _P(r"\b<CUE>\s+(\d{4})\b(?!\s*<COUNT>\b)(?!\s*<DASH>\s*\d)"),
     _h_bare_year, False),

    # -- quarters --
    ("quarter_year",
     _P(r"\bQ([1-4])\s*(?:of\s+)?(?:<APOS>(\d{2})|[\s<DASHCH>/]?\s*(\d{4}))\b"),
     _h_quarter_year, False),
    ("ordinal_quarter",
     _P(r"\b(?:the\s+)?(<ORD>)\s+quarter\s*(?:of\s+|,\s*)?(\d{4})?\b"),
     _h_ordinal_quarter, False),
    ("quarter_bare", _P(r"\bQ([1-4])\b"), _h_quarter_bare, False),

    # -- seasons --
    ("season_year",
     _P(r"\b(?:the\s+)?(<S>)\s+(?:of\s+)?(?:<APOS>(\d{2})|(\d{4}))\b"),
     _h_season_year, False),
    ("season_relative",
     _P(r"\b(last|next|this|past|coming|upcoming|previous|following)\s+(<S>)\b"),
     _h_season_relative, False),

    # -- decades / centuries --
    ("decade_four",
     _P(r"\b(?:the\s+)?(?:(<MOD>)[\s<DASHCH>]+)?((?:1[0-9]|20)\d0)<APOS>?s\b"),
     _h_decade_four, False),
    # Split into two rules rather than one alternation: a single pattern would
    # need two <MOD> groups and the handler could not tell which one fired.
    ("decade_two_apos",
     _P(r"\b(?:the\s+)?(?:(<MOD>)[\s<DASHCH>]+)?<APOS>([0-9]0)s\b"),
     _h_decade_two, False),
    ("decade_two_the",
     _P(r"\bthe\s+(?:(<MOD>)[\s<DASHCH>]+)?([0-9]0)s\b"),
     _h_decade_two, False),
    ("century",
     _P(r"\b(?:the\s+)?(?:(<MOD>)[\s<DASHCH>]+)?(\d{1,2})(?:st|nd|rd|th)\s+"
        r"century\b"),
     _h_century, False),

    # -- relative --
    ("day_before_after",
     _P(r"\bthe\s+day\s+(before|after)\s+(?:yesterday|tomorrow)\b"),
     _h_day_before_after, False),
    ("simple_relative", _P(r"\b(today|tonight|yesterday|tomorrow)\b"),
     _h_simple_relative, False),
    ("relative_weekday",
     _P(r"\b(last|next|this|past|coming|upcoming|previous|following)\s+(<WD>)\b"),
     _h_relative_weekday, False),
    ("relative_unit",
     _P(r"\b(?:the\s+)?(last|next|this|past|coming|upcoming|previous|"
        r"preceding|following|subsequent)\s+"
        r"(week|fortnight|month|quarter|year|decade|century)\b"),
     _h_relative_unit, False),
    ("offset_phrase",
     _P(r"\b(<NUM>)\s+(day|week|fortnight|month|year|decade)s?\s+"
        r"(ago|later|earlier|before|after|from now|from today|previously|prior)\b"),
     _h_offset_phrase, False),
    ("in_offset",
     _P(r"\b(?:in|within|after)\s+(<NUM>)\s+"
        r"(day|week|fortnight|month|year|decade)s?\b(?!\s+(?:ago|earlier))"),
     _h_in_offset, False),
    ("vague",
     _P(r"\b(recently|lately|soon|shortly|nowadays|currently|these days)\b"),
     _h_vague, False),
]


# ---------------------------------------------------------------------------
# Scanning and overlap resolution
# ---------------------------------------------------------------------------

# A single document is not allowed to melt the request thread. 2M characters is
# far beyond any real upload; past that we parse the head and stop.
_MAX_SCAN_CHARS = 2_000_000


def _scan(text: str, ref: date, *, allow_ranges: bool = True) -> list[dict]:
    """Run every rule, then keep the longest non-overlapping set (decision 12)."""
    candidates: list[dict] = []
    for _name, pattern, handler, is_range in _RULES:
        if is_range and not allow_ranges:
            continue
        for match in pattern.finditer(text):
            try:
                found = handler(match, text, ref)  # type: ignore[operator]
            except (ValueError, OverflowError, KeyError, IndexError):
                # A malformed expression is a non-match, never an exception that
                # reaches the caller. Document upload must not 500 on prose.
                found = None
            if found is not None:
                candidates.append(found)

    if not candidates:
        return []

    # Longest first, then most confident, then leftmost: a greedy sweep over
    # that order gives "prefer the longest match at a position" without an
    # interval tree.
    candidates.sort(
        key=lambda c: (-(c["end"] - c["start"]), -c["confidence"], c["start"])
    )
    accepted: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        span = (candidate["start"], candidate["end"])
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        occupied.append(span)
        accepted.append(candidate)

    accepted.sort(key=lambda c: (c["start"], c["end"]))
    return accepted


# ---------------------------------------------------------------------------
# Sentence spans
#
# text_kit.split_sentences normalizes whitespace, which shifts every character
# offset -- useless for mapping a match back to its sentence. This is the same
# boundary logic applied to the RAW string so offsets stay meaningful; the
# sentence text we hand back is then normalized for display.
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"([.!?]+)(\s+|$)")

_SPAN_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie",
    "fig", "al", "inc", "ltd", "co", "corp", "dept", "est", "approx", "no",
    "vol", "ed", "pp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "us", "uk", "eu", "un",
})


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENT_END_RE.finditer(text):
        end = match.end(1)
        if not text[start:end].strip():
            continue
        before = text[:match.start(1)]
        last_word = re.split(r"[\s(\[]", before)[-1].strip().lower().rstrip(".")
        if last_word in _SPAN_ABBREVIATIONS:
            continue
        if len(last_word) == 1 and last_word.isalpha():
            continue
        if last_word and last_word[-1].isdigit() and match.end() < len(text):
            if text[match.end():match.end() + 1].isdigit():
                continue
        spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _sentence_for(spans: list[tuple[int, int]], position: int) -> tuple[int, int] | None:
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end = spans[mid]
        if position < start:
            hi = mid - 1
        elif position >= end:
            lo = mid + 1
        else:
            return spans[mid]
    return None


def _snippet(text: str, start: int, end: int, width: int = 140) -> str:
    """A ~`width`-character window centred on the match, trimmed to whole words
    with ellipses where it was cut."""
    pad = max(0, (width - (end - start)) // 2)
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    if lo > 0:
        space = text.find(" ", lo, start)
        if space != -1:
            lo = space + 1
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        if space != -1:
            hi = space
    body = normalize(text[lo:hi])
    if lo > 0:
        body = "..." + body
    if hi < len(text):
        body = body + "..."
    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_dates(text: str, *, reference_date: date | None = None) -> list[dict]:
    """Find every temporal expression in `text`, in document order.

    Each result is a dict with `text`, `start`, `end`, `kind`, `start_date`,
    `end_date`, `precision` and `confidence`; see the module docstring for what
    the interval and precision mean and for every ambiguity rule applied.

    Relative expressions ("last Tuesday") resolve against `reference_date`,
    defaulting to today. Pass an explicit date whenever you have one -- a
    document written in 2023 means a 2023 Tuesday, and build_timeline() relies
    on exactly this.

    Non-string or empty input returns []; garbage input returns whatever
    genuinely parses and never raises.
    """
    if not isinstance(text, str) or not text:
        return []
    if len(text) > _MAX_SCAN_CHARS:
        text = text[:_MAX_SCAN_CHARS]
    ref = reference_date if isinstance(reference_date, date) else date.today()
    return _scan(text, ref)


def extract_events(text: str, *, reference_date: date | None = None) -> list[dict]:
    """extract_dates() plus the sentence each date sits in.

    A timeline entry with no text next to it is useless, so every event carries
    `sentence` (the whole containing sentence, whitespace-normalized) and
    `snippet` (a ~140-character window centred on the expression, for a compact
    list view).
    """
    dates = extract_dates(text, reference_date=reference_date)
    if not dates:
        return []

    spans = _sentence_spans(text)
    if not spans:
        # Degenerate input with no sentence punctuation at all: fall back to
        # text_kit's splitter over the whole string.
        fallback = split_sentences(text)
        sentence_text = fallback[0] if fallback else normalize(text)
        return [
            {**item, "sentence": sentence_text,
             "snippet": _snippet(text, item["start"], item["end"])}
            for item in dates
        ]

    events: list[dict] = []
    for item in dates:
        span = _sentence_for(spans, item["start"])
        sentence = normalize(text[span[0]:span[1]]) if span else ""
        events.append({
            **item,
            "sentence": sentence,
            "snippet": _snippet(text, item["start"], item["end"]),
        })
    return events


def _document_reference(document: dict, fallback: date) -> date:
    """A document's own creation date is the right anchor for its relative
    expressions -- "last Tuesday" in a 2023 note is a 2023 Tuesday. Falls back
    to the caller's reference date when created_at is missing or unparseable."""
    raw = document.get("created_at")
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and len(raw) >= 10:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return fallback
    return fallback


def build_timeline(
    documents: list[dict], *, reference_date: date | None = None
) -> dict:
    """Merge many documents into one chronology.

    `documents` is a list of {"id", "title", "text", "created_at"} dicts.
    Relative expressions in each document resolve against that document's own
    `created_at` rather than today (see _document_reference) -- otherwise a
    year-old note's "last Tuesday" lands in the wrong place on the axis and the
    whole timeline quietly rots as time passes.

    Returns events sorted by start_date (ties broken by document and position),
    year/decade histograms for an overview strip, and the overall span.
    Malformed documents are skipped rather than raising: a timeline over a
    thousand uploads should not die on one bad row.
    """
    fallback = reference_date if isinstance(reference_date, date) else date.today()
    events: list[dict] = []

    if isinstance(documents, list):
        for document in documents:
            if not isinstance(document, dict):
                continue
            text = document.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            ref = _document_reference(document, fallback)
            for event in extract_events(text, reference_date=ref):
                event["document_id"] = document.get("id")
                event["document_title"] = document.get("title") or ""
                events.append(event)

    events.sort(key=lambda e: (
        e["start_date"], e["end_date"], str(e.get("document_id")), e["start"]
    ))

    by_year: dict[str, int] = {}
    by_decade: dict[str, int] = {}
    for event in events:
        year = event["start_date"][:4]
        by_year[year] = by_year.get(year, 0) + 1
        decade = f"{int(year) // 10 * 10}s"
        by_decade[decade] = by_decade.get(decade, 0) + 1

    return {
        "events": events,
        "buckets": {
            "by_year": {k: by_year[k] for k in sorted(by_year)},
            "by_decade": {k: by_decade[k] for k in sorted(by_decade)},
        },
        "span": {
            "earliest": min((e["start_date"] for e in events), default=None),
            "latest": max((e["end_date"] for e in events), default=None),
        },
        "total": len(events),
    }


# ---------------------------------------------------------------------------
# Self-test
#
# Run directly:  python temporal.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    import time

    REF = date(2024, 6, 15)  # a Saturday, in a leap year -- both matter below

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"PASS  {label}")
        else:
            failed += 1
            print(f"FAIL  {label}  {detail}")

    def first(text: str) -> dict | None:
        found = extract_dates(text, reference_date=REF)
        return found[0] if found else None

    # (sentence, expected literal, start, end, kind, precision)
    CASES = [
        ("The contract was signed on March 3, 2024.",
         "March 3, 2024", "2024-03-03", "2024-03-03", "absolute", "day"),
        ("Filed 3 March 2024 in London.",
         "3 March 2024", "2024-03-03", "2024-03-03", "absolute", "day"),
        ("Effective 2024-03-03 onwards.",
         "2024-03-03", "2024-03-03", "2024-03-03", "absolute", "day"),
        ("Dated 3/3/2024.",
         "3/3/2024", "2024-03-03", "2024-03-03", "absolute", "day"),
        # Both readings valid -> widened interval, MM/DD anchor (decision 1)
        ("Invoice 12/03/2022 was paid.",
         "12/03/2022", "2022-03-12", "2022-12-03", "absolute", "day"),
        # Only DD/MM is valid -> that reading
        ("Shipped 25/12/2022 by courier.",
         "25/12/2022", "2022-12-25", "2022-12-25", "absolute", "day"),
        ("Dotted European form 03.03.2024 here.",
         "03.03.2024", "2024-03-03", "2024-03-03", "absolute", "day"),
        ("Period 03/2024 closed.",
         "03/2024", "2024-03-01", "2024-03-31", "partial", "month"),
        ("Revenue in March 2024 rose.",
         "March 2024", "2024-03-01", "2024-03-31", "partial", "month"),
        ("Shipped Mar '24 to beta users.",
         "Mar '24", "2024-03-01", "2024-03-31", "partial", "month"),
        ("Written March '21 originally.",
         "March '21", "2021-03-01", "2021-03-31", "partial", "month"),
        ("The plant opened in 1997 nearby.",
         "1997", "1997-01-01", "1997-12-31", "partial", "year"),
        ("Guidance for Q3 2024 was raised.",
         "Q3 2024", "2024-07-01", "2024-09-30", "partial", "quarter"),
        ("Results for the third quarter of 2024 are in.",
         "the third quarter of 2024", "2024-07-01", "2024-09-30",
         "partial", "quarter"),
        ("Hiring slowed in Q1 sharply.",
         "Q1", "2024-01-01", "2024-03-31", "partial", "quarter"),
        ("Woodstock, summer 1969, changed music.",
         "summer 1969", "1969-06-01", "1969-08-31", "season", "season"),
        ("The summer of 1969 was hot.",
         "The summer of 1969", "1969-06-01", "1969-08-31", "season", "season"),
        # Winter wraps the year boundary; 2024 is a leap year
        ("We shipped it last winter.",
         "last winter", "2023-12-01", "2024-02-29", "season", "season"),
        ("Grunge defined the 1990s completely.",
         "the 1990s", "1990-01-01", "1999-12-31", "decade", "decade"),
        ("Grunge defined the 90s completely.",
         "the 90s", "1990-01-01", "1999-12-31", "decade", "decade"),
        ("Industrialisation marked the 20th century.",
         "the 20th century", "1901-01-01", "2000-12-31", "decade", "century"),
        ("He served 1997-2003 in office.",
         "1997-2003", "1997-01-01", "2003-12-31", "range", "year"),
        ("Runs from March to June 2024 inclusive.",
         "from March to June 2024", "2024-03-01", "2024-06-30", "range", "month"),
        ("Built between 2010 and 2015 slowly.",
         "between 2010 and 2015", "2010-01-01", "2015-12-31", "range", "year"),
        ("The summit ran March 3-5, 2024 in Bonn.",
         "March 3-5, 2024", "2024-03-03", "2024-03-05", "range", "day"),
        ("It happened yesterday morning.",
         "yesterday", "2024-06-14", "2024-06-14", "relative", "day"),
        ("We met last Tuesday about it.",
         "last Tuesday", "2024-06-11", "2024-06-11", "relative", "day"),
        ("Launching next month for sure.",
         "next month", "2024-07-01", "2024-07-31", "relative", "month"),
        ("Filed three weeks ago already.",
         "three weeks ago", "2024-05-25", "2024-05-25", "relative", "day"),
        ("Confirmed two days later by email.",
         "two days later", "2024-06-17", "2024-06-17", "relative", "day"),
        ("Started a year ago roughly.",
         "a year ago", "2023-06-15", "2023-06-15", "relative", "day"),
        ("Due in two weeks at latest.",
         "in two weeks", "2024-06-29", "2024-06-29", "relative", "day"),
        ("Snow fell in mid-February heavily.",
         "mid-February", "2024-02-11", "2024-02-20", "partial", "month"),
        ("Hiring picked up in early 2024 again.",
         "early 2024", "2024-01-01", "2024-04-30", "partial", "year"),
        ("The thaw came late March that year.",
         "late March", "2024-03-21", "2024-03-31", "partial", "month"),
        ("The bubble burst in the late 1990s.",
         "the late 1990s", "1997-01-01", "1999-12-31", "decade", "decade"),
        ("Cheap credit defined the early 1990s.",
         "the early 1990s", "1990-01-01", "1993-12-31", "decade", "decade"),
        ("Britpop peaked in the mid-1990s.",
         "the mid-1990s", "1994-01-01", "1996-12-31", "decade", "decade"),
        ("Synths dominated the '80s.",
         "the '80s", "1980-01-01", "1989-12-31", "decade", "decade"),
        ("Inflation bit in the 20s.",
         "the 20s", "2020-01-01", "2029-12-31", "decade", "decade"),
        ("Quarter began 2024-03 per the ledger.",
         "2024-03", "2024-03-01", "2024-03-31", "partial", "month"),
        ("Attacks on Sept. 9, 2001 changed policy.",
         "Sept. 9, 2001", "2001-09-09", "2001-09-09", "absolute", "day"),
        ("See you tomorrow afternoon.",
         "tomorrow", "2024-06-16", "2024-06-16", "relative", "day"),
        ("Deadline is next Friday sharp.",
         "next Friday", "2024-06-21", "2024-06-21", "relative", "day"),
        ("Recorded 2024/03/03 in the log.",
         "2024/03/03", "2024-03-03", "2024-03-03", "absolute", "day"),
        ("Sales fell last quarter noticeably.",
         "last quarter", "2024-01-01", "2024-03-31", "relative", "quarter"),
    ]

    print("=" * 70)
    print("A. expression cases")
    print("=" * 70)
    for sentence, literal, start_iso, end_iso, kind, precision in CASES:
        got = first(sentence)
        if got is None:
            check(f"{literal!r}", False, "no match at all")
            continue
        ok = (
            got["text"] == literal
            and got["start_date"] == start_iso
            and got["end_date"] == end_iso
            and got["kind"] == kind
            and got["precision"] == precision
            and 0.0 <= got["confidence"] <= 1.0
            and sentence[got["start"]:got["end"]] == got["text"]
        )
        check(
            f"{literal!r}",
            ok,
            f"got text={got['text']!r} {got['start_date']}..{got['end_date']} "
            f"kind={got['kind']} precision={got['precision']}",
        )

    print()
    print("=" * 70)
    print("B. ambiguity handling")
    print("=" * 70)
    ambiguous = first("Invoice 12/03/2022 was paid.")
    check("ambiguous DD/MM vs MM/DD lowers confidence",
          ambiguous is not None and ambiguous["confidence"] <= 0.6,
          f"confidence={ambiguous and ambiguous['confidence']}")
    unambiguous = first("Shipped 25/12/2022 by courier.")
    check("unambiguous numeric date keeps high confidence",
          unambiguous is not None and unambiguous["confidence"] >= 0.8,
          f"confidence={unambiguous and unambiguous['confidence']}")
    same = first("Dated 3/3/2024.")
    check("day == month collapses to one reading",
          same is not None and same["confidence"] >= 0.85
          and same["start_date"] == same["end_date"],
          f"{same}")
    vague = first("We shipped it recently.")
    check("vague expression emitted at very low confidence",
          vague is not None and vague["confidence"] <= 0.25,
          f"{vague}")

    print()
    print("=" * 70)
    print("C. false positives (must NOT match)")
    print("=" * 70)
    NEGATIVES = [
        "We now serve 4,000 users across the country.",
        "See page 1997 of the appendix.",
        "Upgrade to version 2.0 of the app.",
        "Meet me in Room 2024 upstairs.",
        "The identifier is 1234567 exactly.",
        "About 1/2 of the team agreed.",
        "The score was 3-2 at halftime.",
        "Refer to chapter 12 for details.",
        "Revenue grew 3.5 percent overall.",
        "Order 15000 units for the warehouse.",
        "Ticket #2019 is still open.",
        "See pp. 1997-2003 for the argument.",
        "He is in his 20s and still learning.",
    ]
    for sentence in NEGATIVES:
        found = extract_dates(sentence, reference_date=REF)
        check(f"no match in {sentence!r}", not found,
              f"matched {[f['text'] for f in found]}")

    # ...but the same numerals WITH a cue must still match.
    for sentence, expected in [
        ("The plant opened in 1997.", "1997"),
        ("Operating since 2019 continuously.", "2019"),
        ("Completed by 2030 at the latest.", "2030"),
        ("The war of 1812 is well documented.", "1812"),
    ]:
        got = first(sentence)
        check(f"cue licenses year in {sentence!r}",
              got is not None and got["text"] == expected,
              f"got {got}")

    print()
    print("=" * 70)
    print("D. relative resolution against a fixed reference date")
    print("=" * 70)
    RELATIVE = [
        ("yesterday", "2024-06-14", "2024-06-14"),
        ("today", "2024-06-15", "2024-06-15"),
        ("tomorrow", "2024-06-16", "2024-06-16"),
        ("last Tuesday", "2024-06-11", "2024-06-11"),
        ("next Tuesday", "2024-06-18", "2024-06-18"),
        ("last week", "2024-06-03", "2024-06-09"),
        ("last month", "2024-05-01", "2024-05-31"),
        ("next year", "2025-01-01", "2025-12-31"),
        ("five days ago", "2024-06-10", "2024-06-10"),
        ("two months ago", "2024-04-15", "2024-04-15"),
        ("the day before yesterday", "2024-06-13", "2024-06-13"),
    ]
    for phrase, start_iso, end_iso in RELATIVE:
        got = first(f"It happened {phrase}, apparently.")
        check(f"relative {phrase!r}",
              got is not None and got["start_date"] == start_iso
              and got["end_date"] == end_iso,
              f"got {got and (got['text'], got['start_date'], got['end_date'])}")

    # The same phrase must move with the reference date.
    shifted = extract_dates("It happened last Tuesday.",
                            reference_date=date(2023, 3, 9))
    check("relative dates follow the reference date",
          shifted and shifted[0]["start_date"] == "2023-03-07",
          f"got {shifted}")

    print()
    print("=" * 70)
    print("E. overlap / longest-match resolution")
    print("=" * 70)
    overlap = extract_dates("Revenue in March 2024 was flat.", reference_date=REF)
    check("'March 2024' beats bare 'March'",
          len(overlap) == 1 and overlap[0]["text"] == "March 2024",
          f"got {[o['text'] for o in overlap]}")
    overlap2 = extract_dates("Runs from March to June 2024 inclusive.",
                             reference_date=REF)
    check("range beats its own operands",
          len(overlap2) == 1 and overlap2[0]["text"] == "from March to June 2024",
          f"got {[o['text'] for o in overlap2]}")
    multi = extract_dates(
        "Signed March 3, 2024, effective Q3 2024, expiring 2027-12-31.",
        reference_date=REF,
    )
    check("three distinct expressions, in document order",
          [x["text"] for x in multi] == ["March 3, 2024", "Q3 2024", "2027-12-31"],
          f"got {[x['text'] for x in multi]}")

    print()
    print("=" * 70)
    print("F. ranges")
    print("=" * 70)
    RANGES = [
        ("He served 1997-2003.", "1997-01-01", "2003-12-31", "year"),
        ("Runs from March to June 2024.", "2024-03-01", "2024-06-30", "month"),
        ("Built between 2010 and 2015.", "2010-01-01", "2015-12-31", "year"),
        ("Open March-June 2024 only.", "2024-03-01", "2024-06-30", "month"),
        ("The summit ran March 3-5, 2024.", "2024-03-03", "2024-03-05", "day"),
        ("Valid from 2024-01-15 to 2024-02-20.",
         "2024-01-15", "2024-02-20", "day"),
    ]
    for sentence, start_iso, end_iso, precision in RANGES:
        got = first(sentence)
        check(f"range in {sentence!r}",
              got is not None and got["start_date"] == start_iso
              and got["end_date"] == end_iso and got["precision"] == precision
              and got["kind"] == "range",
              f"got {got}")

    print()
    print("=" * 70)
    print("G. extract_events")
    print("=" * 70)
    doc = (
        "The team met on March 3, 2024. They agreed to ship in Q3 2024. "
        "Nothing happened after that."
    )
    evs = extract_events(doc, reference_date=REF)
    check("two events extracted", len(evs) == 2, f"got {len(evs)}")
    check("first event carries its sentence",
          evs and evs[0]["sentence"] == "The team met on March 3, 2024.",
          f"got {evs and evs[0]['sentence']!r}")
    check("second event carries its sentence",
          len(evs) > 1 and evs[1]["sentence"] == "They agreed to ship in Q3 2024.",
          f"got {len(evs) > 1 and evs[1]['sentence']!r}")
    check("snippets are bounded and non-empty",
          all(0 < len(e["snippet"]) <= 200 for e in evs),
          f"got {[len(e['snippet']) for e in evs]}")

    print()
    print("=" * 70)
    print("H. build_timeline over three documents")
    print("=" * 70)
    DOCS = [
        {"id": 1, "title": "Founding memo",
         "text": "The company was incorporated in 1997. "
                 "Our first office opened in the summer of 1998.",
         "created_at": "2020-01-05T10:00:00"},
        {"id": 2, "title": "Old note",
         "text": "We signed the lease last Tuesday. "
                 "Renovation runs from March to June 2015.",
         "created_at": "2015-02-10"},
        {"id": 3, "title": "Recent update",
         "text": "Guidance for Q3 2024 was raised on March 3, 2024. "
                 "We expect growth through the 2020s.",
         "created_at": "2024-04-01"},
    ]
    timeline = build_timeline(DOCS, reference_date=REF)
    check("timeline has the expected event count",
          timeline["total"] == 7, f"got {timeline['total']}: "
          f"{[(e['document_id'], e['text']) for e in timeline['events']]}")
    check("events sorted by start_date",
          [e["start_date"] for e in timeline["events"]]
          == sorted(e["start_date"] for e in timeline["events"]),
          f"got {[e['start_date'] for e in timeline['events']]}")
    check("every event tagged with its document",
          all(e["document_id"] in (1, 2, 3) and e["document_title"]
              for e in timeline["events"]),
          "missing document tags")
    # Doc 2's "last Tuesday" must resolve against 2015-02-10 (a Tuesday), so the
    # previous Tuesday is 2015-02-03 -- NOT a Tuesday near the reference date.
    relative_event = next(
        (e for e in timeline["events"] if e["text"].lower() == "last tuesday"), None
    )
    check("relative date resolved against the document's own created_at",
          relative_event is not None
          and relative_event["start_date"] == "2015-02-03",
          f"got {relative_event}")
    check("span covers the earliest and latest instants",
          timeline["span"]["earliest"] == "1997-01-01"
          and timeline["span"]["latest"] == "2029-12-31",
          f"got {timeline['span']}")
    check("year buckets counted",
          timeline["buckets"]["by_year"].get("1997") == 1
          and timeline["buckets"]["by_year"].get("2015") == 2,
          f"got {timeline['buckets']['by_year']}")
    check("decade buckets counted",
          sum(timeline["buckets"]["by_decade"].values()) == timeline["total"],
          f"got {timeline['buckets']['by_decade']}")
    check("empty document list is handled",
          build_timeline([]) == {
              "events": [], "buckets": {"by_year": {}, "by_decade": {}},
              "span": {"earliest": None, "latest": None}, "total": 0},
          "unexpected empty result")

    print()
    print("=" * 70)
    print("I. robustness")
    print("=" * 70)
    GARBAGE = [
        "", "   ", "\x00\x01\x02", "2024-13-45 is not a date",
        "30/30/2024 nonsense", "中文文本 2024年",
        "\U0001f600\U0001f680 emoji only", "-" * 500, "/" * 500,
        "2024-02-30 never existed", "99/99/9999",
    ]
    for junk in GARBAGE:
        try:
            extract_dates(junk, reference_date=REF)
            extract_events(junk, reference_date=REF)
            check(f"survives {junk[:24]!r}", True)
        except Exception as exc:  # noqa: BLE001 - this is the point of the test
            check(f"survives {junk[:24]!r}", False, f"raised {exc!r}")

    check("non-string input returns []",
          extract_dates(None) == [] and extract_dates(12345) == [],  # type: ignore[arg-type]
          "did not return []")
    check("impossible dates are declined",
          not extract_dates("2024-02-30 never existed", reference_date=REF),
          f"got {extract_dates('2024-02-30 never existed', reference_date=REF)}")

    big = ("On March 3, 2024 the team met in Berlin to discuss the 1990s "
           "and plans for Q3 2024. ") * 1200
    started = time.perf_counter()
    big_result = extract_dates(big, reference_date=REF)
    elapsed = time.perf_counter() - started
    check(f"100k-char document parses fast ({len(big)} chars, {elapsed:.2f}s)",
          elapsed < 5.0 and len(big_result) == 3600,
          f"elapsed={elapsed:.2f}s matches={len(big_result)}")

    check("offsets always index back to the matched text",
          all(big[e["start"]:e["end"]] == e["text"] for e in big_result[:50]),
          "offset mismatch")

    print()
    print("=" * 70)
    print(f"TOTAL: {passed} passed, {failed} failed "
          f"({passed + failed} checks)")
    print("=" * 70)
    sys.exit(1 if failed else 0)
