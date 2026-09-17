"""
quantities -- deterministic extraction, normalization and contradiction
detection for the numbers a user has read across many documents.

WHY THIS EXISTS
---------------
People read a dozen sources about the same topic and the sources quietly
disagree: one says a product has 4.2 million users, another says 6.1
million. Nobody notices, because the two numbers live in different
documents read three weeks apart. This module finds every quantity in a
corpus, converts them onto a common scale, clusters the ones that are
about the same thing, and reports the disagreements.

WHY NO LANGUAGE MODEL
---------------------
The failure mode we are guarding against is a *wrong number*. A language
model asked "what quantities are in this text" will, often enough to
matter, return a number that is not in the text at all -- which in a
contradiction detector is catastrophic: it manufactures the very thing it
is supposed to find. Everything here is a regex, a graph search and a
clustering rule. It is deterministic (same input, same output, forever),
auditable (every reported number has a character offset you can look at),
free, and offline. A reviewer can check the whole pipeline by hand.

DESIGN IN ONE PARAGRAPH
-----------------------
`extract_quantities` scans normalized text with a single master regex that
understands number forms (separators, decimals, scale words, scientific
notation, spelled-out small numbers), optional approximation markers,
optional currency symbols on either side, ranges, and a trailing unit or
noun. Each hit is converted to the base unit of its dimension by a BFS
over a *unit graph* whose edges are affine transforms (factor + offset),
so temperature -- which is affine, not multiplicative -- composes through
exactly the same machinery as length. `group_claims` clusters hits by
(subject, dimension, base unit), and `find_contradictions` runs the whole
pipeline over a document set and grades the disagreements.

OFFSET CONVENTION
-----------------
All offsets (`start`, `end`) index into `text_kit.normalize(text)`, not
the raw input. Normalization folds unicode, flattens curly quotes and
en-dashes, and collapses runs of whitespace -- we have to normalize before
matching (otherwise "4.2 million\\n users" is invisible), and returning
offsets into a string the caller does not have would be worse than
useless. The `text` field always contains the literal matched substring,
so a caller that needs to locate it in the raw document can search for it.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque

try:  # package import (normal Flask runtime)
    from .text_kit import STOPWORDS, content_tokens, levenshtein, normalize, split_sentences
except ImportError:  # direct execution: `python quantities.py`
    from text_kit import STOPWORDS, content_tokens, levenshtein, normalize, split_sentences


# ===========================================================================
# 1. THE UNIT GRAPH
#
# Units are nodes; conversions are directed edges carrying an AFFINE
# transform `value_to = value_from * factor + offset`. Almost every edge
# has offset 0 (a kilometre is a thousand metres and nothing more), but
# temperature genuinely is affine: 0 degrees Celsius is 273.15 kelvin, not
# 0 kelvin. Multiplying a Celsius reading by a factor is simply wrong, and
# a codebase that models conversions as bare floats will make that mistake
# eventually. Modelling every edge as (factor, offset) costs one extra
# float per edge and makes the wrong thing unrepresentable.
#
# Affine transforms compose cleanly, which is what makes the graph search
# legitimate:
#     z = a2*(a1*x + b1) + b2 = (a2*a1)*x + (a2*b1 + b2)
# and invert cleanly:
#     x = (y - b) / a
# so we only declare each conversion once and get the reverse for free.
#
# We deliberately do NOT hardcode a unit->base factor table. Declaring the
# "natural" edges (mm->cm, cm->m, inch->cm, foot->inch, yard->foot,
# mile->foot) and searching for a path means the factors are the ones a
# reader can verify from memory, and adding a unit means adding one edge
# rather than recomputing a base factor by hand.
# ===========================================================================

# (from_unit, to_unit, factor, offset)
_EDGES: list[tuple[str, str, float, float]] = [
    # --- length (base: meter) ---
    ("millimeter", "centimeter", 0.1, 0.0),
    ("centimeter", "meter", 0.01, 0.0),
    ("kilometer", "meter", 1000.0, 0.0),
    ("inch", "centimeter", 2.54, 0.0),          # exact by international definition
    ("foot", "inch", 12.0, 0.0),
    ("yard", "foot", 3.0, 0.0),
    ("mile", "foot", 5280.0, 0.0),
    ("nautical_mile", "meter", 1852.0, 0.0),

    # --- mass (base: kilogram) ---
    ("milligram", "gram", 0.001, 0.0),
    ("gram", "kilogram", 0.001, 0.0),
    ("tonne", "kilogram", 1000.0, 0.0),         # metric tonne
    ("ounce", "gram", 28.349523125, 0.0),       # exact avoirdupois definition
    ("pound", "ounce", 16.0, 0.0),

    # --- time (base: second) ---
    ("millisecond", "second", 0.001, 0.0),
    ("minute", "second", 60.0, 0.0),
    ("hour", "minute", 60.0, 0.0),
    ("day", "hour", 24.0, 0.0),
    ("week", "day", 7.0, 0.0),
    # A month has no fixed length. We use exactly one twelfth of our own
    # year constant (365.25 / 12 = 30.4375 days) so that "12 months" and
    # "1 year" canonicalize to the identical float -- a unit graph whose
    # year and month constants disagree manufactures contradictions out of
    # its own arithmetic. It is still an APPROXIMATION, which is why any
    # claim measured in months or years is marked approximate below:
    # "3 months" against "90 days" is a 1.4% gap that is an artifact of
    # this constant, not a disagreement between the two authors.
    ("month", "day", 365.25 / 12.0, 0.0),
    ("year", "day", 365.25, 0.0),

    # --- data (base: byte) ---
    # Decimal (SI) vs binary (IEC) is a real and frequently-botched
    # distinction: a "1 TB" drive holds 10^12 bytes, while 1 TiB of RAM is
    # 2^40 bytes -- a 10% gap that would otherwise show up as a phantom
    # contradiction. We follow the standards: KB/MB/GB/TB are powers of
    # 1000, KiB/MiB/GiB/TiB are powers of 1024. Writers who mean 1024 by
    # "KB" will be read as meaning 1000; that is a defensible, documented,
    # deterministic choice. Note the consequence: a document saying
    # "512 GB" and one saying "512 GiB" are 7.4% apart and WILL surface as
    # a minor discrepancy. That is the right outcome -- those two
    # documents really do state different capacities -- and the 2x floor
    # for "major" guarantees a prefix mismatch can never be escalated to
    # a headline contradiction.
    ("bit", "byte", 0.125, 0.0),
    ("kilobyte", "byte", 1000.0, 0.0),
    ("megabyte", "kilobyte", 1000.0, 0.0),
    ("gigabyte", "megabyte", 1000.0, 0.0),
    ("terabyte", "gigabyte", 1000.0, 0.0),
    ("petabyte", "terabyte", 1000.0, 0.0),
    ("kibibyte", "byte", 1024.0, 0.0),
    ("mebibyte", "kibibyte", 1024.0, 0.0),
    ("gibibyte", "mebibyte", 1024.0, 0.0),
    ("tebibyte", "gibibyte", 1024.0, 0.0),

    # --- percent (base: percent) ---
    ("basis_point", "percent", 0.01, 0.0),

    # --- temperature (base: kelvin) --- THE AFFINE CASE
    # Kelvin is the base on purpose. Contradiction severity below is a
    # RATIO of canonical values, and ratios are only meaningful on an
    # absolute scale: in Celsius, 0.1 vs 5.0 is a 50x "disagreement" and
    # -5 vs 5 is a ratio with a sign flip, both nonsense. On the kelvin
    # scale those are 273.25 vs 278.15 -- a 1.8% difference, which is what
    # a physicist would call them.
    ("celsius", "kelvin", 1.0, 273.15),
    ("fahrenheit", "celsius", 5.0 / 9.0, -160.0 / 9.0),   # (F - 32) * 5/9

    # --- speed (base: meter_per_second) ---
    ("kilometer_per_hour", "meter_per_second", 1.0 / 3.6, 0.0),
    ("mile_per_hour", "meter_per_second", 0.44704, 0.0),  # exact
    ("knot", "meter_per_second", 1852.0 / 3600.0, 0.0),
    ("foot_per_second", "meter_per_second", 0.3048, 0.0),

    # --- currency ---
    # NOTE, AND THIS IS THE IMPORTANT ONE: there are NO edges between
    # currencies. Not because it is hard, but because it is WRONG. Every
    # other edge in this graph is a definition -- a mile has been 5280
    # feet for centuries and will be tomorrow. An exchange rate is a
    # market price that moved while you were reading this sentence, and
    # the rate that matters for a claim is the rate on the day the claim
    # was written, which we do not know and the document usually does not
    # say. Converting "$3M" and "EUR 3M" onto one axis would let us report
    # a "contradiction" that is really just FX drift, or hide a real one
    # behind it. So each currency is its own closed dimension: a dollar
    # figure is only ever compared with other dollar figures. Cross-
    # currency claims about the same subject are surfaced as separate
    # groups, never as a conflict.
    #
    # Sub-units convert within their own currency only.
    ("cent", "usd", 0.01, 0.0),
    ("pence", "gbp", 0.01, 0.0),
    ("paise", "inr", 0.01, 0.0),
    ("eurocent", "eur", 0.01, 0.0),
]

# Which dimension each unit belongs to, and the base unit of that
# dimension. `dimension_key` is the internal clustering key and differs
# from the public `dimension` only for currency, where each currency is
# its own comparison universe (see the long note above).
_DIMENSION_BASE: dict[str, str] = {
    "length": "meter",
    "mass": "kilogram",
    "time": "second",
    "data": "byte",
    "percent": "percent",
    "temperature": "kelvin",
    "speed": "meter_per_second",
}

_CURRENCIES = ("usd", "eur", "gbp", "inr", "jpy")

# unit -> dimension
_UNIT_DIMENSION: dict[str, str] = {}
for _u in (
    "millimeter", "centimeter", "meter", "kilometer", "inch", "foot", "yard",
    "mile", "nautical_mile",
):
    _UNIT_DIMENSION[_u] = "length"
for _u in ("milligram", "gram", "kilogram", "tonne", "ounce", "pound"):
    _UNIT_DIMENSION[_u] = "mass"
for _u in (
    "millisecond", "second", "minute", "hour", "day", "week", "month", "year",
):
    _UNIT_DIMENSION[_u] = "time"
for _u in (
    "bit", "byte", "kilobyte", "megabyte", "gigabyte", "terabyte", "petabyte",
    "kibibyte", "mebibyte", "gibibyte", "tebibyte",
):
    _UNIT_DIMENSION[_u] = "data"
for _u in ("percent", "basis_point"):
    _UNIT_DIMENSION[_u] = "percent"
for _u in ("celsius", "fahrenheit", "kelvin"):
    _UNIT_DIMENSION[_u] = "temperature"
for _u in (
    "meter_per_second", "kilometer_per_hour", "mile_per_hour", "knot",
    "foot_per_second",
):
    _UNIT_DIMENSION[_u] = "speed"
for _u in _CURRENCIES + ("cent", "pence", "paise", "eurocent"):
    _UNIT_DIMENSION[_u] = "currency"

# Currency sub-units resolve to their own parent currency, never to a
# shared base -- that is the whole point.
_CURRENCY_ROOT: dict[str, str] = {c: c for c in _CURRENCIES}
_CURRENCY_ROOT.update({"cent": "usd", "pence": "gbp", "paise": "inr", "eurocent": "eur"})

# Units that appear in a *time-like* approximation. See the month note.
_APPROXIMATE_UNITS = frozenset({"month", "year"})


def _build_graph() -> dict[str, list[tuple[str, float, float]]]:
    """Adjacency list with both directions of every declared edge.

    The inverse of `y = a*x + b` is `x = (1/a)*y + (-b/a)`, so a single
    declaration gives a fully connected two-way graph and there is no way
    for the forward and reverse factors to drift apart.
    """
    graph: dict[str, list[tuple[str, float, float]]] = {}
    for src, dst, factor, offset in _EDGES:
        graph.setdefault(src, []).append((dst, factor, offset))
        graph.setdefault(dst, []).append((src, 1.0 / factor, -offset / factor))
    # Isolated nodes (bases with no outgoing declaration, e.g. jpy) still
    # need to exist so lookups do not KeyError.
    for unit in _UNIT_DIMENSION:
        graph.setdefault(unit, [])
    return graph


_GRAPH = _build_graph()


def base_unit_of(unit: str) -> str | None:
    """The unit every value in `unit`'s dimension is compared against."""
    dimension = _UNIT_DIMENSION.get(unit)
    if dimension is None:
        return None
    if dimension == "currency":
        return _CURRENCY_ROOT.get(unit, unit)
    return _DIMENSION_BASE[dimension]


def _find_transform(src: str, dst: str) -> tuple[float, float] | None:
    """Breadth-first search for an affine transform from `src` to `dst`.

    Returns `(factor, offset)` such that `value_dst = value_src * factor +
    offset`, or None when the two units are not connected -- which for
    currencies is the normal, intended answer.

    BFS rather than DFS so the composed transform goes through the fewest
    edges, which keeps floating-point error to a minimum: mile -> meter
    via foot -> inch -> centimeter -> meter accumulates four roundings,
    and there is no shorter path, but we should never take a longer one.
    """
    if src == dst:
        return (1.0, 0.0)
    if src not in _GRAPH or dst not in _GRAPH:
        return None

    queue: deque[tuple[str, float, float]] = deque([(src, 1.0, 0.0)])
    seen = {src}
    while queue:
        node, factor, offset = queue.popleft()
        for neighbour, edge_factor, edge_offset in _GRAPH[node]:
            if neighbour in seen:
                continue
            # compose: value_n = (value_src*factor + offset)*ef + eo
            new_factor = factor * edge_factor
            new_offset = offset * edge_factor + edge_offset
            if neighbour == dst:
                return (new_factor, new_offset)
            seen.add(neighbour)
            queue.append((neighbour, new_factor, new_offset))
    return None


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    """Convert between any two connected units. None when unconnected
    (different dimensions, or two different currencies)."""
    from_unit = resolve_unit(from_unit) or from_unit
    to_unit = resolve_unit(to_unit) or to_unit
    transform = _find_transform(from_unit, to_unit)
    if transform is None:
        return None
    factor, offset = transform
    return value * factor + offset


def normalize_unit(value: float, unit: str) -> tuple[float, str]:
    """Convert `value` to the base unit of `unit`'s dimension.

    Returns `(canonical_value, base_unit)`. An unrecognized unit is
    returned unchanged with itself as its base -- silently guessing would
    be worse than admitting we do not know this unit, and the caller can
    still group claims by the literal unit string.

    Currency note: `normalize_unit(5, "gbp")` returns `(5.0, "gbp")`, not
    a dollar figure, by design (see the unit-graph comment).
    """
    canonical = resolve_unit(unit)
    if canonical is None:
        return (float(value), unit)
    base = base_unit_of(canonical)
    if base is None:
        return (float(value), canonical)
    transform = _find_transform(canonical, base)
    if transform is None:
        return (float(value), canonical)
    factor, offset = transform
    return (float(value) * factor + offset, base)


# ===========================================================================
# 2. UNIT ALIASES -- how units are actually written in prose
# ===========================================================================

# canonical unit -> spellings found in the wild. Matching is
# case-insensitive; the handful of aliases where case or context decides
# the meaning are re-checked after matching (see _AMBIGUOUS_ALIASES).
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    # length
    "millimeter": ("mm", "millimeter", "millimeters", "millimetre", "millimetres"),
    "centimeter": ("cm", "centimeter", "centimeters", "centimetre", "centimetres"),
    "meter": ("m", "meter", "meters", "metre", "metres"),
    "kilometer": ("km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"),
    "inch": ("in.", "in", "inch", "inches"),
    "foot": ("ft", "foot", "feet"),
    "yard": ("yd", "yds", "yard", "yards"),
    "mile": ("mi", "mile", "miles"),
    "nautical_mile": ("nmi", "nautical mile", "nautical miles"),
    # mass
    "milligram": ("mg", "milligram", "milligrams"),
    "gram": ("g", "gm", "gram", "grams", "gramme", "grammes"),
    "kilogram": ("kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"),
    "tonne": ("tonne", "tonnes", "metric ton", "metric tons", "t"),
    "ounce": ("oz", "ounce", "ounces"),
    "pound": ("lb", "lbs", "pound", "pounds"),
    # time
    "millisecond": ("ms", "millisecond", "milliseconds", "msec", "msecs"),
    "second": ("sec", "secs", "second", "seconds", "s"),
    "minute": ("min", "mins", "minute", "minutes"),
    "hour": ("hr", "hrs", "hour", "hours", "h"),
    "day": ("day", "days"),
    "week": ("wk", "wks", "week", "weeks"),
    "month": ("mo", "month", "months"),
    "year": ("yr", "yrs", "year", "years"),
    # data -- IEC binary prefixes listed first so they win the
    # longest-alternative-first ordering against "kb"/"mb"
    "kibibyte": ("kib", "kibibyte", "kibibytes"),
    "mebibyte": ("mib", "mebibyte", "mebibytes"),
    "gibibyte": ("gib", "gibibyte", "gibibytes"),
    "tebibyte": ("tib", "tebibyte", "tebibytes"),
    "kilobyte": ("kb", "kilobyte", "kilobytes"),
    "megabyte": ("mb", "megabyte", "megabytes"),
    "gigabyte": ("gb", "gigabyte", "gigabytes"),
    "terabyte": ("tb", "terabyte", "terabytes"),
    "petabyte": ("pb", "petabyte", "petabytes"),
    "byte": ("byte", "bytes"),
    "bit": ("bit", "bits"),
    # percent
    "percent": ("%", "percent", "per cent", "pct", "percentage points",
                "percentage point"),
    "basis_point": ("bps", "basis points", "basis point"),
    # temperature
    "celsius": ("°c", "° c", "degrees celsius", "degree celsius",
                "deg c", "celsius", "centigrade", "c"),
    "fahrenheit": ("°f", "° f", "degrees fahrenheit",
                   "degree fahrenheit", "deg f", "fahrenheit", "f"),
    "kelvin": ("°k", "kelvin", "kelvins", "k"),
    # speed
    "meter_per_second": ("m/s", "mps", "meters per second", "metres per second"),
    "kilometer_per_hour": ("km/h", "kmh", "kph", "km per hour",
                           "kilometers per hour", "kilometres per hour"),
    "mile_per_hour": ("mph", "miles per hour", "mi/h"),
    "knot": ("kn", "knot", "knots"),
    "foot_per_second": ("ft/s", "fps", "feet per second"),
    # currency (symbols are handled separately, before the number, too)
    "usd": ("usd", "us$", "u.s. dollars", "dollar", "dollars", "$"),
    "eur": ("eur", "euro", "euros", "€"),
    "gbp": ("gbp", "pound sterling", "pounds sterling", "quid", "£"),
    "inr": ("inr", "rs.", "rs", "rupee", "rupees", "₹"),
    "jpy": ("jpy", "yen", "¥"),
    "cent": ("cent", "cents"),
    "pence": ("pence",),
    "paise": ("paise",),
}

_ALIAS_TO_UNIT: dict[str, str] = {}
for _unit, _aliases in _UNIT_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_UNIT.setdefault(_alias.lower(), _unit)
# the canonical names themselves are always valid input to normalize_unit
for _unit in _UNIT_DIMENSION:
    _ALIAS_TO_UNIT.setdefault(_unit, _unit)


def resolve_unit(unit: str | None) -> str | None:
    """Alias (or canonical name) -> canonical unit key, or None."""
    if not unit:
        return None
    return _ALIAS_TO_UNIT.get(unit.strip().lower().rstrip("."), _ALIAS_TO_UNIT.get(unit.strip().lower()))


# Aliases whose surface form is genuinely ambiguous in English prose.
# Each is re-examined after the regex matches; see `_guard_unit`.
#   "in"  -- overwhelmingly the preposition ("5 in the morning")
#   "c"/"f"/"k" -- only a temperature when capitalised and standing alone
#   "t"   -- tonne, but also a common variable/letter
#   "m"   -- metre, but "$5m"/"5m users" is five million
#   "h"   -- hour, but also a letter
#   "s"   -- seconds when attached ("250ms", "1.2s"), otherwise a plural
_AMBIGUOUS_ALIASES = frozenset({"in", "in.", "c", "f", "k", "t", "m", "h", "b", "s"})

# Words that, following a bare "in", prove it was a preposition.
_PREPOSITION_FOLLOWERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "his", "her", "its",
    "their", "our", "my", "your", "order", "total", "fact", "which", "some",
    "addition", "part", "terms", "which", "recent", "each", "every", "all",
    "and", "or", "of",
})


# ===========================================================================
# 3. NUMBER PARSING
# ===========================================================================

# Spelled-out numbers: INCLUDED, but only for the closed set below
# (0-20 plus the tens and "dozen"), and only when the number is followed
# by a unit or a plausible noun. Rationale: "three engineers" is a real
# quantity a reader would want compared against "5 engineers", but "one of
# the reasons" and "no one" are not quantities at all, and an open-ended
# spelled-number parser generates far more noise than signal in ordinary
# prose. Spelled numbers get a confidence penalty so a ranking UI can
# de-emphasise them. Ordinals ("third") are excluded: they are positions,
# not amounts.
_SPELLED: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

# Scale words. Indian-English "lakh" (10^5) and "crore" (10^7) are
# included because they appear constantly in South-Asian reporting and
# silently dropping them turns "5 crore users" into "5 users".
_SCALES: dict[str, float] = {
    "hundred": 1e2, "hundreds": 1e2,
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "lakh": 1e5, "lakhs": 1e5,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6,
    "crore": 1e7, "crores": 1e7,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12,
    "dozen": 12.0, "dozens": 12.0,
}

# Currency symbols and codes written BEFORE the number.
_CURRENCY_PREFIX: dict[str, str] = {
    "$": "usd", "us$": "usd", "usd": "usd",
    "£": "gbp", "gbp": "gbp",
    "€": "eur", "eur": "eur",
    "₹": "inr", "inr": "inr", "rs": "inr", "rs.": "inr",
    "¥": "jpy", "jpy": "jpy",
}

# Approximation markers. Their presence widens the contradiction tolerance
# (see `_tolerance_for`) -- "about 5" and "5.2" are the same claim told at
# different precisions, and reporting that as a contradiction would train
# users to ignore the feature.
_APPROX_MARKERS: dict[str, str] = {
    "~": "approx", "≈": "approx", "circa": "approx", "ca.": "approx",
    "about": "approx", "approximately": "approx", "approx": "approx",
    "approx.": "approx", "roughly": "approx", "around": "approx",
    "nearly": "approx", "almost": "approx", "some": "approx",
    "over": "lower_bound", "more than": "lower_bound",
    "greater than": "lower_bound", "at least": "lower_bound",
    "north of": "lower_bound", "upwards of": "lower_bound",
    "in excess of": "lower_bound", "exceeding": "lower_bound",
    "under": "upper_bound", "less than": "upper_bound",
    "fewer than": "upper_bound", "at most": "upper_bound",
    "up to": "upper_bound", "no more than": "upper_bound",
}


def _alt(strings) -> str:
    """Regex alternation, longest-first so "kilometers" is not shadowed by
    "km", with literal spaces relaxed to \\s+ (we run in VERBOSE mode,
    where a raw space is ignored)."""
    parts = sorted(set(strings), key=len, reverse=True)
    return "|".join(re.escape(p).replace("\\ ", r"\s+").replace(" ", r"\s+") for p in parts)


_NUM_CORE = r"""
    (?<![\w.])
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234,567.89
      | \d+(?:\.\d+)?                  # 1234 or 12.5
      | \.\d+                          # .5
    )
    (?:[eE][+-]?\d+)?                  # scientific notation: 1.5e9
"""
_NUMTOK = rf"(?:{_NUM_CORE}|(?<![A-Za-z])(?:{_alt(_SPELLED)})(?![A-Za-z]))"
_SCALE_ALT = rf"(?:{_alt(_SCALES)})(?![A-Za-z])"
_CUR_ALT = rf"(?:{_alt(_CURRENCY_PREFIX)})\.?"
_UNIT_ALT = rf"(?:{_alt(_ALIAS_TO_UNIT)})(?![A-Za-z])"
_APPROX_ALT = rf"(?:{_alt(_APPROX_MARKERS)})"
_NOUN_ALT = r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2}"

# The master scanner. Range alternative first so "10-20 users" is one
# range rather than two counts; the driver loop (`_scan`) can rewind when
# the range turns out to be spurious, which is why we scan with
# `search(pos)` rather than `finditer`.
_MASTER = re.compile(
    rf"""
    (?:(?P<approx>{_APPROX_ALT})\s*)?
    (?:
        # ---------- range ----------
        (?P<between>between\s+)?
        (?:(?P<lo_cur>{_CUR_ALT})\s*)?
        (?P<lo_neg>-\s*|minus\s+)?
        (?P<lo_num>{_NUMTOK})
        (?:\s*(?P<lo_scale>{_SCALE_ALT}))?
        \s*(?P<conn>-|to|and)\s*
        (?:(?P<hi_cur>{_CUR_ALT})\s*)?
        (?P<hi_neg>-\s*|minus\s+)?
        (?P<hi_num>{_NUMTOK})
        (?:\s*(?P<hi_scale>{_SCALE_ALT}))?
        (?:\s*(?P<r_unit>{_UNIT_ALT}))?
        (?:\s+(?P<r_noun>{_NOUN_ALT}))?
      |
        # ---------- single ----------
        (?:(?P<cur>{_CUR_ALT})\s*)?
        (?P<neg>-\s*|minus\s+)?
        (?P<num>{_NUMTOK})
        (?:\s*(?P<scale>{_SCALE_ALT}))?
        (?:\s*(?P<unit>{_UNIT_ALT}))?
        (?:\s+(?P<noun>{_NOUN_ALT}))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Verbs and connectives that separate a quantity from the thing it
# measures. Used by subject extraction on both sides of the number.
_VERB_HINTS = frozenset("""
grew grow grows growing rose rise rises risen fell fall falls fallen
increased increase increases decreased decrease decreases reached reach
reaches hit totaled totalled totals total stood stand stands reported
report reports reporting estimated estimate estimates estimating claims
claimed claim costs cost weighs weigh weighed measures measure measured
takes take took holds hold held spans span sells sell sold raised raise
generated generate posted post recorded record climbed climb dropped drop
jumped jump surged surge declined decline averaged average averages
remains remain stayed stay reaching hitting adding added gained gain lost
lose counts count counted contains contain containing includes include
including according compared versus roughly about approximately around
nearly almost over under least most than per each
""".split())

_CONNECTORS = frozenset({"of", "in", "for", "per", "to", "on", "at", "from",
                         "with", "by", "and", "or", "the", "a", "an", "its",
                         "their", "our", "his", "her", "was", "were", "is",
                         "are", "be", "been", "has", "have", "had", "up",
                         "down", "out", "into", "across", "worth", "as",
                         # temporal determiners: "1.2s last year" is a
                         # duration, and its subject is not "last year"
                         "last", "next", "previous", "prior", "past",
                         "recent", "upcoming", "current", "latest", "same"})

# Words that introduce an IDENTIFIER rather than an amount. "Version 2.0",
# "Figure 3", "Step 2", "Chapter 12" are labels; comparing them across
# documents produces contradictions about nothing at all.
_IDENTIFIER_CUES = frozenset({
    "version", "v", "release", "build", "revision", "rev", "chapter",
    "figure", "fig", "table", "step", "section", "page", "item", "no",
    "number", "issue", "ticket", "pr", "part", "phase", "tier", "level",
    "line", "row", "column", "appendix", "note", "clause", "article",
    "question", "exercise", "slide", "track", "episode", "season",
})
_PRECEDING_WORD = re.compile(r"([A-Za-z][A-Za-z.'\-]*)\s*$")

# Adverbs and temporal modifiers that TRAIL a noun phrase rather than
# belonging to it. "4.2 million users today" is a claim about users, not
# about "users today"; cutting the subject here is what makes it cluster
# with "6.1 million users" from another document. They are deliberately
# NOT in the leading-skip sets: "monthly active users" is a perfectly good
# subject, and the same word can open a noun phrase or close one.
_TRAILING_STOP = frozenset("""
today yesterday tonight now currently already still yet worldwide globally
nationwide daily weekly monthly quarterly annually yearly overall alone far
since ago respectively combined total instead again anyway however though
although despite versus approximately roughly nearly almost onwards
""".split())

_WORD_OR_NUM = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d[\d.,]*")
_BOUNDARY = re.compile(r"[,;:.!?()\"]")
_HAS_DIGIT = re.compile(r"\d")

# Dates are numbers that are not quantities. "2024-01-15" would otherwise
# parse as the range 2024-to-1 (midpoint 1012.5) followed by the count
# -15, and both would be offered to the user as facts. We find date spans
# first and step the scanner straight over them.
_DATE_SPAN = re.compile(
    r"""
      \d{4}-\d{1,2}-\d{1,2}            # 2024-01-15
    | \d{1,2}/\d{1,2}/\d{2,4}          # 15/01/2024
    | \d{4}/\d{1,2}/\d{1,2}
    | \d{1,2}:\d{2}(?::\d{2})?         # 09:30:00
    """,
    re.VERBOSE,
)

# A bare four-digit integer in this window, with no unit and no noun
# attached, is a calendar year rather than a measurement. Years are the
# single most common false positive in ordinary prose, and a "conflict"
# between the years two articles were published is pure noise.
_YEAR_MIN, _YEAR_MAX = 1500, 2100


def _looks_verbal(clean: str, index: int) -> bool:
    """Is this token a verb continuing the sentence rather than part of
    the subject? A past participle or gerund in any position other than
    the first is almost always the predicate -- "three engineers reviewed
    it" describes engineers, not "engineers reviewed". In first position
    the same shape is usually an adjective ("500 automated tests"), so the
    rule only fires later in the phrase. Cheaper and far more general than
    enumerating English verbs, which is a list that is never finished.
    """
    return index > 0 and (clean.endswith("ed") or clean.endswith("ing"))


def _clean(token: str) -> str:
    return re.sub(r"[^a-z'\-]", "", token.lower()).strip("-'")


def _parse_number(raw: str) -> float | None:
    """Digits, separators, scientific notation or a spelled-out word."""
    raw = raw.strip()
    word = raw.lower()
    if word in _SPELLED:
        return float(_SPELLED[word])
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


# ===========================================================================
# 4. EXTRACTION
# ===========================================================================


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, sentence) for each sentence of already-normalized text."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in split_sentences(text):
        index = text.find(sentence, cursor)
        if index < 0:
            index = cursor
        spans.append((index, index + len(sentence), sentence))
        cursor = index + len(sentence)
    if not spans and text:
        spans.append((0, len(text), text))
    return spans


def _sentence_for(spans: list[tuple[int, int, str]], position: int) -> tuple[int, int, str]:
    for start, end, sentence in spans:
        if start <= position < end:
            return (start, end, sentence)
    return spans[-1] if spans else (0, 0, "")


def _trim_noun(raw: str | None) -> list[str]:
    """Keep only the leading run of content words of a captured trailing
    phrase. "users in 2023" -> ["users"]; "in revenue" -> []."""
    if not raw:
        return []
    kept: list[str] = []
    for token in raw.split():
        clean = _clean(token)
        if (not clean or clean in STOPWORDS or clean in _VERB_HINTS
                or clean in _CONNECTORS or clean in _TRAILING_STOP
                or _looks_verbal(clean, len(kept))):
            break
        kept.append(clean)
    return kept


def _words_after(text: str, start: int, sentence_end: int) -> list[str]:
    """Content words following the match, stopping at punctuation, at a
    number (a new quantity is a new subject), or after three words."""
    tail = text[start:sentence_end]
    boundary = _BOUNDARY.search(tail)
    if boundary:
        tail = tail[: boundary.start()]
    words: list[str] = []
    started = False
    for token in _WORD_OR_NUM.findall(tail):
        if token[0].isdigit():
            break
        if _HAS_DIGIT.search(token):
            # "FY2024", "Q3", "H1" -- a reporting period, never the thing
            # being measured. Letting one through makes every figure in a
            # filing cluster under "fy".
            if started:
                break
            continue
        clean = _clean(token)
        if not clean:
            continue
        if not started and (clean in _CONNECTORS or clean in STOPWORDS or clean in _VERB_HINTS):
            continue  # skip the leading "of"/"in"/"reached"
        if started and (clean in _CONNECTORS or clean in STOPWORDS
                        or clean in _VERB_HINTS or clean in _TRAILING_STOP
                        or _looks_verbal(clean, len(words))):
            break
        if resolve_unit(clean) and not words:
            continue  # a stray unit word, not the subject
        started = True
        words.append(clean)
        if len(words) == 3:
            break
    return words


def _words_before(text: str, sentence_start: int, end: int) -> list[str]:
    """Content words preceding the match, read right-to-left: this is how
    "annual revenue of $3M" and "the fleet covers 400 km" both give a
    usable subject."""
    head = text[sentence_start:end]
    tokens = _WORD_OR_NUM.findall(head)
    words: list[str] = []
    started = False
    for token in reversed(tokens):
        if token[0].isdigit() or _HAS_DIGIT.search(token):
            if started:
                break
            continue
        clean = _clean(token)
        if not clean:
            continue
        # Reading right-to-left, the words nearest the number are the ones
        # that attach it to its subject: a verb ("revenue *grew* 25%"), a
        # preposition ("*of* $3M") or the unit of the previous quantity
        # ("150 lbs (68 kg)"). Skip past all of those; stop at the first
        # real content word and take up to three.
        if not started and (clean in _CONNECTORS or clean in STOPWORDS
                            or clean in _VERB_HINTS
                            or clean.endswith(("ed", "ing"))
                            or resolve_unit(clean)):
            continue
        if started and (clean in _CONNECTORS or clean in STOPWORDS or clean in _VERB_HINTS):
            break
        started = True
        words.append(clean)
        if len(words) == 3:
            break
    return list(reversed(words))


def _guard_unit(alias: str, raw_alias: str, attached: bool, following: str,
                number_raw: str = "") -> str | None:
    """Second-guess the ambiguous unit spellings.

    The regex is case-insensitive and context-free, which is right for
    "km" and disastrous for "in", "C" and "m". Rather than complicate the
    pattern with lookarounds nobody can read, we match permissively and
    reject here, where the rule can be explained.
    """
    canonical = resolve_unit(alias)
    if canonical is None:
        return None
    key = alias.lower()
    if key not in _AMBIGUOUS_ALIASES:
        return canonical

    next_word = _clean(following.split()[0]) if following.split() else ""

    if key in ("in", "in."):
        # "36 in wide" is inches; "5 in the morning" is a preposition.
        if key == "in" and next_word in _PREPOSITION_FOLLOWERS:
            return None
        return "inch"
    if key in ("c", "f"):
        # Only the capital letter is a temperature: "25 C" yes, "25 c" no.
        return canonical if raw_alias.isupper() else None
    if key == "k":
        # "300K users" is three hundred thousand; "300 K" is kelvin. The
        # scale table already claimed the attached form, so reaching here
        # with `attached` true means a stray letter, not a temperature.
        return "kelvin" if (raw_alias.isupper() and not attached) else None
    if key == "t":
        return "tonne" if not raw_alias.isupper() else None
    if key == "h":
        return "hour"
    if key == "m":
        return "meter"
    if key == "s":
        # "250ms"/"1.2s" is how engineers write durations, but a detached
        # "s" is a plural marker or a stray letter, and "1990s" is a
        # decade. Attached, and not a bare year, or it is not a unit.
        if not attached:
            return None
        digits = number_raw.replace(",", "")
        if digits.isdigit() and len(digits) == 4:
            return None
        return "second"
    if key == "b":
        return None
    return canonical


def _apply_scale_guard(scale_raw: str, currency: str | None, noun_tokens: list[str]) -> tuple[float | None, str | None]:
    """Disambiguate the single-letter scale abbreviations.

    "$5m" is five million dollars; "5m" on its own is five metres; "5m
    users" is five million users. The tell is (a) a currency, (b) the
    capital letter, or (c) a plural noun following -- nobody measures
    "users" in metres. Returns (multiplier, unit_override); a unit
    override means "that was not a scale word at all".
    """
    lowered = scale_raw.lower()
    if lowered not in ("m", "b", "k"):
        return (_SCALES[lowered], None)

    if lowered == "k":
        return (1e3, None)  # "k" is never a unit on its own
    if lowered == "b":
        return (1e9, None)  # "b" is not a unit we recognise either
    # lowered == "m"
    if currency or scale_raw.isupper():
        return (1e6, None)
    if noun_tokens and noun_tokens[0].endswith("s"):
        return (1e6, None)
    return (None, "meter")


def extract_quantities(text: str) -> list[dict]:
    """Every quantity in `text`, left to right.

    See the module docstring for the offset convention. Each result is a
    plain dict (JSON-serialisable, no classes to import elsewhere) with
    the fields documented in the feature spec, plus three extras the
    contradiction stage needs: `approximate`, `range` and `marker`.
    """
    text = normalize(text or "")
    if not text:
        return []

    spans = _sentence_spans(text)
    date_spans = [(m.start(), m.end()) for m in _DATE_SPAN.finditer(text)]
    results: list[dict] = []
    pos = 0
    length = len(text)

    while pos < length:
        match = _MASTER.search(text, pos)
        if match is None:
            break

        # Step over dates and clock times wholesale rather than trying to
        # reinterpret their pieces as quantities.
        skipped = False
        for date_start, date_end in date_spans:
            if date_start <= match.start() < date_end:
                pos = date_end
                skipped = True
                break
        if skipped:
            continue

        is_range = match.group("hi_num") is not None
        consumed_end = match.end()

        if is_range:
            connector = (match.group("conn") or "").lower()
            if connector == "and" and not match.group("between"):
                # "10 and 20" is only a range when "between" announced it;
                # otherwise it is a list ("versions 10 and 20"). Rewind and
                # re-read just the first number as a standalone quantity,
                # which is why this loop scans with search(pos) instead of
                # finditer -- finditer cannot give the position back.
                sub = _MASTER.match(text, match.start(), match.start("conn"))
                if sub is not None and sub.group("num"):
                    item, consumed_end = _build_single(text, sub, spans)
                else:
                    item, consumed_end = None, match.end("lo_num")
            else:
                item, consumed_end = _build_range(text, match, spans)
        else:
            item, consumed_end = _build_single(text, match, spans)

        if item is not None:
            results.append(item)

        pos = max(consumed_end, match.start() + 1)

    return results


def _prefix_marker(match: re.Match) -> tuple[str | None, str | None]:
    raw = match.group("approx")
    if not raw:
        return (None, None)
    key = re.sub(r"\s+", " ", raw.strip().lower())
    return (key, _APPROX_MARKERS.get(key) or _APPROX_MARKERS.get(key.rstrip(".")))


def _resolve_unit_group(text: str, match: re.Match, unit_group: str,
                        number_raw: str, currency: str | None) -> tuple[str | None, int | None]:
    """Run the ambiguity guard over a matched unit group.

    When a currency symbol opened the quantity, a trailing non-currency
    unit is always a misread: "$3.2 billion in revenue" is dollars, and
    the "in" is the preposition, not inches. Dropping it here (rather
    than after the fact) also keeps `end` from swallowing the word.
    """
    raw_alias = match.group(unit_group)
    if not raw_alias:
        return (None, None)
    alias_start = match.start(unit_group)
    alias_end = match.end(unit_group)
    attached = alias_start > 0 and not text[alias_start - 1].isspace()
    following = text[alias_end: alias_end + 40]
    canonical = _guard_unit(raw_alias, raw_alias, attached, following, number_raw)
    if canonical is None:
        return (None, None)
    if currency and _UNIT_DIMENSION.get(canonical) != "currency":
        return (None, None)
    return (canonical, alias_end)


def _finish(text: str, spans, start: int, end: int, value: float,
            unit: str | None, unit_text: str, noun_tokens: list[str],
            marker: str | None, is_range: bool, spelled: bool,
            reinterpreted: bool, extra: dict) -> dict:
    """Assemble one result dict: canonicalize, find the subject, score."""
    sentence_start, sentence_end, sentence = _sentence_for(spans, start)

    if unit:
        canonical_value, base = normalize_unit(value, unit)
        dimension = _UNIT_DIMENSION.get(unit)
    else:
        canonical_value, base = float(value), None
        dimension = "count"

    # --- subject ---
    if noun_tokens:
        subject_words = noun_tokens
    else:
        subject_words = _words_after(text, end, sentence_end)
        if not subject_words:
            subject_words = _words_before(text, sentence_start, start)
    subject = " ".join(subject_words)

    # --- confidence ---
    # Starts at 1.0 for "a digit string with an explicit unit and an
    # obvious noun attached" and is docked for every way the reading could
    # be wrong. The numbers are deliberately coarse: this is a ranking
    # signal for the UI, not a probability.
    confidence = 1.0
    if marker:
        confidence -= 0.12
    if is_range:
        confidence -= 0.12
    if spelled:
        confidence -= 0.08
    if unit is None:
        confidence -= 0.10
    if not subject:
        confidence -= 0.20
    if reinterpreted:
        confidence -= 0.05
    confidence = round(max(0.15, min(1.0, confidence)), 2)

    item = {
        "text": text[start:end],
        "start": start,
        "end": end,
        "value": float(value),
        "unit": unit,
        "unit_text": unit_text.strip(),
        "dimension": dimension,
        "canonical_value": float(canonical_value),
        "base_unit": base,
        "subject": subject,
        "sentence": sentence,
        "confidence": confidence,
        # extras used by the contradiction stage
        "approximate": bool(marker) or is_range or (unit in _APPROXIMATE_UNITS),
        "range": is_range,
        "marker": marker,
    }
    item.update(extra)
    return item


def _build_single(text: str, match: re.Match, spans) -> tuple[dict | None, int]:
    raw_marker, marker = _prefix_marker(match)
    raw_num = match.group("num")
    value = _parse_number(raw_num)
    if value is None:
        return (None, match.end("num"))
    spelled = raw_num.lower() in _SPELLED

    if match.group("neg"):
        value = -value

    currency = None
    if match.group("cur"):
        currency = _CURRENCY_PREFIX.get(match.group("cur").strip().lower().rstrip("."))

    noun_tokens = _trim_noun(match.group("noun"))

    # --- scale ---
    multiplier = 1.0
    unit_override = None
    reinterpreted = False
    scale_raw = match.group("scale")
    if scale_raw:
        multiplier, unit_override = _apply_scale_guard(scale_raw, currency, noun_tokens)
        if multiplier is None:
            multiplier = 1.0
            reinterpreted = True
        value *= multiplier

    # --- unit ---
    unit = unit_override
    unit_end = match.end("scale") if scale_raw else match.end("num")
    if unit is None:
        resolved, alias_end = _resolve_unit_group(text, match, "unit", raw_num, currency)
        if resolved:
            unit = resolved
            unit_end = alias_end
    if unit is None and currency:
        # "$5 million" -- the symbol is the unit.
        unit = currency
    elif currency and _UNIT_DIMENSION.get(unit) != "currency":
        unit = currency

    start = match.start("cur") if match.group("cur") else (
        match.start("neg") if match.group("neg") else match.start("num"))
    if raw_marker:
        start = match.start("approx")

    # Where the match really ends: the noun is only part of the quantity
    # when there is no unit (bare counts like "4.2 million users").
    if unit is not None:
        end = unit_end
        noun_tokens = []
    elif noun_tokens:
        # trim back to the last kept noun token
        noun_raw = match.group("noun") or ""
        noun_start = match.start("noun")
        kept = len(" ".join(noun_raw.split()[: len(noun_tokens)]))
        end = noun_start + kept
    else:
        end = unit_end

    # Spelled-out numbers only count when something makes them a quantity.
    if spelled and unit is None and not noun_tokens:
        return (None, match.end("num"))

    # Calendar years and decades are not measurements -- see
    # _YEAR_MIN/_YEAR_MAX. "1990s saw 3 recessions" must yield only the 3.
    decade = text[match.end("num"): match.end("num") + 1].lower() == "s"
    if (unit is None and not scale_raw and raw_num.isdigit()
            and len(raw_num) == 4 and _YEAR_MIN <= value <= _YEAR_MAX
            and (not noun_tokens or decade)):
        return (None, end)

    # Identifiers ("Version 2.0", "Figure 3") are labels, not amounts.
    if unit is None:
        preceding = _PRECEDING_WORD.search(text[:match.start("num")])
        if preceding and _clean(preceding.group(1)).rstrip(".") in _IDENTIFIER_CUES:
            return (None, end)

    unit_text = text[match.end("num"): end].strip()
    if currency and not unit_text:
        unit_text = match.group("cur")

    item = _finish(text, spans, start, end, value, unit, unit_text, noun_tokens,
                   marker, False, spelled, reinterpreted, {})
    return (item, end)


def _build_range(text: str, match: re.Match, spans) -> tuple[dict | None, int]:
    """A range ("10-20 users", "between $2M and $4M") collapses to its
    midpoint, flagged approximate.

    WHY the midpoint: downstream everything compares single canonical
    values, and a range is a claim about a central tendency with stated
    uncertainty. Keeping `value_low`/`value_high` on the dict means a
    caller that wants interval logic still has it, while the default
    behaviour -- midpoint plus a widened tolerance -- does the right thing
    for the common case of "10-20%" versus "15%".
    """
    lo = _parse_number(match.group("lo_num"))
    hi = _parse_number(match.group("hi_num"))
    if lo is None or hi is None:
        return (None, match.end("lo_num"))

    lo_spelled = match.group("lo_num").lower() in _SPELLED
    hi_spelled = match.group("hi_num").lower() in _SPELLED

    if match.group("lo_neg"):
        lo = -lo
    if match.group("hi_neg"):
        hi = -hi

    currency = None
    for group in ("lo_cur", "hi_cur"):
        raw = match.group(group)
        if raw:
            currency = _CURRENCY_PREFIX.get(raw.strip().lower().rstrip("."))
            break

    noun_tokens = _trim_noun(match.group("r_noun"))

    hi_scale_raw = match.group("hi_scale")
    lo_scale_raw = match.group("lo_scale")
    reinterpreted = False
    hi_multiplier = 1.0
    unit_override = None
    if hi_scale_raw:
        hi_multiplier, unit_override = _apply_scale_guard(hi_scale_raw, currency, noun_tokens)
        if hi_multiplier is None:
            hi_multiplier, reinterpreted = 1.0, True
    lo_multiplier = hi_multiplier  # "between 10 and 20 million" -> both scale
    if lo_scale_raw:
        explicit, override_lo = _apply_scale_guard(lo_scale_raw, currency, noun_tokens)
        if explicit is None:
            explicit, reinterpreted = 1.0, True
            unit_override = unit_override or override_lo
        lo_multiplier = explicit

    lo *= lo_multiplier
    hi *= hi_multiplier
    if hi < lo:
        lo, hi = hi, lo

    unit = unit_override
    end = match.end("hi_scale") if hi_scale_raw else match.end("hi_num")
    if unit is None:
        resolved, alias_end = _resolve_unit_group(
            text, match, "r_unit", match.group("hi_num"), currency)
        if resolved:
            unit, end = resolved, alias_end
    if unit is None and currency:
        unit = currency
    elif currency:
        unit = currency

    if unit is not None:
        noun_tokens = []
    elif noun_tokens:
        noun_raw = match.group("r_noun") or ""
        end = match.start("r_noun") + len(" ".join(noun_raw.split()[: len(noun_tokens)]))

    if (lo_spelled or hi_spelled) and unit is None and not noun_tokens:
        return (None, match.end("lo_num"))

    start = match.start("approx") if match.group("approx") else (
        match.start("between") if match.group("between") else
        (match.start("lo_cur") if match.group("lo_cur") else
         (match.start("lo_neg") if match.group("lo_neg") else match.start("lo_num"))))

    _, marker = _prefix_marker(match)
    midpoint = (lo + hi) / 2.0
    unit_text = text[match.end("hi_num"): end].strip()
    if currency and not unit_text:
        unit_text = match.group("lo_cur") or match.group("hi_cur") or ""

    item = _finish(text, spans, start, end, midpoint, unit, unit_text, noun_tokens,
                   marker, True, lo_spelled or hi_spelled, reinterpreted,
                   {"value_low": float(lo), "value_high": float(hi)})
    if unit:
        item["canonical_low"] = normalize_unit(lo, unit)[0]
        item["canonical_high"] = normalize_unit(hi, unit)[0]
    else:
        item["canonical_low"], item["canonical_high"] = float(lo), float(hi)
    return (item, end)


# ===========================================================================
# 5. CLUSTERING CLAIMS
# ===========================================================================

def _subject_key(subject: str) -> tuple[str, ...]:
    """Stemmed, stopword-free, order-insensitive identity of a subject, so
    "monthly active users" and "active monthly users" collide."""
    return tuple(sorted(set(content_tokens(subject))))


def _subjects_match(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Are two subject keys about the same measurable thing?

    Three rules, in order of confidence:
      1. One is a subset of the other -- "users" vs "active users". A
         document that says "users" and one that says "monthly users" are
         very probably talking about the same headline number, and NOT
         merging them is the failure we are trying to avoid.
      2. Jaccard overlap >= 0.5 -- "annual recurring revenue" vs
         "recurring revenue".
      3. Single-token subjects within one edit -- "subscriber" vs
         "subscribers" survive stemming, but "employee"/"employes"
         (typos, OCR) do not.
    """
    if not a or not b:
        return False
    set_a, set_b = set(a), set(b)
    if set_a <= set_b or set_b <= set_a:
        return True
    overlap = len(set_a & set_b) / len(set_a | set_b)
    if overlap >= 0.5:
        return True
    if len(set_a) == 1 and len(set_b) == 1:
        return levenshtein(next(iter(set_a)), next(iter(set_b)), max_distance=1) <= 1
    return False


# Tolerances, and the reasoning for each number.
#
# _BASE_TOLERANCE (5%): published figures get rounded, restated, and
# recomputed on slightly different date cutoffs. "4.2 million" and "4.18
# million" are the same claim. Below 5% we say nothing; flagging rounding
# as contradiction is how a feature like this loses the user's trust in
# the first session.
_BASE_TOLERANCE = 0.05
# _APPROX_TOLERANCE (25%): when either side hedged ("about 5", "over
# 5", "10-20"), the author explicitly declined to commit to a precise
# value. "about 5" against 5.9 is consistent reporting, not a conflict.
_APPROX_TOLERANCE = 0.25
# _MAJOR_RATIO (2.0): a claim that is double another is not a revision or
# a rounding, it is a different fact. Between the tolerance and 2x we say
# "minor" -- worth a glance, probably a different time period or scope.
_MAJOR_RATIO = 2.0
# Percentages additionally need an absolute floor: 0.1% vs 0.3% is 3x but
# is two decimal places of the same near-zero figure, and reporting it as
# a MAJOR conflict would be absurd.
_PERCENT_ABSOLUTE_FLOOR = 1.0


def _tolerance_for(claims: list[dict]) -> float:
    return _APPROX_TOLERANCE if any(c.get("approximate") for c in claims) else _BASE_TOLERANCE


def _spread_ratio(low: float, high: float) -> float:
    """max/min on canonical values, defined for the awkward cases.

    With a positive minimum this is the plain ratio. When the minimum is
    zero or negative a ratio is meaningless (or infinite), so we fall back
    to a normalized absolute gap expressed on the same 1.0-is-agreement
    scale, which keeps every downstream comparison a single rule.
    """
    if low > 0:
        return high / low
    scale = max(abs(low), abs(high), 1e-9)
    return 1.0 + abs(high - low) / scale


def _severity(ratio: float, tolerance: float, dimension: str | None,
              low: float, high: float) -> tuple[bool, str]:
    if dimension == "percent" and abs(high - low) <= _PERCENT_ABSOLUTE_FLOOR:
        return (False, "none")
    if ratio <= 1.0 + tolerance:
        return (False, "none")
    if ratio < _MAJOR_RATIO:
        return (True, "minor")
    return (True, "major")


def group_claims(quantities: list[dict]) -> list[dict]:
    """Cluster quantities that describe the same measurable thing.

    Two quantities may only join a group when they share a dimension AND a
    base unit. The base-unit condition is what keeps dollars away from
    euros (each currency is its own base -- see the unit graph note) while
    still letting kilometres meet miles, which both reduce to metres.

    Subjectless quantities are dropped: a number we cannot attribute to
    anything cannot be contradicted by another number, and including them
    produces a "group" of unrelated figures that merely share a unit.
    """
    buckets: dict[tuple, list[dict]] = {}
    for quantity in quantities:
        key_tokens = _subject_key(quantity.get("subject", ""))
        if not key_tokens:
            continue
        base = quantity.get("base_unit")
        dimension = quantity.get("dimension")
        buckets.setdefault((dimension, base, key_tokens), []).append(quantity)

    # Agglomerate buckets that share a dimension+base and whose subjects
    # match under `_subjects_match`. Single-link clustering: cheap, and
    # for a few hundred quantities per corpus its O(n^2) is irrelevant.
    by_axis: dict[tuple, list[tuple[tuple[str, ...], list[dict]]]] = {}
    for (dimension, base, key_tokens), items in buckets.items():
        by_axis.setdefault((dimension, base), []).append((key_tokens, items))

    groups: list[dict] = []
    for (dimension, base), entries in by_axis.items():
        merged: list[tuple[set[str], list[dict]]] = []
        for key_tokens, items in sorted(entries, key=lambda e: (-len(e[1]), e[0])):
            target = None
            for candidate_key, candidate_items in merged:
                if _subjects_match(tuple(sorted(candidate_key)), key_tokens):
                    target = (candidate_key, candidate_items)
                    break
            if target is None:
                merged.append((set(key_tokens), list(items)))
            else:
                target[0].update(key_tokens)
                target[1].extend(items)

        for _key, claims in merged:
            groups.append(_summarize_group(claims, dimension, base))

    # Stable, useful ordering: conflicts first, then biggest disagreement.
    groups.sort(key=lambda g: (not g["conflicting"], -g["spread_ratio"], g["subject"]))
    return groups


def _display_subject(claims: list[dict]) -> str:
    """The most common surface form, breaking ties toward the shortest --
    a group label should be the plainest thing any source called it."""
    counts = Counter(c.get("subject", "") for c in claims if c.get("subject"))
    if not counts:
        return ""
    best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
    return best[0]


def _summarize_group(claims: list[dict], dimension: str | None, base: str | None) -> dict:
    claims = sorted(claims, key=lambda c: (c.get("document_id", 0), c.get("start", 0)))
    values = [c["canonical_value"] for c in claims]
    low, high = min(values), max(values)
    ratio = _spread_ratio(low, high)

    if len(claims) < 2:
        conflicting, severity = False, "none"
    else:
        tolerance = _tolerance_for(claims)
        conflicting, severity = _severity(ratio, tolerance, dimension, low, high)

    return {
        "subject": _display_subject(claims),
        "dimension": dimension,
        "unit": base,
        "claims": claims,
        "min": low,
        "max": high,
        "spread_ratio": round(ratio, 4) if math.isfinite(ratio) else float("inf"),
        "conflicting": conflicting,
        "severity": severity,
    }


# ===========================================================================
# 6. THE PIPELINE
# ===========================================================================

def find_contradictions(documents: list[dict]) -> dict:
    """Extract -> normalize -> cluster -> flag, over a document set.

    `documents` is a list of {"id", "title", "text"}. Every claim carries
    its `document_id` and `document_title` so the UI can say *where* the
    disagreement is, which is the only form of this feature that is
    actually actionable.
    """
    all_quantities: list[dict] = []
    scanned = 0
    for document in documents or []:
        text = document.get("text") or ""
        scanned += 1
        for quantity in extract_quantities(text):
            quantity = dict(quantity)
            quantity["document_id"] = document.get("id")
            quantity["document_title"] = document.get("title", "")
            all_quantities.append(quantity)

    groups = [g for g in group_claims(all_quantities) if len(g["claims"]) >= 2]
    return {
        "groups": groups,
        "conflicts": [g for g in groups if g["conflicting"]],
        "total_quantities": len(all_quantities),
        "documents_scanned": scanned,
    }


# ===========================================================================
# 7. SELF-TEST
# ===========================================================================

if __name__ == "__main__":  # pragma: no cover
    PASSED = 0
    FAILED = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        global PASSED, FAILED
        if condition:
            PASSED += 1
            print(f"PASS  {name}")
        else:
            FAILED += 1
            print(f"FAIL  {name}  {detail}")

    def first(text: str) -> dict:
        items = extract_quantities(text)
        return items[0] if items else {}

    def close(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(b))

    print("--- number & unit extraction ---")

    q = first("The platform has 4.2 million users today.")
    check("4.2 million users -> value", close(q.get("value", 0), 4_200_000.0), str(q))
    check("4.2 million users -> subject", q.get("subject") == "users", str(q.get("subject")))
    check("4.2 million users -> text", q.get("text") == "4.2 million users", str(q.get("text")))
    check("4.2 million users -> dimension count", q.get("dimension") == "count")

    q = first("Total headcount reached 1,234,567 employees.")
    check("thousand separators", close(q.get("value", 0), 1234567.0), str(q))

    q = first("Revenue of $3M was reported.")
    check("$3M value", close(q.get("value", 0), 3_000_000.0), str(q))
    check("$3M unit", q.get("unit") == "usd", str(q.get("unit")))
    check("$3M subject from the left", q.get("subject") == "revenue", str(q.get("subject")))

    q = first("The round raised 3bn in fresh capital.")
    check("3bn scale", close(q.get("value", 0), 3e9), str(q))

    q = first("We shipped 12k units last quarter.")
    check("12k scale", close(q.get("value", 0), 12000.0), str(q))

    q = first("Churn sits between 10 and 20 percent.")
    check("range midpoint", close(q.get("value", 0), 15.0), str(q))
    check("range flagged", q.get("range") is True and q.get("approximate") is True)
    check("range unit percent", q.get("unit") == "percent", str(q.get("unit")))

    q = first("Latency is 10-20 ms under load.")
    check("hyphen range", close(q.get("value", 0), 15.0), str(q))
    check("hyphen range unit", q.get("unit") == "millisecond", str(q.get("unit")))

    q = first("There are about 5 outages per month.")
    check("about 5 -> value", close(q.get("value", 0), 5.0), str(q))
    check("about 5 -> approximate", q.get("approximate") is True)
    check("about 5 -> marker", q.get("marker") == "approx", str(q.get("marker")))

    q = first("Roughly 5 engineers maintain it.")
    check("roughly marker", q.get("marker") == "approx", str(q.get("marker")))

    q = first("~5 GB of logs are produced daily.")
    check("tilde approximation", q.get("marker") == "approx" and close(q.get("value", 0), 5.0), str(q))
    check("GB unit", q.get("unit") == "gigabyte", str(q.get("unit")))

    q = first("At least 5 regions are affected.")
    check("at least -> lower bound", q.get("marker") == "lower_bound", str(q.get("marker")))

    q = first("Over 5 million downloads were recorded.")
    check("over 5 million", close(q.get("value", 0), 5e6) and q.get("marker") == "lower_bound", str(q))

    q = first("Conversion improved to 45% this month.")
    check("percent sign", q.get("unit") == "percent" and close(q.get("value", 0), 45.0), str(q))

    q = first("Margins are 12.5 percent.")
    check("percent word", q.get("unit") == "percent" and close(q.get("value", 0), 12.5), str(q))

    q = first("The fee is 100 dollars.")
    check("currency after the number", q.get("unit") == "usd" and close(q.get("value", 0), 100.0), str(q))

    q = first("It costs €250 per seat.")
    check("euro symbol", q.get("unit") == "eur" and close(q.get("value", 0), 250.0), str(q))

    q = first("The grant was ₹50 lakh.")
    check("rupee + lakh", q.get("unit") == "inr" and close(q.get("value", 0), 5_000_000.0), str(q))

    q = first("£2.5bn was written off.")
    check("pound + bn", q.get("unit") == "gbp" and close(q.get("value", 0), 2.5e9), str(q))

    q = first("The anomaly was -5 degrees celsius overnight.")
    check("negative number", close(q.get("value", 0), -5.0), str(q))
    check("negative celsius canonical", close(q.get("canonical_value", 0), 268.15), str(q))

    q = first("The constant is 1.5e9 joules.")
    check("scientific notation", close(q.get("value", 0), 1.5e9), str(q))

    q = first("Three engineers reviewed the change.")
    check("spelled-out number", close(q.get("value", 0), 3.0), str(q))
    check("spelled-out subject", q.get("subject") == "engineers", str(q.get("subject")))
    check("spelled-out confidence penalty", q.get("confidence", 1.0) < 1.0, str(q.get("confidence")))

    check("bare spelled number rejected", extract_quantities("One of the reasons is speed.") == [],
          str(extract_quantities("One of the reasons is speed.")))

    q = first("The trail is 26.2 miles long.")
    check("miles", q.get("unit") == "mile" and close(q.get("canonical_value", 0), 42164.8128, 1e-6), str(q))

    q = first("The package weighs 2.5 kg.")
    check("kg canonical", q.get("unit") == "kilogram" and close(q.get("canonical_value", 0), 2.5), str(q))

    q = first("The meeting ran 90 minutes.")
    check("minutes canonical", close(q.get("canonical_value", 0), 5400.0), str(q))

    q = first("Top speed is 120 km/h on the straight.")
    check("km/h", q.get("unit") == "kilometer_per_hour" and close(q.get("canonical_value", 0), 33.3333333, 1e-6), str(q))

    q = first("He arrived at 5 in the morning.")
    check("'in' guarded as preposition", q.get("unit") != "inch", str(q.get("unit")))

    q = first("The board is 36 in wide.")
    check("'in' accepted as inches", q.get("unit") == "inch", str(q.get("unit")))

    q = first("The rope is 5m long.")
    check("bare 5m -> metres", q.get("unit") == "meter" and close(q.get("value", 0), 5.0), str(q))

    q = first("We now serve $5m in annual revenue.")
    check("$5m -> million", close(q.get("value", 0), 5e6), str(q))

    q = first("The service crossed 5m subscribers.")
    check("5m subscribers -> million", close(q.get("value", 0), 5e6), str(q))

    qs = extract_quantities("Storage grew from 1 TB to 4 TB over the year.")
    check("multiple quantities in one sentence", len(qs) >= 2, str([x['text'] for x in qs]))

    check("empty input", extract_quantities("") == [])
    check("no numbers", extract_quantities("No numbers here at all.") == [])

    qs = extract_quantities("Version 2.0 shipped on 2024-01-15 with 3 fixes.")
    check("dates and version numbers skipped",
          [x["text"] for x in qs] == ["3 fixes"], str([x["text"] for x in qs]))

    qs = extract_quantities("The 1990s saw 3 recessions.")
    check("decades skipped", [x["text"] for x in qs] == ["3 recessions"],
          str([x["text"] for x in qs]))

    qs = extract_quantities("In 1999 the company had 500 employees.")
    check("bare calendar year skipped", [x["text"] for x in qs] == ["500 employees"],
          str([x["text"] for x in qs]))

    qs = extract_quantities("Our p99 latency is 250ms, down from 1.2s last year.")
    check("attached ms and s units", [x["unit"] for x in qs] == ["millisecond", "second"],
          str([(x["text"], x["unit"]) for x in qs]))
    check("both latency claims share a subject",
          all(x["subject"] == "latency" for x in qs), str([x["subject"] for x in qs]))

    q = first("The CEO said revenue grew 25% to $3.2 billion in FY2024.")
    check("percent before a currency", q.get("unit") == "percent" and q.get("subject") == "revenue",
          str(q))
    q = extract_quantities("The CEO said revenue grew 25% to $3.2 billion in FY2024.")[1]
    check("currency not extended by a stray 'in'", q["text"] == "$3.2 billion", q["text"])
    check("reporting period is not a subject", q["subject"] == "revenue", q["subject"])

    print("--- unit graph / normalize_unit ---")

    check("length: km -> m", close(normalize_unit(1, "km")[0], 1000.0))
    check("length: mile -> m", close(normalize_unit(1, "mile")[0], 1609.344))
    check("length: inch -> m", close(normalize_unit(1, "inch")[0], 0.0254))
    check("length base name", normalize_unit(1, "mm")[1] == "meter")
    check("mass: lb -> kg", close(normalize_unit(1, "lb")[0], 0.45359237, 1e-9))
    check("mass: tonne -> kg", close(normalize_unit(1, "tonne")[0], 1000.0))
    check("mass: oz -> kg", close(normalize_unit(16, "oz")[0], 0.45359237, 1e-9))
    check("time: hour -> s", close(normalize_unit(1, "hour")[0], 3600.0))
    check("time: week -> s", close(normalize_unit(1, "week")[0], 604800.0))
    check("time: year -> s", close(normalize_unit(1, "year")[0], 31557600.0))
    check("time: month approx", close(normalize_unit(12, "month")[0], normalize_unit(1, "year")[0]))
    check("data: GB -> byte", close(normalize_unit(1, "GB")[0], 1e9))
    check("data: bit -> byte", close(normalize_unit(8, "bit")[0], 1.0))
    check("percent: bps -> percent", close(normalize_unit(100, "bps")[0], 1.0))
    check("speed: mph -> m/s", close(normalize_unit(1, "mph")[0], 0.44704))
    check("speed: knot -> m/s", close(normalize_unit(1, "knot")[0], 0.5144444, 1e-6))

    # KiB vs KB -- the distinction that silently corrupts storage claims.
    kb = normalize_unit(1, "KB")[0]
    kib = normalize_unit(1, "KiB")[0]
    check("KB is decimal (1000)", close(kb, 1000.0), str(kb))
    check("KiB is binary (1024)", close(kib, 1024.0), str(kib))
    check("KiB != KB", kib != kb)
    check("GiB is 2^30", close(normalize_unit(1, "GiB")[0], 1073741824.0), str(normalize_unit(1, "GiB")))
    check("TiB is 2^40", close(normalize_unit(1, "TiB")[0], 1099511627776.0))

    # Temperature: affine, and a full round trip.
    check("0 C -> 273.15 K", close(normalize_unit(0, "celsius")[0], 273.15))
    check("100 C -> 373.15 K", close(normalize_unit(100, "celsius")[0], 373.15))
    check("32 F -> 273.15 K", close(normalize_unit(32, "fahrenheit")[0], 273.15))
    check("212 F -> 373.15 K", close(normalize_unit(212, "fahrenheit")[0], 373.15))
    check("-40 C == -40 F", close(normalize_unit(-40, "celsius")[0], normalize_unit(-40, "fahrenheit")[0]))
    check("C is NOT multiplicative", not close(normalize_unit(100, "celsius")[0],
                                               10 * normalize_unit(10, "celsius")[0]))
    c0 = 21.5
    f = convert(c0, "celsius", "fahrenheit")
    k = convert(f, "fahrenheit", "kelvin")
    back = convert(k, "kelvin", "celsius")
    check("C -> F", close(f, 70.7, 1e-9), str(f))
    check("C -> F -> K", close(k, 294.65, 1e-9), str(k))
    check("C -> F -> K -> C round trip", close(back, c0, 1e-9), str(back))

    # Currencies: isolated by design.
    check("usd base is usd", normalize_unit(5, "usd") == (5.0, "usd"))
    check("gbp base is gbp", normalize_unit(5, "gbp") == (5.0, "gbp"))
    check("no usd->eur conversion", convert(5, "usd", "eur") is None)
    check("no gbp->inr conversion", convert(5, "gbp", "inr") is None)
    check("cents convert within usd", close(normalize_unit(250, "cents")[0], 2.5))
    check("pence convert within gbp", normalize_unit(250, "pence") == (2.5, "gbp"))
    check("unknown unit passes through", normalize_unit(5, "widgets") == (5.0, "widgets"))
    check("no cross-dimension conversion", convert(5, "meter", "second") is None)

    print("--- contradictions ---")

    docs_conflict = [
        {"id": 1, "title": "TechCrunch", "text": "The platform now has 4.2 million users worldwide."},
        {"id": 2, "title": "Company blog", "text": "Our service reached 6.1 million users this year."},
    ]
    report = find_contradictions(docs_conflict)
    check("conflict detected", len(report["conflicts"]) == 1, str(report["conflicts"]))
    if report["conflicts"]:
        conflict = report["conflicts"][0]
        check("conflict subject", "user" in conflict["subject"], conflict["subject"])
        check("conflict min/max", close(conflict["min"], 4.2e6) and close(conflict["max"], 6.1e6), str(conflict))
        check("conflict severity minor", conflict["severity"] == "minor", conflict["severity"])
        check("conflict has both documents",
              {c["document_id"] for c in conflict["claims"]} == {1, 2}, str(conflict["claims"]))
    check("total_quantities counted", report["total_quantities"] >= 2, str(report["total_quantities"]))
    check("documents_scanned", report["documents_scanned"] == 2)

    docs_major = [
        {"id": 1, "title": "A", "text": "The fleet covers 400 km per trip."},
        {"id": 2, "title": "B", "text": "The fleet covers 1200 km per trip."},
    ]
    major = find_contradictions(docs_major)
    check("3x gap is major", major["conflicts"] and major["conflicts"][0]["severity"] == "major",
          str(major["conflicts"]))

    docs_rounding = [
        {"id": 1, "title": "A", "text": "Annual revenue was $4.2 million last year."},
        {"id": 2, "title": "B", "text": "Annual revenue was $4.18 million last year."},
    ]
    rounding = find_contradictions(docs_rounding)
    check("rounding is NOT a conflict", rounding["conflicts"] == [], str(rounding["conflicts"]))
    check("rounding still grouped", len(rounding["groups"]) == 1, str(rounding["groups"]))

    docs_approx = [
        {"id": 1, "title": "A", "text": "There are about 5 outages each month."},
        {"id": 2, "title": "B", "text": "There were 5.2 outages each month."},
    ]
    approx = find_contradictions(docs_approx)
    check("'about 5' vs 5.2 is NOT a conflict", approx["conflicts"] == [], str(approx["conflicts"]))

    docs_currency = [
        {"id": 1, "title": "US filing", "text": "Annual revenue of $3 million was reported."},
        {"id": 2, "title": "EU filing", "text": "Annual revenue of €3 million was reported."},
    ]
    currency = find_contradictions(docs_currency)
    check("different currencies never conflict", currency["conflicts"] == [], str(currency["conflicts"]))
    check("different currencies stay in separate groups",
          all(len(g["claims"]) < 2 for g in currency["groups"]) or currency["groups"] == [],
          str(currency["groups"]))

    docs_units = [
        {"id": 1, "title": "A", "text": "The route is 100 km long."},
        {"id": 2, "title": "B", "text": "The route is 62.14 miles long."},
    ]
    units = find_contradictions(docs_units)
    check("km and miles compare without conflicting", units["conflicts"] == [], str(units["conflicts"]))
    check("km and miles land in one group", len(units["groups"]) == 1, str(units["groups"]))

    docs_dims = [
        {"id": 1, "title": "A", "text": "The trip takes 5 hours."},
        {"id": 2, "title": "B", "text": "The trip takes 5 km."},
    ]
    dims = find_contradictions(docs_dims)
    check("different dimensions never conflict", dims["conflicts"] == [], str(dims["conflicts"]))

    docs_percent = [
        {"id": 1, "title": "A", "text": "The error rate is 0.1%."},
        {"id": 2, "title": "B", "text": "The error rate is 0.3%."},
    ]
    percent = find_contradictions(docs_percent)
    check("tiny percentages are not a major conflict", percent["conflicts"] == [], str(percent["conflicts"]))

    docs_storage = [
        {"id": 1, "title": "A", "text": "Each node ships with 512 GB of storage."},
        {"id": 2, "title": "B", "text": "Each node ships with 512 GiB of storage."},
    ]
    storage = find_contradictions(docs_storage)
    check("GB vs GiB grouped together", len(storage["groups"]) == 1, str(storage["groups"]))
    check("GB vs GiB is a 7% gap -> minor, never major",
          storage["groups"][0]["severity"] == "minor"
          and close(storage["groups"][0]["spread_ratio"], 1.0737, 1e-3),
          str(storage["groups"][0]["severity"]))

    docs_equal_data = [
        {"id": 1, "title": "A", "text": "Each node ships with 1 TB of storage."},
        {"id": 2, "title": "B", "text": "Each node ships with 1000 GB of storage."},
    ]
    equal_data = find_contradictions(docs_equal_data)
    check("1 TB == 1000 GB, no conflict", equal_data["conflicts"] == [], str(equal_data["conflicts"]))

    print("--- realistic multi-document pipeline ---")

    corpus = [
        {"id": 1, "title": "TechCrunch", "text":
            "The platform reported 4.2 million monthly active users and annual "
            "revenue of $120 million. Latency averages about 250 ms."},
        {"id": 2, "title": "Company blog", "text":
            "We now serve 6.1 million active users. Revenue reached $122 million "
            "last year, with latency near 240ms."},
        {"id": 3, "title": "Analyst note", "text":
            "Our estimate puts revenue at EUR 110 million and active users at "
            "roughly 6 million."},
    ]
    corpus_report = find_contradictions(corpus)
    by_subject = {g["subject"]: g for g in corpus_report["groups"]}
    check("users clustered across all three documents",
          "active users" in by_subject and len(by_subject["active users"]["claims"]) == 3,
          str(list(by_subject)))
    check("user counts flagged as a conflict",
          by_subject.get("active users", {}).get("severity") == "minor",
          str(by_subject.get("active users", {}).get("severity")))
    check("revenue in USD only -- euro claim excluded",
          by_subject.get("revenue", {}).get("unit") == "usd"
          and len(by_subject.get("revenue", {}).get("claims", [])) == 2,
          str(by_subject.get("revenue", {}).get("claims")))
    check("usd revenue is not a conflict",
          by_subject.get("revenue", {}).get("conflicting") is False,
          str(by_subject.get("revenue", {})))
    check("ms and s latency compared on one scale",
          by_subject.get("latency", {}).get("unit") == "second"
          and by_subject.get("latency", {}).get("conflicting") is False,
          str(by_subject.get("latency", {})))
    check("exactly one conflict in the corpus", len(corpus_report["conflicts"]) == 1,
          str([c["subject"] for c in corpus_report["conflicts"]]))

    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if FAILED:
        raise SystemExit(1)
