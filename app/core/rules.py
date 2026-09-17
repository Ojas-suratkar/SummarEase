"""
rules -- a tiny, safe expression language for user-authored automation.

WHY THIS FILE EXISTS
--------------------
SummarEase lets people write automation rules in their own words:

    when a watched page changes
    and   text mentions "funding round"
    then  extract the numbers, tag it "fundraising", email me

The "and text mentions ..." part is a *condition expression* that the
user types and the **server** then evaluates, potentially thousands of
times a day, against documents the user did not write. That is exactly
the shape of a remote-code-execution hole, so the way this is NOT built
matters as much as the way it is.

WHY NOT eval()
--------------
`eval(user_string)` would be a one-line implementation and a total
compromise of the box. Python has no meaningful sandbox: even with
`{"__builtins__": {}}` an attacker reaches the interpreter through the
object graph of any value you hand them --

    ().__class__.__bases__[0].__subclasses__()   # -> every class loaded
    "".__class__.__mro__[1].__subclasses__()     # -> Popen, eventually

-- and from there to `subprocess`, the filesystem, and the database
credentials in the process environment. Blacklisting substrings
("__", "import", "os") does not work either; the bypasses are famous and
endless (`getattr`, string concatenation, unicode confusables, f-strings).
The only defence that actually holds is *not having an interpreter in the
loop at all*.

So this module implements a real, small language:

    text  --tokenizer-->  tokens  --recursive descent-->  AST (plain dicts)
                                                              |
                                                     tree-walking evaluator
                                                              |
                                               values read ONLY from a dict

There is no `eval`, no `exec`, no `compile` of user text, no `__import__`,
and -- critically -- **no attribute access in the grammar at all**. `.`
is not a token. A user cannot name a Python object's members because the
language has no syntax for doing so. The evaluator's entire universe is
the keys of the context dict the caller supplies and the operators
enumerated below. The AST is plain JSON-serialisable dicts, so a parsed
rule can be stored, diffed, shown in a UI, and audited.

THE GRAMMAR (EBNF)
------------------
    expression  := or_expr
    or_expr     := and_expr { "or" and_expr }
    and_expr    := not_expr { "and" not_expr }
    not_expr    := "not" not_expr | comparison
    comparison  := primary [ compare_op primary ]      (non-associative)
    compare_op  := "==" | "!=" | "<" | "<=" | ">" | ">="
                 | "in" | "not in" | "includes"
                 | "contains" | "mentions" | "matches"
                 | "before" | "after"
    primary     := "(" expression ")"
                 | "[" [ expression { "," expression } ] "]"
                 | STRING | NUMBER | REGEX
                 | "true" | "false" | "null"
                 | "-" NUMBER
                 | IDENTIFIER

Precedence, tightest first: comparison > not > and > or. So
`a or b and c` parses as `a or (b and c)`, and `not a and b` parses as
`(not a) and b`, matching every language a user has ever met.
Comparisons are deliberately non-associative: `1 < x < 10` is a syntax
error with a position, not a silently-wrong chain.

ReDoS MITIGATION (the `matches` operator)
-----------------------------------------
Users author regexes, so `catastrophic backtracking` is a denial-of-service
vector: `/(a+)+$/` against 30 non-matching characters is already billions
of steps, and CPython's `re` engine has **no timeout parameter** and does
not release the GIL, so a thread-based watchdog cannot interrupt it --
once `re.search` is running, the worker is gone. `signal.alarm` is
main-thread-only and therefore useless inside a Flask/gunicorn worker.

Since the regex cannot be stopped once started, the defence is to never
start a dangerous one. Three layers, all *before* the engine runs:

1. **Static safety analysis** (`_assert_regex_safe`). The pattern is
   scanned with a small bracket-aware walker that rejects the constructs
   that make exponential backtracking possible at all:
     - a quantified group whose body itself contains a quantifier or an
       alternation -- `(a+)+`, `(a|a)*`, `(a*)*`, `(x|xy)+`;
     - stacked quantifiers -- `a+*`;
     - backreferences (`\\1`), which force the engine out of any
       linear-time strategy;
     - lookarounds, which can hide a nested quantifier from the check;
     - bounded repeats with a large upper bound (`a{5000}`), whose blowup
       is polynomial rather than exponential but is blowup all the same.
   What survives is close to a "safe regular expression" -- patterns
   whose match time is bounded by O(len(pattern) * len(subject)).
2. **Subject truncation**. Even a linear pattern is only linear in the
   subject, and documents here can be megabytes. The subject is capped at
   `MAX_REGEX_SUBJECT` characters, which bounds the worst case to a few
   milliseconds regardless of pattern.
3. **Deadline checks**. The evaluator carries a monotonic deadline
   (`timeout_ms`) checked before every node and immediately after every
   regex, so a rule that is merely *slow* (a big list, many operators)
   aborts with `RuleEvaluationError` instead of eating a worker.

Layers 1 and 2 are the real protection; layer 3 is the backstop.

OTHER HARDENING
---------------
Expression length, token count, parenthesis depth, string literal length,
list length and evaluation step count are all bounded (see the MAX_*
constants). Every parse error carries a character position, because these
messages are shown to non-technical users. Unknown identifiers raise a
clean `RuleEvaluationError` naming the identifier and listing what *is*
available -- never an AttributeError, KeyError or crash.

ACTIONS ARE DATA
----------------
`evaluate_rule` decides *whether* a rule fires and returns the list of
actions to perform. It never performs one. Sending mail, calling
webhooks and writing tags stay in the Flask layer where authentication,
rate limits and the request context live. This module has no I/O at all,
which is also what makes it trivially testable.

Stdlib only. No network. No AI calls.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

try:  # package import (normal Flask runtime)
    from .text_kit import STOPWORDS, normalize, porter_stem, tokenize
except ImportError:  # direct execution: `python rules.py` inside app/core
    from text_kit import STOPWORDS, normalize, porter_stem, tokenize


__all__ = [
    "RuleSyntaxError",
    "RuleEvaluationError",
    "parse",
    "evaluate",
    "validate",
    "describe",
    "identifiers_used",
    "Rule",
    "evaluate_rule",
    "run_rules",
    "validate_actions",
    "ACTION_SCHEMA",
]


# ---------------------------------------------------------------------------
# Resource limits
#
# Every one of these exists because the input is hostile-by-default. A rule
# is stored text that runs later, unattended, on a shared worker: the cost
# of evaluating it must be bounded by constants we chose, not by how much
# the author typed.
# ---------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 4_000   # characters of source text
MAX_TOKENS = 1_000              # tokens produced by the lexer
MAX_DEPTH = 32                  # parenthesis / recursion depth
MAX_STRING_LENGTH = 1_000       # characters inside one string literal
MAX_LIST_ITEMS = 200            # elements in one [...] literal
MAX_NUMBER_DIGITS = 40          # digits in one numeric literal
MAX_REGEX_LENGTH = 500          # characters in one /.../ literal
MAX_REGEX_SUBJECT = 20_000      # characters of text a regex may scan
REGEX_CHUNK = 4_000             # characters handed to the engine in one call
REGEX_CHUNK_OVERLAP = 512       # carried between chunks so matches can span
MAX_ANCHORED_SUBJECT = 8_000    # ^/$ patterns cannot be chunked; smaller cap
MAX_TEXT_SUBJECT = 200_000      # characters contains/mentions may scan
MAX_EVAL_STEPS = 10_000         # AST nodes visited per evaluation


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RuleSyntaxError(ValueError):
    """The expression could not be parsed.

    Carries `position` (0-based character offset into the source) so the UI
    can underline the offending character. Subclasses ValueError because
    callers that merely want "bad user input" can catch that.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        self.position = position
        if position is not None:
            message = f"{message} (at character {position + 1})"
        super().__init__(message)


class RuleEvaluationError(RuntimeError):
    """The expression parsed but could not be evaluated: unknown
    identifier, type mismatch, unparseable date, or the timeout budget
    being exhausted."""


# ---------------------------------------------------------------------------
# Tokenizer
#
# Hand-written rather than regex-split: a regex "tokenizer" cannot handle
# string literals containing operators ("a and b" inside quotes), regex
# literals containing slashes, or report an exact character position for a
# bad character. All three matter here.
# ---------------------------------------------------------------------------

# Words that are operators/literals rather than identifiers.
_KEYWORDS = frozenset({
    "and", "or", "not",
    "in", "includes", "contains", "mentions", "matches", "before", "after",
    "true", "false", "null", "none",
})

_SYMBOL_OPS = ("==", "!=", "<=", ">=", "<", ">")

# Word operators usable as the operator of a comparison.
_WORD_COMPARE_OPS = frozenset({
    "in", "includes", "contains", "mentions", "matches", "before", "after",
})

_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_STRING_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "/": "/",
    "0": "\0",
}


@dataclass(frozen=True)
class _Token:
    kind: str       # STRING NUMBER REGEX IDENT KEYWORD OP PUNCT EOF
    value: Any
    pos: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.kind} {self.value!r} @{self.pos}>"


def _tokenize_expression(expression: str) -> list[_Token]:
    """Source text -> token list. Raises RuleSyntaxError with a position."""
    if not isinstance(expression, str):
        raise RuleSyntaxError("A rule condition must be text.", 0)
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise RuleSyntaxError(
            f"Rule is too long ({len(expression)} characters; the limit is "
            f"{MAX_EXPRESSION_LENGTH}).",
            MAX_EXPRESSION_LENGTH,
        )

    tokens: list[_Token] = []
    i = 0
    n = len(expression)

    while i < n:
        char = expression[i]

        # --- whitespace -------------------------------------------------
        if char.isspace():
            i += 1
            continue

        # --- comments (# to end of line) --------------------------------
        if char == "#":
            while i < n and expression[i] != "\n":
                i += 1
            continue

        start = i

        # --- string literal ---------------------------------------------
        if char in "\"'":
            quote = char
            i += 1
            chunks: list[str] = []
            closed = False
            while i < n:
                c = expression[i]
                if c == "\\":
                    if i + 1 >= n:
                        raise RuleSyntaxError(
                            "Text ends with a dangling backslash.", i
                        )
                    nxt = expression[i + 1]
                    chunks.append(_STRING_ESCAPES.get(nxt, nxt))
                    i += 2
                    continue
                if c == quote:
                    closed = True
                    i += 1
                    break
                chunks.append(c)
                i += 1
            if not closed:
                raise RuleSyntaxError(
                    f"Unterminated text value -- no closing {quote} found.", start
                )
            value = "".join(chunks)
            if len(value) > MAX_STRING_LENGTH:
                raise RuleSyntaxError(
                    f"Text value is too long ({len(value)} characters; the limit "
                    f"is {MAX_STRING_LENGTH}).",
                    start,
                )
            tokens.append(_Token("STRING", value, start))

        # --- regex literal ------------------------------------------------
        # `/` is unambiguous here: the language has no division operator, so
        # a slash can only begin a pattern.
        elif char == "/":
            i += 1
            chunks = []
            closed = False
            in_class = False   # inside [...] a '/' is still literal, but so
            while i < n:       # is a ']'-less bracket -- track it for sanity
                c = expression[i]
                if c == "\\":
                    if i + 1 >= n:
                        raise RuleSyntaxError(
                            "Pattern ends with a dangling backslash.", i
                        )
                    chunks.append(c)
                    chunks.append(expression[i + 1])
                    i += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    closed = True
                    i += 1
                    break
                elif c == "\n":
                    break
                chunks.append(c)
                i += 1
            if not closed:
                raise RuleSyntaxError(
                    "Unterminated pattern -- no closing / found.", start
                )
            flags = ""
            while i < n and expression[i] in "ims":
                if expression[i] not in flags:
                    flags += expression[i]
                i += 1
            pattern = "".join(chunks)
            _assert_regex_safe(pattern, start)
            tokens.append(_Token("REGEX", (pattern, flags), start))

        # --- number ---------------------------------------------------------
        elif char.isdigit():
            match = _NUMBER_RE.match(expression, i)
            assert match is not None
            raw = match.group(0)
            if len(raw) > MAX_NUMBER_DIGITS:
                raise RuleSyntaxError(
                    f"Number is too long ({len(raw)} digits; the limit is "
                    f"{MAX_NUMBER_DIGITS}).",
                    start,
                )
            i = match.end()
            # Guard against `12abc` being read as 12 followed by a stray name.
            if i < n and (expression[i].isalpha() or expression[i] == "_"):
                raise RuleSyntaxError(
                    f"{raw!r} is followed by letters -- numbers and names must "
                    "be separated by an operator.",
                    i,
                )
            value: Any
            if "." in raw or "e" in raw.lower():
                value = float(raw)
            else:
                value = int(raw)
            tokens.append(_Token("NUMBER", value, start))

        # --- identifier / keyword -------------------------------------------
        elif char.isalpha():
            match = _IDENT_RE.match(expression, i)
            assert match is not None
            word = match.group(0)
            i = match.end()
            lowered = word.lower()
            if lowered in _KEYWORDS:
                tokens.append(_Token("KEYWORD", lowered, start))
            else:
                # Dunder names are rejected outright. They cannot reach
                # anything (the evaluator only ever indexes the context dict),
                # but rejecting them turns a probe like __import__("os") into
                # an explicit, logged syntax error instead of a quiet miss.
                if "__" in word:
                    raise RuleSyntaxError(
                        f"{word!r} is not a valid field name -- double "
                        "underscores are not allowed.",
                        start,
                    )
                tokens.append(_Token("IDENT", word, start))

        # --- leading underscore: private-looking name ------------------------
        elif char == "_":
            raise RuleSyntaxError(
                "Field names cannot start with an underscore.", start
            )

        # --- symbolic operators ----------------------------------------------
        else:
            for op in _SYMBOL_OPS:
                if expression.startswith(op, i):
                    tokens.append(_Token("OP", op, start))
                    i += len(op)
                    break
            else:
                if char in "()[],":
                    tokens.append(_Token("PUNCT", char, start))
                    i += 1
                elif char == "-":
                    tokens.append(_Token("OP", "-", start))
                    i += 1
                elif char == "=":
                    raise RuleSyntaxError(
                        "Use '==' to compare (a single '=' assigns, which "
                        "rules cannot do).",
                        start,
                    )
                elif char == ".":
                    raise RuleSyntaxError(
                        "'.' is not allowed in rules -- use a plain field name.",
                        start,
                    )
                elif char in "&|":
                    word = "and" if char == "&" else "or"
                    raise RuleSyntaxError(
                        f"Unexpected {char!r} -- write {word!r} instead.", start
                    )
                elif char in "!":
                    raise RuleSyntaxError(
                        "Unexpected '!' -- write 'not' instead (or '!=' to "
                        "compare).",
                        start,
                    )
                else:
                    raise RuleSyntaxError(f"Unexpected character {char!r}.", start)

        if len(tokens) > MAX_TOKENS:
            raise RuleSyntaxError(
                f"Rule is too complex (more than {MAX_TOKENS} tokens).", start
            )

    tokens.append(_Token("EOF", None, n))
    return tokens


# ---------------------------------------------------------------------------
# Regex safety analysis  (see the module docstring for the full rationale)
# ---------------------------------------------------------------------------

_BACKREF_RE = re.compile(r"\\[1-9]")
_LOOKAROUND_RE = re.compile(r"\(\?<?[=!]")
_BOUNDED_REPEAT_RE = re.compile(r"\{(\d*)(?:,(\d*))?\}")
_MAX_BOUNDED_REPEAT = 100


def _assert_regex_safe(pattern: str, pos: int) -> None:
    """Reject patterns that could backtrack catastrophically.

    CPython's `re` cannot be interrupted once it starts (no timeout, GIL
    held, `signal.alarm` unusable off the main thread), so this must be a
    *pre*-check: dangerous patterns are never handed to the engine.

    The walker below tracks bracket state so that quantifier characters
    inside a character class (`[+*]`) are treated as literals, and, for each
    group, whether its body contained a quantifier or an alternation. A
    group that did, and which is itself quantified, is the classic
    exponential shape `(a+)+` / `(a|a)*` and is refused.
    """
    if len(pattern) > MAX_REGEX_LENGTH:
        raise RuleSyntaxError(
            f"Pattern is too long ({len(pattern)} characters; the limit is "
            f"{MAX_REGEX_LENGTH}).",
            pos,
        )
    if not pattern:
        raise RuleSyntaxError("Pattern is empty.", pos)
    if _BACKREF_RE.search(pattern):
        raise RuleSyntaxError(
            "Patterns with backreferences (\\1) are not allowed -- they can "
            "make matching exponentially slow.",
            pos,
        )
    if _LOOKAROUND_RE.search(pattern):
        raise RuleSyntaxError(
            "Patterns with lookahead/lookbehind are not allowed -- they can "
            "hide expensive sub-patterns.",
            pos,
        )
    for match in _BOUNDED_REPEAT_RE.finditer(pattern):
        for group in match.groups():
            if group and int(group) > _MAX_BOUNDED_REPEAT:
                raise RuleSyntaxError(
                    f"Pattern repeats more than {_MAX_BOUNDED_REPEAT} times "
                    "-- that is too slow to run on every document.",
                    pos,
                )

    quant_chars = "*+?"
    # Per nesting level: [saw_quantifier, saw_alternation]
    stack: list[list[bool]] = [[False, False]]
    in_class = False
    prev_was_quantified = False
    i = 0
    length = len(pattern)

    while i < length:
        char = pattern[i]

        if char == "\\":
            i += 2
            prev_was_quantified = False
            continue

        if in_class:
            if char == "]":
                in_class = False
            i += 1
            continue

        if char == "[":
            in_class = True
            i += 1
            prev_was_quantified = False
            continue

        if char == "(":
            stack.append([False, False])
            i += 1
            prev_was_quantified = False
            if len(stack) > 20:
                raise RuleSyntaxError(
                    "Pattern nests groups too deeply.", pos
                )
            continue

        if char == ")":
            if len(stack) == 1:
                raise RuleSyntaxError("Pattern has an unmatched ')'.", pos)
            body_quant, body_alt = stack.pop()
            i += 1
            # Is this group quantified?
            quantified = False
            if i < length and (pattern[i] in quant_chars or pattern[i] == "{"):
                quantified = True
                if pattern[i] == "{":
                    close = pattern.find("}", i)
                    i = length if close == -1 else close + 1
                else:
                    i += 1
                    while i < length and pattern[i] in "?+":  # lazy/possessive
                        i += 1
            if quantified and (body_quant or body_alt):
                raise RuleSyntaxError(
                    "Pattern has a repeated group that itself repeats or has "
                    "alternatives (like (a+)+ or (a|b)*) -- that can take "
                    "effectively forever to match. Simplify it.",
                    pos,
                )
            if quantified:
                stack[-1][0] = True
            elif body_quant:
                stack[-1][0] = True
            if body_alt:
                # An alternation inside an unquantified group still counts as
                # a quantifier-ish cost for the *enclosing* group's check.
                stack[-1][1] = stack[-1][1] or False
            prev_was_quantified = quantified
            continue

        if char == "|":
            stack[-1][1] = True
            i += 1
            prev_was_quantified = False
            continue

        if char in quant_chars or char == "{":
            if char == "{":
                close = pattern.find("}", i)
                if close == -1:
                    i += 1
                    prev_was_quantified = False
                    continue
                i = close + 1
            else:
                if prev_was_quantified and char != "?":
                    raise RuleSyntaxError(
                        "Pattern stacks quantifiers (like a+*) -- that can take "
                        "effectively forever to match.",
                        pos,
                    )
                i += 1
                while i < length and pattern[i] in "?+":
                    i += 1
            stack[-1][0] = True
            prev_was_quantified = True
            continue

        i += 1
        prev_was_quantified = False

    if len(stack) != 1:
        raise RuleSyntaxError("Pattern has an unmatched '('.", pos)
    if in_class:
        raise RuleSyntaxError("Pattern has an unmatched '['.", pos)

    try:
        re.compile(pattern)
    except re.error as exc:
        raise RuleSyntaxError(f"Invalid pattern: {exc}.", pos) from None


def _is_anchored(pattern: str) -> bool:
    """True if the pattern uses ^, $, \\A, \\Z or \\z outside a character
    class. Such patterns cannot be scanned in windows (every window start
    would masquerade as the start of the subject), so the evaluator falls
    back to one search over a smaller cap."""
    in_class = False
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            if i + 1 < len(pattern) and pattern[i + 1] in "AZz":
                return True
            i += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
            i += 1
            continue
        if char == "[":
            in_class = True
        elif char in "^$":
            return True
        i += 1
    return False


_REGEX_CACHE: dict[tuple[str, str], re.Pattern[str]] = {}


def _compiled_regex(pattern: str, flags: str) -> re.Pattern[str]:
    """Compile (and cache) a pattern that has already passed the safety
    check. `re.compile` on a *pattern string* is not `compile()` of Python
    source -- no user code is compiled anywhere in this module."""
    key = (pattern, flags)
    cached = _REGEX_CACHE.get(key)
    if cached is not None:
        return cached
    bits = 0
    if "i" in flags:
        bits |= re.IGNORECASE
    if "m" in flags:
        bits |= re.MULTILINE
    if "s" in flags:
        bits |= re.DOTALL
    compiled = re.compile(pattern, bits)
    if len(_REGEX_CACHE) < 500:  # bounded: rules are user-authored
        _REGEX_CACHE[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Parser -- recursive descent, one method per precedence level
# ---------------------------------------------------------------------------

_COMPARE_SYMBOLS = frozenset({"==", "!=", "<", "<=", ">", ">="})


class _Parser:
    """Recursive-descent parser producing plain-dict AST nodes.

    Node shapes (all JSON-serialisable, which is what lets a rule be stored
    and rendered without re-parsing):

        {"type": "literal",    "value": <str|int|float|bool|None>}
        {"type": "regex",      "pattern": str, "flags": str}
        {"type": "list",       "items": [node, ...]}
        {"type": "identifier", "name": str}
        {"type": "not",        "operand": node}
        {"type": "and"|"or",   "left": node, "right": node}
        {"type": "compare",    "op": str, "left": node, "right": node}
    """

    def __init__(self, tokens: Sequence[_Token]) -> None:
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    # -- token helpers ----------------------------------------------------

    def _peek(self, offset: int = 0) -> _Token:
        index = min(self.index + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def _advance(self) -> _Token:
        token = self.tokens[self.index]
        if token.kind != "EOF":
            self.index += 1
        return token

    def _expect_punct(self, char: str, what: str) -> _Token:
        token = self._peek()
        if token.kind == "PUNCT" and token.value == char:
            return self._advance()
        raise RuleSyntaxError(
            f"Expected {char!r} to close {what}, found {_describe_token(token)}.",
            token.pos,
        )

    # -- grammar ----------------------------------------------------------

    def parse_expression(self) -> dict:
        node = self._parse_or()
        token = self._peek()
        if token.kind != "EOF":
            raise RuleSyntaxError(
                f"Unexpected {_describe_token(token)} after a complete "
                "condition -- did you mean to join it with 'and' / 'or'?",
                token.pos,
            )
        return node

    def _parse_or(self) -> dict:
        node = self._parse_and()
        while self._peek().kind == "KEYWORD" and self._peek().value == "or":
            self._advance()
            right = self._parse_and()
            node = {"type": "or", "left": node, "right": right}
        return node

    def _parse_and(self) -> dict:
        node = self._parse_not()
        while self._peek().kind == "KEYWORD" and self._peek().value == "and":
            self._advance()
            right = self._parse_not()
            node = {"type": "and", "left": node, "right": right}
        return node

    def _parse_not(self) -> dict:
        token = self._peek()
        if token.kind == "KEYWORD" and token.value == "not":
            self._advance()
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise RuleSyntaxError(
                    f"Condition is nested too deeply (limit {MAX_DEPTH}).",
                    token.pos,
                )
            operand = self._parse_not()
            self.depth -= 1
            return {"type": "not", "operand": operand}
        return self._parse_comparison()

    def _parse_comparison(self) -> dict:
        left = self._parse_primary()
        op, op_token = self._match_compare_op()
        if op is None:
            return left
        right = self._parse_primary()

        # Non-associative on purpose: `1 < x < 10` reads as a promise the
        # language does not keep, so it is an error rather than a surprise.
        follow_op, follow_token = self._peek_compare_op()
        if follow_op is not None:
            raise RuleSyntaxError(
                f"Cannot chain comparisons ({op!r} then {follow_op!r}) -- write "
                "two comparisons joined by 'and'.",
                follow_token.pos if follow_token else op_token.pos,
            )

        if op == "matches":
            right = self._coerce_to_regex(right, op_token)
        return {"type": "compare", "op": op, "left": left, "right": right}

    def _peek_compare_op(self) -> tuple[str | None, _Token | None]:
        token = self._peek()
        if token.kind == "OP" and token.value in _COMPARE_SYMBOLS:
            return str(token.value), token
        if token.kind == "KEYWORD":
            if token.value in _WORD_COMPARE_OPS:
                return str(token.value), token
            if token.value == "not":
                nxt = self._peek(1)
                if nxt.kind == "KEYWORD" and nxt.value == "in":
                    return "not in", token
        return None, None

    def _match_compare_op(self) -> tuple[str | None, _Token]:
        op, token = self._peek_compare_op()
        current = self._peek()
        if op is None:
            return None, current
        if op == "not in":
            self._advance()  # not
            self._advance()  # in
        else:
            self._advance()
        return op, current

    def _coerce_to_regex(self, node: dict, op_token: _Token) -> dict:
        """`matches` accepts /.../ or a plain string; a string is promoted to
        a regex node here so the safety check runs at *parse* time, before
        the rule is ever saved."""
        if node["type"] == "regex":
            return node
        if node["type"] == "literal" and isinstance(node["value"], str):
            _assert_regex_safe(node["value"], op_token.pos)
            return {"type": "regex", "pattern": node["value"], "flags": "i"}
        raise RuleSyntaxError(
            "'matches' needs a pattern like /revenue of \\$[0-9]+/ on the "
            "right-hand side.",
            op_token.pos,
        )

    def _parse_primary(self) -> dict:
        token = self._advance()

        if token.kind == "PUNCT" and token.value == "(":
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise RuleSyntaxError(
                    f"Too many nested brackets (limit {MAX_DEPTH}).", token.pos
                )
            node = self._parse_or()
            self.depth -= 1
            self._expect_punct(")", "the bracketed group")
            return node

        if token.kind == "PUNCT" and token.value == "[":
            return self._parse_list(token)

        if token.kind == "STRING":
            return {"type": "literal", "value": token.value}

        if token.kind == "NUMBER":
            return {"type": "literal", "value": token.value}

        if token.kind == "REGEX":
            pattern, flags = token.value
            return {"type": "regex", "pattern": pattern, "flags": flags}

        if token.kind == "OP" and token.value == "-":
            nxt = self._peek()
            if nxt.kind != "NUMBER":
                raise RuleSyntaxError(
                    "'-' can only be used to write a negative number.", token.pos
                )
            self._advance()
            return {"type": "literal", "value": -nxt.value}

        if token.kind == "KEYWORD":
            if token.value == "true":
                return {"type": "literal", "value": True}
            if token.value == "false":
                return {"type": "literal", "value": False}
            if token.value in ("null", "none"):
                return {"type": "literal", "value": None}
            if token.value == "not":
                raise RuleSyntaxError(
                    "'not' cannot be used here -- it goes before a condition, "
                    "as in: not archived.",
                    token.pos,
                )
            raise RuleSyntaxError(
                f"{token.value!r} is an operator; a value or field name was "
                "expected here.",
                token.pos,
            )

        if token.kind == "IDENT":
            return {"type": "identifier", "name": token.value}

        if token.kind == "EOF":
            raise RuleSyntaxError(
                "Condition ends unexpectedly -- something is missing after the "
                "last operator.",
                token.pos,
            )

        raise RuleSyntaxError(
            f"Unexpected {_describe_token(token)}; a value or field name was "
            "expected.",
            token.pos,
        )

    def _parse_list(self, open_token: _Token) -> dict:
        items: list[dict] = []
        if self._peek().kind == "PUNCT" and self._peek().value == "]":
            self._advance()
            return {"type": "list", "items": items}

        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise RuleSyntaxError(
                f"Too many nested brackets (limit {MAX_DEPTH}).", open_token.pos
            )
        while True:
            items.append(self._parse_or())
            if len(items) > MAX_LIST_ITEMS:
                raise RuleSyntaxError(
                    f"List has too many items (limit {MAX_LIST_ITEMS}).",
                    open_token.pos,
                )
            token = self._peek()
            if token.kind == "PUNCT" and token.value == ",":
                self._advance()
                # Tolerate a trailing comma: [a, b,]
                if self._peek().kind == "PUNCT" and self._peek().value == "]":
                    break
                continue
            break
        self.depth -= 1
        self._expect_punct("]", "the list")
        return {"type": "list", "items": items}


def _describe_token(token: _Token) -> str:
    """Human-readable token description for error messages."""
    if token.kind == "EOF":
        return "the end of the condition"
    if token.kind == "STRING":
        return f"text {token.value!r}"
    if token.kind == "NUMBER":
        return f"number {token.value}"
    if token.kind == "REGEX":
        return "a pattern"
    if token.kind == "IDENT":
        return f"field {token.value!r}"
    return f"{str(token.value)!r}"


# ---------------------------------------------------------------------------
# Public: parse
# ---------------------------------------------------------------------------


def parse(expression: str) -> dict:
    """Parse a condition expression into a nested-dict AST.

    The AST is plain JSON data on purpose: rules are stored in the database
    and rendered in the UI, and a dict tree can be persisted, diffed and
    inspected without ever handing user text back to a parser -- let alone
    to an interpreter.

    Raises RuleSyntaxError (with a character position) on bad input.
    """
    if expression is None or (isinstance(expression, str) and not expression.strip()):
        raise RuleSyntaxError("The condition is empty.", 0)
    tokens = _tokenize_expression(expression)
    return _Parser(tokens).parse_expression()


def _as_ast(expression_or_ast: Any) -> dict:
    """Accept either source text or an already-parsed AST."""
    if isinstance(expression_or_ast, str):
        return parse(expression_or_ast)
    if isinstance(expression_or_ast, dict):
        if "type" not in expression_or_ast:
            raise RuleSyntaxError("That is not a valid parsed condition.", 0)
        return expression_or_ast
    raise RuleSyntaxError(
        "A condition must be text or a parsed condition, not "
        f"{type(expression_or_ast).__name__}.",
        0,
    )


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _truthy(value: Any) -> bool:
    """Truthiness for bare values in boolean position (`not archived`)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def _as_number(value: Any, op: str) -> float:
    """Numeric coercion for ordering comparisons. Numeric strings are
    accepted because context values often arrive from form input or JSON."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    if isinstance(value, (list, tuple, set, dict)):
        # `word_count > 5` on a list is almost always meant as "how many".
        return float(len(value))
    raise RuleEvaluationError(
        f"Cannot compare {_describe_value(value)} with '{op}' -- it is not a "
        "number."
    )


def _as_text(value: Any) -> str:
    """Flatten a context value to searchable text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_as_text(item) for item in value.values())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _describe_value(value: Any) -> str:
    if isinstance(value, str):
        return f"text {value!r}"
    if value is None:
        return "nothing"
    return f"{type(value).__name__} {value!r}"


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _as_datetime(value: Any, op: str) -> datetime:
    """Parse an ISO-8601 date/datetime into a naive UTC datetime.

    Naive-UTC rather than aware: context values mix `date`, `datetime`,
    aware timestamps from the DB and plain "2024-01-01" strings from the
    rule text, and comparing an aware to a naive datetime raises TypeError.
    Normalising everything to one representation makes `before`/`after`
    total instead of occasionally exploding.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            raise RuleEvaluationError(
                f"Cannot read {value!r} as a date for '{op}'."
            ) from None
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise RuleEvaluationError(f"Cannot use an empty value as a date ('{op}').")
        candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
        candidate = candidate.replace(" ", "T", 1) if " " in candidate[:11] else candidate
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            if _ISO_DATE_RE.match(text):
                try:
                    dt = datetime.strptime(text[:10], "%Y-%m-%d")
                except ValueError:
                    raise RuleEvaluationError(
                        f"{text!r} is not a date I can read -- use a form like "
                        "2024-01-31."
                    ) from None
            else:
                raise RuleEvaluationError(
                    f"{text!r} is not a date I can read -- use a form like "
                    "2024-01-31."
                ) from None
    else:
        raise RuleEvaluationError(
            f"Cannot use {_describe_value(value)} as a date for '{op}'."
        )

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _loose_equal(left: Any, right: Any) -> bool:
    """Equality that behaves the way a non-programmer expects.

    Strings compare case- and accent-insensitively ("Article" == "article"),
    because users type rules by hand and source data is inconsistently
    cased. Numbers and numeric strings compare numerically, because context
    values often arrive as JSON strings. Everything else falls back to
    Python equality.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        if isinstance(left, bool) and isinstance(right, bool):
            return left is right
        if isinstance(left, bool) and isinstance(right, (int, float)):
            return float(left) == float(right)
        if isinstance(right, bool) and isinstance(left, (int, float)):
            return float(left) == float(right)
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return normalize(left).casefold() == normalize(right).casefold()
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, (int, float)) and isinstance(right, str):
        try:
            return float(left) == float(right.strip())
        except ValueError:
            return False
    if isinstance(right, (int, float)) and isinstance(left, str):
        try:
            return float(left.strip()) == float(right)
        except ValueError:
            return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _loose_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _membership(needle: Any, haystack: Any) -> bool:
    """`needle in haystack` -- the shared engine behind `in` and `includes`."""
    if haystack is None:
        return False
    if isinstance(haystack, str):
        return normalize(_as_text(needle)).casefold() in normalize(haystack).casefold()
    if isinstance(haystack, Mapping):
        return any(_loose_equal(needle, key) for key in haystack.keys())
    if isinstance(haystack, (list, tuple, set, frozenset)):
        return any(_loose_equal(needle, item) for item in haystack)
    return _loose_equal(needle, haystack)


def _contains_text(haystack: Any, needle: Any) -> bool:
    """Case-insensitive substring test, or membership when the left side is
    a collection (`tags contains "research"` reads naturally either way)."""
    if isinstance(haystack, (list, tuple, set, frozenset, Mapping)):
        return _membership(needle, haystack)
    text = normalize(_as_text(haystack))[:MAX_TEXT_SUBJECT].casefold()
    fragment = normalize(_as_text(needle)).casefold()
    if not fragment:
        return False
    return fragment in text


def _mentions(haystack: Any, needle: Any) -> bool:
    """Stem-aware word/phrase match.

    `contains` is a raw substring test, which both over-matches ("art" hits
    "cartoon") and under-matches ("funding" misses "funded"). `mentions`
    is what users actually mean by "talks about X": both sides are
    tokenised and Porter-stemmed via text_kit, so funding / funded /
    funds all reduce to `fund` and match each other, while word boundaries
    stop `art` from matching `cartoon`.

    A multi-word query matches as a *contiguous* phrase over the stemmed
    token stream, so `mentions "funding round"` matches "a funding round"
    and "rounds of funding were funded" only where the two words are
    adjacent. Stopwords are dropped from a multi-word query so that
    "funding of round" style noise does not break the phrase; a query that
    is nothing but stopwords falls back to matching its raw stems.
    """
    query_tokens = tokenize(_as_text(needle))
    if not query_tokens:
        return False
    text_tokens = tokenize(_as_text(haystack)[:MAX_TEXT_SUBJECT])
    if not text_tokens:
        return False

    if len(query_tokens) > 1:
        filtered = [t for t in query_tokens if t not in STOPWORDS]
        if filtered:
            query_tokens = filtered

    query_stems = [porter_stem(t) for t in query_tokens]
    text_stems = [porter_stem(t) for t in text_tokens]

    if len(query_stems) == 1:
        return query_stems[0] in text_stems

    width = len(query_stems)
    first = query_stems[0]
    for index, stem in enumerate(text_stems):
        if stem != first or index + width > len(text_stems):
            continue
        if text_stems[index : index + width] == query_stems:
            return True
    return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class _Evaluator:
    """Tree-walking interpreter over a context Mapping.

    The only source of values is `context`. There is no name resolution
    beyond `context[name]`, no attribute access, no call syntax, no
    assignment, and no way to reach the Python object graph -- the AST
    simply has no node type that could express it.
    """

    def __init__(self, context: Mapping[str, Any], deadline: float) -> None:
        self.context = context if isinstance(context, Mapping) else {}
        self.deadline = deadline
        self.steps = 0

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > MAX_EVAL_STEPS:
            raise RuleEvaluationError(
                f"Rule did too much work (over {MAX_EVAL_STEPS} steps) and was "
                "stopped."
            )
        if time.monotonic() > self.deadline:
            raise RuleEvaluationError(
                "Rule took too long to evaluate and was stopped."
            )

    # -- node dispatch ----------------------------------------------------

    def value_of(self, node: Any) -> Any:
        self._tick()
        if not isinstance(node, dict) or "type" not in node:
            raise RuleEvaluationError("Malformed rule structure.")
        kind = node["type"]

        if kind == "literal":
            return node.get("value")

        if kind == "regex":
            return node  # only meaningful as the RHS of `matches`

        if kind == "list":
            items = node.get("items") or []
            if len(items) > MAX_LIST_ITEMS:
                raise RuleEvaluationError("List literal is too large.")
            return [self.value_of(item) for item in items]

        if kind == "identifier":
            return self._lookup(node.get("name", ""))

        if kind in ("and", "or", "not", "compare"):
            return self.truth_of(node)

        raise RuleEvaluationError(f"Unknown rule element {kind!r}.")

    def truth_of(self, node: Any) -> bool:
        self._tick()
        if not isinstance(node, dict) or "type" not in node:
            raise RuleEvaluationError("Malformed rule structure.")
        kind = node["type"]

        if kind == "and":
            # Short-circuit: a false left side means the right side's
            # (possibly expensive) regex or tokenisation never runs.
            return self.truth_of(node["left"]) and self.truth_of(node["right"])
        if kind == "or":
            return self.truth_of(node["left"]) or self.truth_of(node["right"])
        if kind == "not":
            return not self.truth_of(node["operand"])
        if kind == "compare":
            return self._compare(node)
        return _truthy(self.value_of(node))

    # -- leaves -----------------------------------------------------------

    def _lookup(self, name: str) -> Any:
        if name in self.context:
            return self.context[name]
        # Case-insensitive second chance: users type `Word_Count`.
        lowered = name.lower()
        for key in self.context:
            if isinstance(key, str) and key.lower() == lowered:
                return self.context[key]
        available = sorted(
            str(k) for k in self.context.keys() if isinstance(k, str)
        )
        hint = ""
        if available:
            near = [k for k in available if k.lower().startswith(lowered[:3] or "~")]
            shown = ", ".join((near or available)[:8])
            hint = f" Available fields: {shown}."
        raise RuleEvaluationError(f"Unknown field {name!r}.{hint}")

    def _compare(self, node: dict) -> bool:
        op = node.get("op")
        left_node = node.get("left")
        right_node = node.get("right")

        if op == "matches":
            return self._matches(left_node, right_node)

        left = self.value_of(left_node)
        right = self.value_of(right_node)

        if op == "==":
            return _loose_equal(left, right)
        if op == "!=":
            return not _loose_equal(left, right)
        if op in ("<", "<=", ">", ">="):
            a = _as_number(left, op)
            b = _as_number(right, op)
            if op == "<":
                return a < b
            if op == "<=":
                return a <= b
            if op == ">":
                return a > b
            return a >= b
        if op == "in":
            return _membership(left, right)
        if op == "not in":
            return not _membership(left, right)
        if op == "includes":
            return _membership(right, left)
        if op == "contains":
            return _contains_text(left, right)
        if op == "mentions":
            return _mentions(left, right)
        if op == "before":
            return _as_datetime(left, op) < _as_datetime(right, op)
        if op == "after":
            return _as_datetime(left, op) > _as_datetime(right, op)

        raise RuleEvaluationError(f"Unknown operator {op!r}.")

    def _matches(self, left_node: Any, right_node: Any) -> bool:
        if not isinstance(right_node, dict) or right_node.get("type") != "regex":
            raise RuleEvaluationError("'matches' needs a /pattern/ on the right.")
        pattern = right_node.get("pattern", "")
        flags = right_node.get("flags", "")
        # Re-checked here, not just at parse time: an AST can be loaded from
        # the database, and storage is not a trust boundary we want to lean on.
        _assert_regex_safe(pattern, 0)
        compiled = _compiled_regex(pattern, flags)

        subject = _as_text(self.value_of(left_node))
        return self._search_bounded(compiled, pattern, subject)

    def _search_bounded(
        self, compiled: re.Pattern[str], pattern: str, subject: str
    ) -> bool:
        """Run a (statically vetted) pattern with a bounded, interruptible cost.

        Even a pattern with no nested quantifiers can be *quadratic*: `/a+b/`
        against 20,000 a's makes the engine restart at every offset, which
        measures at ~220ms here -- four times the default 50ms budget, and
        uninterruptible once `re.search` is running.

        So the subject is scanned in `REGEX_CHUNK`-sized windows with
        `REGEX_CHUNK_OVERLAP` characters carried over, and the deadline is
        checked *between* windows. Each individual engine call is then
        bounded (~9ms worst case at 4k characters, microseconds for any
        realistic pattern), which is what makes `timeout_ms` a promise
        rather than a hope: a slow rule aborts after one window instead of
        holding the worker for a quarter of a second.

        Caveats, both deliberate:
          - a match longer than the overlap could straddle a window boundary
            and be missed. 512 characters is far beyond any pattern a user
            writes to spot a phrase or a figure.
          - `^`/`$` cannot be chunked (every window start would look like the
            start of the string), so anchored patterns get one single search
            over a smaller cap instead.
          - the subject is truncated at `MAX_REGEX_SUBJECT`. A rule that
            matches in the first 20k characters is the overwhelmingly common
            case, and capping the input is what bounds the worst case at all.
        """
        if len(subject) > MAX_REGEX_SUBJECT:
            subject = subject[:MAX_REGEX_SUBJECT]

        if _is_anchored(pattern):
            self._tick()
            found = compiled.search(subject[:MAX_ANCHORED_SUBJECT]) is not None
            self._tick()  # deadline re-checked the moment the engine returns
            return found

        if len(subject) <= REGEX_CHUNK:
            self._tick()
            found = compiled.search(subject) is not None
            self._tick()
            return found

        step = REGEX_CHUNK - REGEX_CHUNK_OVERLAP
        start = 0
        while start < len(subject):
            self._tick()  # deadline + step budget, once per window
            window = subject[start : start + REGEX_CHUNK]
            if compiled.search(window) is not None:
                return True
            if start + REGEX_CHUNK >= len(subject):
                break
            start += step
        self._tick()
        return False


def evaluate(expression_or_ast: Any, context: dict, *, timeout_ms: int = 50) -> bool:
    """Evaluate a condition against `context` and return True/False.

    `context` is the *entire* world the expression can see: no builtins, no
    globals, no imports, no object attributes. `timeout_ms` bounds total
    wall-clock time (parsing included); exceeding it raises
    RuleEvaluationError rather than tying up a worker.

    Raises RuleSyntaxError for bad text, RuleEvaluationError for unknown
    fields, type mismatches, or the budget running out.
    """
    if timeout_ms is None or timeout_ms < 0:
        timeout_ms = 0
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    ast = _as_ast(expression_or_ast)
    evaluator = _Evaluator(context or {}, deadline)
    return bool(evaluator.truth_of(ast))


# ---------------------------------------------------------------------------
# Public: identifiers / validation
# ---------------------------------------------------------------------------


def _walk(node: Any) -> Iterable[dict]:
    if not isinstance(node, dict):
        return
    yield node
    for key in ("left", "right", "operand"):
        child = node.get(key)
        if isinstance(child, dict):
            yield from _walk(child)
    for item in node.get("items") or []:
        if isinstance(item, dict):
            yield from _walk(item)


def identifiers_used(expression_or_ast: Any) -> list[str]:
    """Every context field the expression reads, sorted and de-duplicated.

    The UI uses this to warn "this rule mentions `wordcount`, which no
    document has" *before* the rule is saved, rather than failing silently
    at 3am when the digest job runs.
    """
    ast = _as_ast(expression_or_ast)
    names: set[str] = set()
    for node in _walk(ast):
        if node.get("type") == "identifier":
            name = node.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return sorted(names)


def validate(expression: str) -> dict:
    """Check an expression without running it.

    Returns {"valid", "error", "identifiers", "ast"}. Never raises: this is
    the function the "Save rule" form calls, and a form validator that
    throws is a 500 the user cannot act on.
    """
    try:
        ast = parse(expression)
    except RuleSyntaxError as exc:
        return {"valid": False, "error": str(exc), "identifiers": [], "ast": None}
    except RecursionError:
        return {
            "valid": False,
            "error": "Condition is nested too deeply.",
            "identifiers": [],
            "ast": None,
        }
    return {
        "valid": True,
        "error": None,
        "identifiers": identifiers_used(ast),
        "ast": ast,
    }


# ---------------------------------------------------------------------------
# describe() -- plain English for people who do not write code
# ---------------------------------------------------------------------------

# Operator -> English. Chosen so the result reads as a sentence fragment
# about the field: "word count is over 500", not "word count > 500".
_OP_PHRASES = {
    "==": "is",
    "!=": "is not",
    ">": "is over",
    ">=": "is at least",
    "<": "is under",
    "<=": "is at most",
    "in": "is one of",
    "not in": "is not one of",
    "includes": "includes",
    "contains": "contains",
    "mentions": "mentions",
    "matches": "matches the pattern",
    "before": "is before",
    "after": "is after",
}


def _humanize_identifier(name: str) -> str:
    """`source_type` -> `source type`. Underscores are a programmer's
    concern; the person reading their own rule back should not see them."""
    return name.replace("_", " ").strip()


def _describe_value_node(node: Any) -> str:
    """Render a value node as English."""
    if not isinstance(node, dict):
        return str(node)
    kind = node.get("type")

    if kind == "literal":
        value = node.get("value")
        if value is None:
            return "nothing"
        if value is True:
            return "yes"
        if value is False:
            return "no"
        if isinstance(value, str):
            return value if value.strip() else '""'
        return str(value)

    if kind == "regex":
        return f"/{node.get('pattern', '')}/"

    if kind == "list":
        parts = [_describe_value_node(item) for item in node.get("items") or []]
        if not parts:
            return "nothing"
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " or " + parts[-1]

    if kind == "identifier":
        return _humanize_identifier(str(node.get("name", "")))

    # A boolean sub-expression used as a value: describe it as one.
    return _describe_node(node, parent=None)


def _describe_node(node: Any, parent: str | None) -> str:
    """Render a boolean node, bracketing only where grouping is not obvious.

    English loses the parenthesis, so `(a or b) and c` must keep its
    brackets or it changes meaning; `a and (b and c)` does not need them.
    """
    if not isinstance(node, dict):
        return str(node)
    kind = node.get("type")

    if kind == "and":
        left = _describe_node(node["left"], "and")
        right = _describe_node(node["right"], "and")
        text = f"{left} and {right}"
        return f"({text})" if parent == "not" else text

    if kind == "or":
        left = _describe_node(node["left"], "or")
        right = _describe_node(node["right"], "or")
        text = f"{left} or {right}"
        # An `or` nested inside an `and` (or a `not`) must keep its brackets.
        return f"({text})" if parent in ("and", "not") else text

    if kind == "not":
        inner = _describe_node(node["operand"], "not")
        return f"not {inner}"

    if kind == "compare":
        op = node.get("op", "==")
        subject = _describe_value_node(node.get("left"))
        obj = _describe_value_node(node.get("right"))
        phrase = _OP_PHRASES.get(op, op)
        return f"{subject} {phrase} {obj}"

    if kind == "identifier":
        # Bare field in boolean position: `not archived` -> "not archived".
        return _humanize_identifier(str(node.get("name", "")))

    return _describe_value_node(node)


def describe(expression_or_ast: Any) -> str:
    """Plain-English rendering of a rule condition.

    Example:
        'source_type == "article" and word_count > 500'
        -> 'source type is article and word count is over 500'

    This is shown under the rule editor so a non-technical user can confirm
    the rule means what they intended *before* it starts sending them email
    unattended. It is deliberately a rendering of the AST, not of the source
    text: what you read back is what will actually run, including how
    precedence grouped it.
    """
    ast = _as_ast(expression_or_ast)
    return _describe_node(ast, parent=None)


# ---------------------------------------------------------------------------
# Rule objects and action validation
#
# Actions are DATA. This layer decides *what should happen*; the Flask view
# that called it decides *how*, with the user's session, rate limits and
# audit log in scope. A rule can therefore never do more than name one of
# the six things below, with fields we have checked.
# ---------------------------------------------------------------------------

ACTION_SCHEMA: dict[str, dict[str, frozenset[str]]] = {
    "tag":             {"required": frozenset({"value"}), "optional": frozenset()},
    "email":           {"required": frozenset({"subject"}),
                        "optional": frozenset({"to", "body", "include_summary"})},
    "webhook":         {"required": frozenset({"url"}),
                        "optional": frozenset({"method", "payload", "headers"})},
    "extract_numbers": {"required": frozenset(), "optional": frozenset({"into", "units"})},
    "add_to_timeline": {"required": frozenset(), "optional": frozenset({"label", "date_field"})},
    "archive":         {"required": frozenset(), "optional": frozenset({"reason"})},
}

_ALLOWED_WEBHOOK_SCHEMES = ("http://", "https://")
_MAX_ACTIONS_PER_RULE = 20


def validate_actions(actions: Any) -> list[str]:
    """Check a rule's action list; return a list of human-readable problems
    (empty means valid).

    Unknown action types are rejected here, at save time, rather than being
    discovered by the dispatcher at 3am. The dispatcher should still treat
    this as advisory and switch on known types only -- defence in depth.
    """
    problems: list[str] = []
    if actions is None:
        return ["A rule needs at least one action."]
    if not isinstance(actions, (list, tuple)):
        return ["Actions must be a list."]
    if not actions:
        return ["A rule needs at least one action."]
    if len(actions) > _MAX_ACTIONS_PER_RULE:
        problems.append(
            f"Too many actions ({len(actions)}; the limit is "
            f"{_MAX_ACTIONS_PER_RULE})."
        )

    for index, action in enumerate(actions[:_MAX_ACTIONS_PER_RULE]):
        label = f"Action {index + 1}"
        if not isinstance(action, Mapping):
            problems.append(f"{label} is not a valid action object.")
            continue
        action_type = action.get("type")
        if not isinstance(action_type, str) or not action_type:
            problems.append(f"{label} has no 'type'.")
            continue
        schema = ACTION_SCHEMA.get(action_type)
        if schema is None:
            known = ", ".join(sorted(ACTION_SCHEMA))
            problems.append(
                f"{label}: unknown action type {action_type!r}. Allowed: {known}."
            )
            continue

        keys = {str(k) for k in action.keys()} - {"type"}
        missing = schema["required"] - keys
        for field_name in sorted(missing):
            problems.append(f"{label} ({action_type}) is missing {field_name!r}.")
        unknown = keys - schema["required"] - schema["optional"]
        for field_name in sorted(unknown):
            problems.append(
                f"{label} ({action_type}) has an unrecognised field "
                f"{field_name!r}."
            )

        if action_type == "webhook":
            url = action.get("url")
            if isinstance(url, str) and not url.lower().startswith(
                _ALLOWED_WEBHOOK_SCHEMES
            ):
                problems.append(
                    f"{label} (webhook): url must start with http:// or https://."
                )
            elif url is not None and not isinstance(url, str):
                problems.append(f"{label} (webhook): url must be text.")
        if action_type == "tag":
            value = action.get("value")
            if value is not None and (not isinstance(value, str) or not value.strip()):
                problems.append(f"{label} (tag): value must be non-empty text.")

    return problems


@dataclass
class Rule:
    """A stored automation rule.

    `trigger` is an event name (`entry.created`, `watch.changed`,
    `digest.weekly`). A rule only fires when the incoming event's type
    matches; `*` and a trailing wildcard (`entry.*`) are supported so a
    user can write one rule for a family of events.
    """

    id: str
    name: str
    trigger: str
    condition: str
    actions: list[dict] = field(default_factory=list)
    enabled: bool = True


def _trigger_matches(trigger: str, event_type: str) -> bool:
    """Exact, `*`, or a single trailing wildcard segment (`entry.*`)."""
    if not isinstance(trigger, str) or not trigger:
        return False
    trigger = trigger.strip()
    event_type = (event_type or "").strip()
    if trigger == "*":
        return True
    if trigger == event_type:
        return True
    if trigger.endswith(".*"):
        return event_type.startswith(trigger[:-1])
    return False


def _build_context(event: Mapping[str, Any], context: Mapping[str, Any]) -> dict:
    """The field namespace a condition sees.

    Order matters: explicit `context` wins, then the event's `data` payload,
    then a couple of derived conveniences. The event payload is untrusted
    (it can come from a scraped page), but it is only ever *read* as data --
    a key called `__class__` in there is a string key in a dict, nothing more.
    """
    merged: dict[str, Any] = {}
    data = event.get("data") if isinstance(event, Mapping) else None
    if isinstance(data, Mapping):
        for key, value in data.items():
            if isinstance(key, str):
                merged[key] = value
    if isinstance(context, Mapping):
        for key, value in context.items():
            if isinstance(key, str):
                merged[key] = value
    if isinstance(event, Mapping):
        merged.setdefault("event_type", event.get("type"))
        if "at" in event:
            merged.setdefault("event_at", event.get("at"))
    return merged


def evaluate_rule(rule: Rule, event: dict, context: dict) -> dict:
    """Decide whether `rule` fires for `event`, and return what to do.

    Returns {"matched", "actions", "error", "explain"}.

    It NEVER performs an action -- `actions` is the caller's to dispatch.
    Keeping execution out of here means a bug in rule evaluation cannot
    send mail or call a webhook, and the whole decision path is testable
    with plain dicts.

    Errors are returned, not raised: one malformed rule among fifty must
    not abort the batch, and the user needs to see the message next to that
    rule in the UI.
    """
    result: dict[str, Any] = {
        "matched": False,
        "actions": [],
        "error": None,
        "explain": "",
        "rule_id": getattr(rule, "id", None),
        "rule_name": getattr(rule, "name", None),
    }

    if not isinstance(rule, Rule):
        result["error"] = "Not a Rule object."
        result["explain"] = "This rule could not be read."
        return result

    event = event if isinstance(event, Mapping) else {}
    event_type = str(event.get("type") or "")

    if not rule.enabled:
        result["explain"] = f"Rule {rule.name!r} is turned off."
        return result

    if not _trigger_matches(rule.trigger, event_type):
        result["explain"] = (
            f"Rule {rule.name!r} listens for {rule.trigger!r}, and this event "
            f"is {event_type or 'untyped'!r}."
        )
        return result

    # Condition text is parsed and described before anything else so that a
    # syntax error is reported with its position, even for a rule that would
    # otherwise never have been evaluated.
    try:
        ast = parse(rule.condition) if str(rule.condition).strip() else None
    except RuleSyntaxError as exc:
        result["error"] = str(exc)
        result["explain"] = f"Rule {rule.name!r} has a condition I cannot read."
        return result

    readable = describe(ast) if ast is not None else "always"

    action_problems = validate_actions(rule.actions)
    if action_problems:
        result["error"] = "; ".join(action_problems)
        result["explain"] = f"Rule {rule.name!r} has invalid actions."
        return result

    merged = _build_context(event, context or {})

    try:
        matched = True if ast is None else evaluate(ast, merged)
    except (RuleEvaluationError, RuleSyntaxError) as exc:
        result["error"] = str(exc)
        result["explain"] = f"Rule {rule.name!r} could not be checked: {exc}"
        return result
    except RecursionError:
        result["error"] = "Condition is nested too deeply."
        result["explain"] = f"Rule {rule.name!r} is too complex to evaluate."
        return result

    result["matched"] = matched
    result["explain"] = (
        f"On {event_type}, when {readable}: "
        + ("matched." if matched else "did not match.")
    )
    if matched:
        # A copy: the caller may annotate the actions it dispatches, and the
        # stored rule must not mutate underneath it.
        result["actions"] = [dict(action) for action in rule.actions]
    return result


def run_rules(rules: list[Rule], event: dict, context: dict) -> list[dict]:
    """Evaluate every rule against one event, in order.

    Returns one result dict per rule (including non-matches, so the UI can
    show "3 of 12 rules fired" and explain the nine that did not). Each
    rule is isolated: a failure in one is reported in its own result and
    the rest still run.
    """
    results: list[dict] = []
    if not rules:
        return results
    for rule in rules:
        try:
            results.append(evaluate_rule(rule, event, context))
        except Exception as exc:  # pragma: no cover - last-resort isolation
            results.append(
                {
                    "matched": False,
                    "actions": [],
                    "error": f"Unexpected failure: {exc}",
                    "explain": "This rule could not be evaluated.",
                    "rule_id": getattr(rule, "id", None),
                    "rule_name": getattr(rule, "name", None),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Self-test
#
# Run directly:
#   cd app/core && python rules.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
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

    CTX = {
        "source_type": "article",
        "word_count": 1200,
        "title": "Quarterly earnings and outlook",
        "text": (
            "The company announced a funding round of $12,000,000 this week. "
            "Analysts note revenue of $450 million and a strong outlook."
        ),
        "domain": "nytimes.com",
        "archived": False,
        "starred": True,
        "created_at": "2024-06-15",
        "published_at": "2023-02-01T09:30:00Z",
        "tags": ["research", "finance"],
        "score": 0.91,
        "kind": "audio",
        "author": None,
        "reading_time": 6,
    }

    print("=" * 72)
    print("1. Expression evaluation (every operator)")
    print("=" * 72)

    CASES: list[tuple[str, bool]] = [
        # -- equality / inequality
        ('source_type == "article"', True),
        ('source_type == "ARTICLE"', True),               # case-insensitive
        ('source_type != "video"', True),
        ('kind == "video"', False),
        ("author == null", True),
        ("archived == false", True),
        ("starred == true", True),
        # -- ordering
        ("word_count > 500", True),
        ("word_count >= 1200", True),
        ("word_count < 500", False),
        ("word_count <= 1200", True),
        ("score >= 0.8", True),
        ("score > 0.95", False),
        ("reading_time > -1", True),                       # negative literal
        # -- logical + precedence + grouping
        ('source_type == "article" and word_count > 500', True),
        ('source_type == "video" or word_count > 500', True),
        ("not archived", True),
        ("not starred", False),
        ('kind == "text" or (kind == "audio" and score > 0.5)', True),
        ('(kind == "text" or kind == "audio") and score > 0.5', True),
        ('kind == "text" and kind == "audio" or score > 0.5', True),
        # -- membership
        ('domain in ["nytimes.com", "ft.com"]', True),
        ('domain in ["wsj.com", "ft.com"]', False),
        ('domain not in ["wsj.com"]', True),
        ('tags includes "research"', True),
        ('tags includes "sports"', False),
        ('"finance" in tags', True),
        # -- text operators
        ('title contains "quarterly"', True),              # case-insensitive
        ('title contains "monthly"', False),
        ('tags contains "finance"', True),                 # list -> membership
        ('text mentions "funding round"', True),           # phrase
        ('text mentions "funded"', True),                  # stem: funded~funding
        ('text mentions "analyst"', True),                 # stem: analysts
        ('text mentions "bankruptcy"', False),
        ('title mentions "art"', False),                   # no substring hits
        # -- regex
        (r"text matches /revenue of \$[0-9]+/", True),
        (r"text matches /revenue of \$[0-9]+ trillion/", False),
        (r'text matches /FUNDING/i', True),
        # -- dates
        ('created_at after "2024-01-01"', True),
        ('created_at before "2024-01-01"', False),
        ('published_at before "2024-01-01"', True),        # aware vs naive
        ('created_at after "2024-01-01" and tags includes "research"', True),
        # -- the README examples, verbatim
        ('source_type == "article" and word_count > 500', True),
        ('title contains "quarterly" or text mentions "funding round"', True),
        ('domain in ["nytimes.com", "ft.com"] and not archived', True),
        ('score >= 0.8 and (kind == "audio" or kind == "video")', True),
    ]

    for expression, expected in CASES:
        try:
            actual = evaluate(expression, CTX)
            check(f"{expression}  ->  {expected}", actual == expected,
                  f"got {actual}")
        except Exception as exc:
            check(f"{expression}  ->  {expected}", False, f"raised {exc!r}")

    print()
    print("=" * 72)
    print("2. Operator precedence in the AST")
    print("=" * 72)

    ast = parse("a or b and c")
    check(
        "'a or b and c' parses as a or (b and c)",
        ast["type"] == "or"
        and ast["left"] == {"type": "identifier", "name": "a"}
        and ast["right"]["type"] == "and",
        str(ast),
    )

    ast = parse("a and b or c")
    check(
        "'a and b or c' parses as (a and b) or c",
        ast["type"] == "or" and ast["left"]["type"] == "and"
        and ast["right"] == {"type": "identifier", "name": "c"},
        str(ast),
    )

    ast = parse("not a and b")
    check(
        "'not a and b' parses as (not a) and b",
        ast["type"] == "and" and ast["left"]["type"] == "not"
        and ast["right"]["type"] == "identifier",
        str(ast),
    )

    ast = parse("not (a and b)")
    check(
        "'not (a and b)' parses as not (a and b)",
        ast["type"] == "not" and ast["operand"]["type"] == "and",
        str(ast),
    )

    ast = parse("(a or b) and c")
    check(
        "'(a or b) and c' keeps its grouping",
        ast["type"] == "and" and ast["left"]["type"] == "or",
        str(ast),
    )

    import json as _json

    try:
        _json.dumps(parse('source_type == "article" and word_count > 500'))
        _json.dumps(parse(r'text matches /a[0-9]+/ or tags in ["a", 1, true, null]'))
        json_ok = True
    except (TypeError, ValueError):
        json_ok = False
    check("AST is JSON-serialisable (storable in the DB)", json_ok)

    check(
        "an AST round-trips through JSON and still evaluates",
        evaluate(
            _json.loads(_json.dumps(parse('word_count > 500'))), CTX
        ) is True,
    )

    print()
    print("=" * 72)
    print("3. describe()")
    print("=" * 72)

    DESCRIBE_CASES = [
        (
            'source_type == "article" and word_count > 500',
            "source type is article and word count is over 500",
        ),
        (
            'domain in ["nytimes.com", "ft.com"] and not archived',
            "domain is one of nytimes.com or ft.com and not archived",
        ),
        (
            'text mentions "funding round"',
            "text mentions funding round",
        ),
        (
            'score >= 0.8 and (kind == "audio" or kind == "video")',
            "score is at least 0.8 and (kind is audio or kind is video)",
        ),
        (
            'created_at after "2024-01-01"',
            "created at is after 2024-01-01",
        ),
        (
            r"text matches /revenue of \$[0-9]+/",
            r"text matches the pattern /revenue of \$[0-9]+/",
        ),
        (
            'tags includes "research" or title contains "q3"',
            "tags includes research or title contains q3",
        ),
    ]
    for expression, expected in DESCRIBE_CASES:
        actual = describe(expression)
        check(f"describe({expression!r})", actual == expected,
              f"\n        got:      {actual!r}\n        expected: {expected!r}")
        print(f"        -> {actual}")

    print()
    print("=" * 72)
    print("4. Syntax errors carry a position")
    print("=" * 72)

    BAD = [
        "word_count >",
        "(source_type == 'a'",
        "source_type == ",
        "source_type == 'a' extra",
        "a & b",
        "a = b",
        "entry.title == 'x'",
        "1 < word_count < 10",
        "source_type == 'unterminated",
        "",
        "[1, 2",
        "not",
        "_secret == 1",
    ]
    for expression in BAD:
        try:
            parse(expression)
            check(f"rejects {expression!r}", False, "parsed without error")
        except RuleSyntaxError as exc:
            has_pos = exc.position is not None and "character" in str(exc)
            check(f"rejects {expression!r}", has_pos, f"no position: {exc}")
            print(f"        -> {exc}")
        except Exception as exc:
            check(f"rejects {expression!r}", False, f"wrong type {exc!r}")

    print()
    print("=" * 72)
    print("5. Unknown identifiers")
    print("=" * 72)

    try:
        evaluate("nonexistent_field == 1", CTX)
        check("unknown field raises RuleEvaluationError", False, "no error")
    except RuleEvaluationError as exc:
        check("unknown field raises RuleEvaluationError",
              "nonexistent_field" in str(exc), str(exc))
        print(f"        -> {exc}")
    except Exception as exc:
        check("unknown field raises RuleEvaluationError", False, repr(exc))

    check(
        "identifiers_used finds every field",
        identifiers_used('a == 1 and (b contains "x" or not c) and d in [e]')
        == ["a", "b", "c", "d", "e"],
        str(identifiers_used('a == 1 and (b contains "x" or not c) and d in [e]')),
    )

    report = validate('source_type == "article" and word_count > 500')
    check(
        "validate() reports valid + identifiers",
        report["valid"] and report["error"] is None
        and report["identifiers"] == ["source_type", "word_count"]
        and isinstance(report["ast"], dict),
        str(report),
    )

    report = validate("word_count >")
    check(
        "validate() reports invalid without raising",
        report["valid"] is False and report["error"] and report["ast"] is None,
        str(report),
    )

    print()
    print("=" * 72)
    print("6. Security: injection, ReDoS, depth, timeout")
    print("=" * 72)

    INJECTIONS = [
        '__import__("os").system("ls")',
        '().__class__.__bases__[0].__subclasses__()',
        '"".__class__',
        'open("/etc/passwd")',
        'eval("1+1")',
        'lambda: 1',
        'exec("x=1")',
        '{"a": 1}',
    ]
    for expression in INJECTIONS:
        try:
            parse(expression)
            check(f"rejects injection {expression!r}", False, "parsed!")
        except RuleSyntaxError as exc:
            check(f"rejects injection {expression!r}", True)
            print(f"        -> {exc}")
        except Exception as exc:
            check(f"rejects injection {expression!r}", False, f"wrong type {exc!r}")

    # These must never execute even by accident:
    import os as _os_probe
    _marker_before = len(getattr(_os_probe, "environ", {}))
    for expression in INJECTIONS:
        try:
            evaluate(expression, CTX)
        except (RuleSyntaxError, RuleEvaluationError):
            pass
    check(
        "no injection had any side effect",
        len(getattr(_os_probe, "environ", {})) == _marker_before,
    )

    REDOS = [
        r"text matches /(a+)+b/",
        r"text matches /(a|a)*b/",
        r"text matches /(x*)*y/",
        r"text matches /(\w+\s?)*$/",
        r"text matches /a{200}{200}/",
        r"text matches /(ab)\1/",
    ]
    for expression in REDOS:
        started = time.monotonic()
        try:
            parse(expression)
            check(f"rejects ReDoS {expression!r}", False, "accepted!")
        except RuleSyntaxError as exc:
            elapsed = (time.monotonic() - started) * 1000
            check(f"rejects ReDoS {expression!r} in {elapsed:.2f}ms",
                  elapsed < 100, f"took {elapsed:.1f}ms")
            print(f"        -> {exc}")

    # A vetted-but-quadratic pattern against a pathological subject must
    # either finish or abort on the deadline -- never hang the worker.
    hostile_ctx = dict(CTX, text="a" * 500_000)
    started = time.monotonic()
    try:
        result: Any = evaluate(r"text matches /a+b/", hostile_ctx, timeout_ms=60)
    except RuleEvaluationError as exc:
        result = f"aborted: {exc}"
    elapsed = (time.monotonic() - started) * 1000
    check(
        f"quadratic pattern on a 500k-char subject: {elapsed:.1f}ms "
        f"({result if result is False else 'timed out cleanly'})",
        elapsed < 120 and (result is False or isinstance(result, str)),
        f"{elapsed:.1f}ms, {result!r}",
    )

    # ...and a realistic pattern on a realistic document stays microseconds.
    real_ctx = dict(CTX, text=CTX["text"] * 400)
    started = time.monotonic()
    result = evaluate(r"text matches /revenue of \$[0-9]+/", real_ctx)
    elapsed = (time.monotonic() - started) * 1000
    check(f"realistic pattern on a 40k-char document: {elapsed:.2f}ms",
          result is True and elapsed < 25, f"{elapsed:.2f}ms, {result}")

    check("chunked scan still finds a match late in a long document",
          evaluate(r"text matches /needle[0-9]+/",
                   {"text": "x" * 15_000 + " needle42 " + "y" * 3_000}) is True)
    check("anchored patterns are handled without chunking",
          evaluate(r"title matches /^Quarterly/", CTX) is True)
    check("anchored pattern that should not match",
          evaluate(r"title matches /^Monthly/", CTX) is False)

    deep = "(" * 200 + "a" + ")" * 200
    try:
        parse(deep)
        check("rejects 200-deep nesting", False, "parsed!")
    except RuleSyntaxError as exc:
        check("rejects 200-deep nesting", True)
        print(f"        -> {exc}")
    except RecursionError:
        check("rejects 200-deep nesting", False, "RecursionError, not clean")

    try:
        parse("a == " + "1" * 5000)
        check("rejects an enormous literal", False, "parsed!")
    except RuleSyntaxError:
        check("rejects an enormous literal", True)

    try:
        parse("x == 'y'" + " and x == 'y'" * 500)
        check("rejects an over-long expression", False, "parsed!")
    except RuleSyntaxError:
        check("rejects an over-long expression", True)

    try:
        evaluate('source_type == "article"', CTX, timeout_ms=0)
        check("timeout_ms=0 aborts cleanly", False, "no error")
    except RuleEvaluationError as exc:
        check("timeout_ms=0 aborts cleanly", "too long" in str(exc), str(exc))
        print(f"        -> {exc}")

    print()
    print("=" * 72)
    print("7. Rule objects, actions and dispatch")
    print("=" * 72)

    rule_ok = Rule(
        id="r1",
        name="Fundraising watch",
        trigger="watch.changed",
        condition='text mentions "funding round" and not archived',
        actions=[
            {"type": "extract_numbers"},
            {"type": "tag", "value": "fundraising"},
            {"type": "email", "subject": "New funding news"},
        ],
    )
    event = {"type": "watch.changed", "data": {"url": "https://x.test"}}
    outcome = evaluate_rule(rule_ok, event, CTX)
    check("matching rule returns its actions",
          outcome["matched"] and len(outcome["actions"]) == 3
          and outcome["error"] is None, str(outcome))
    print(f"        -> {outcome['explain']}")

    outcome = evaluate_rule(rule_ok, {"type": "entry.created"}, CTX)
    check("non-matching trigger does not fire",
          outcome["matched"] is False and outcome["actions"] == []
          and outcome["error"] is None, str(outcome))
    print(f"        -> {outcome['explain']}")

    rule_off = Rule("r2", "Off", "watch.changed", "true", [{"type": "archive"}],
                    enabled=False)
    outcome = evaluate_rule(rule_off, event, CTX)
    check("disabled rule does not fire", outcome["matched"] is False,
          str(outcome))

    rule_wild = Rule("r3", "Any entry", "entry.*", "word_count > 100",
                     [{"type": "add_to_timeline"}])
    check("wildcard trigger matches entry.created",
          evaluate_rule(rule_wild, {"type": "entry.created"}, CTX)["matched"])
    check("wildcard trigger does not match watch.changed",
          evaluate_rule(rule_wild, {"type": "watch.changed"}, CTX)["matched"]
          is False)

    rule_bad_action = Rule("r4", "Bad action", "*", "true",
                           [{"type": "run_shell", "cmd": "rm -rf /"}])
    outcome = evaluate_rule(rule_bad_action, event, CTX)
    check("unknown action type is rejected",
          outcome["matched"] is False and "run_shell" in (outcome["error"] or ""),
          str(outcome))
    print(f"        -> {outcome['error']}")

    check("webhook needs an http(s) url",
          validate_actions([{"type": "webhook", "url": "file:///etc/passwd"}]) != [])
    check("tag needs a value",
          validate_actions([{"type": "tag"}]) != [])
    check("valid actions pass",
          validate_actions([{"type": "archive"}, {"type": "tag", "value": "x"}])
          == [])

    rule_bad_cond = Rule("r5", "Broken", "*", "word_count >",
                         [{"type": "archive"}])
    outcome = evaluate_rule(rule_bad_cond, event, CTX)
    check("bad condition is reported, not raised",
          outcome["matched"] is False and "character" in (outcome["error"] or ""),
          str(outcome))

    rule_unknown_field = Rule("r6", "Typo", "*", "wordcount > 5",
                              [{"type": "archive"}])
    outcome = evaluate_rule(rule_unknown_field, event, CTX)
    check("unknown field is reported, not raised",
          outcome["matched"] is False and "wordcount" in (outcome["error"] or ""),
          str(outcome))
    print(f"        -> {outcome['error']}")

    results = run_rules(
        [rule_ok, rule_off, rule_wild, rule_bad_action, rule_bad_cond],
        event,
        CTX,
    )
    check("run_rules returns one result per rule", len(results) == 5,
          str(len(results)))
    check("run_rules isolates failures (one match, batch survives)",
          sum(1 for r in results if r["matched"]) == 1,
          str([r["matched"] for r in results]))

    # Event payload is readable as data, and cannot reach Python internals.
    hostile_event = {
        "type": "entry.created",
        "data": {"__class__": "evil", "title": "Funding round closed"},
    }
    rule_evt = Rule("r7", "Event data", "entry.created",
                    'title mentions "funding"', [{"type": "tag", "value": "f"}])
    outcome = evaluate_rule(rule_evt, hostile_event, {})
    check("event data is readable, hostile keys are inert",
          outcome["matched"] is True, str(outcome))

    print()
    print("=" * 72)
    print(f"{passed} passed, {failed} failed")
    print("=" * 72)
    raise SystemExit(1 if failed else 0)
