from __future__ import annotations

import re

_PROFANITY: frozenset[str] = frozenset(
    {
        "fuck",
        "shit",
        "cunt",
        "bitch",
        "asshole",
        "bastard",
        "piss",
        "wank",
    }
)

_CODING_SIGNALS: frozenset[str] = frozenset(
    {
        "function",
        "method",
        "class",
        "implement",
        "code",
        "script",
        "program",
        "algorithm",
        "api",
        "endpoint",
        "database",
        "query",
        "parse",
        "sort",
        "dict",
        "integer",
        "float",
        "boolean",
        "variable",
        "import",
        "module",
        "package",
        "framework",
        "refactor",
        "debug",
        "bug",
        "exception",
        "async",
        "await",
        "http",
        "json",
        "sql",
        "html",
        "css",
        "regex",
        "fetch",
        "auth",
        "token",
        "encrypt",
        "compile",
        "deploy",
        "docker",
        "git",
        "python",
        "javascript",
        "typescript",
        "rust",
        "java",
        "c++",
        "bash",
        "shell",
        "unittest",
        "pytest",
        "lambda",
        "recursion",
        "iterator",
        "generator",
    }
)

_GIBBERISH: re.Pattern[str] = re.compile(r"^[^a-zA-Z\s]*$|^(.)\1{4,}$")


def validate(prompt: str) -> str | None:
    """Return error message if the prompt should be rejected, else None."""
    words = prompt.strip().split()

    if not words:
        return "Prompt is empty."

    lower = prompt.lower()
    tokens = set(re.findall(r"\b\w+\b", lower))

    if tokens & _PROFANITY:
        return "Prompt contains inappropriate language."

    if _GIBBERISH.match(prompt.strip()):
        return "Prompt appears to be gibberish."

    if len(words) < 4 and not (tokens & _CODING_SIGNALS):
        return "Prompt is too vague to analyse. Describe a coding task."

    if len(words) < 12 and not (tokens & _CODING_SIGNALS):
        return "Prompt doesn't appear to be a coding task. Choptimize analyses prompts for code generation."

    return None
