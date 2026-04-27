from __future__ import annotations

import re


def split_columns(line: str) -> list[str]:
    s = line.strip("\n\r")
    if "\t" in s:
        return [c.strip() for c in s.split("\t")]
    cols = [c.strip() for c in re.split(r"\s{2,}", s) if c.strip()]
    if len(cols) >= 2:
        return cols
    return [c.strip() for c in re.split(r"\s+", s) if c.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def pick_col_index(headers: list[str], candidates: list[str]) -> int | None:
    hnorm = [_norm(h) for h in headers]
    for cand in candidates:
        c = _norm(cand)
        for i, h in enumerate(hnorm):
            if c in h:
                return i
    return None


def parse_3col_numeric_table(text: str) -> tuple[list[str], list[list[str]]]:
    """
    Parse a whitespace/tab separated table.
    Returns (headers, rows). Headers may be empty.
    """

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []

    # If first non-empty line contains any letters, treat it as header
    first_cols = split_columns(lines[0])
    if any(re.search(r"[A-Za-z]", c) for c in first_cols):
        headers = first_cols
        data_lines = lines[1:]
    else:
        headers = []
        data_lines = lines

    rows = [split_columns(ln) for ln in data_lines if ln.strip()]
    return headers, rows

