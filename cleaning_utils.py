"""Minimal internal cleaning utilities extracted from original csv_excel_cleaner.

Only the pieces required by the pipeline are retained to reduce surface area:
  - Header detection (detect_header_row, build_headers)
  - Row/blank filtering (_drop_fully_blank_rows)
  - Numeric detection & normalization (is_numeric_candidate, normalize_numeric_series)
  - Basic text normalization helpers (_normalize_whitespace_and_minus)

This allows removal of the external csv_excel_cleaner package for a leaner codebase.
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import re
import numpy as np
import pandas as pd

# -----------------------------
# Null tokens (lowercased set)
# -----------------------------
NULL_TOKENS = {
    "",
    "-",
    "--",
    "—",
    "–",
    "n/a",
    "n.a.",
    "n.a",
    "na",
    "nan",
    "none",
    "null",
    "nil",
    "missing",
    "no data",
    "not available",
    "not applicable",
    "#n/a",
    "#null!",
    "#div/0!",
    "#value!",
    "#ref!",
    "#name?",
    "#num!",
    "#calc!",
    "#getting_data",
    "(null)",
    "(empty)",
    "(blank)",
    "(na)",
    "(n/a)",
}
_NULL_TOKENS_LOWER = {t.lower() for t in NULL_TOKENS}

CURRENCY_PATTERN = re.compile(
    r"(?:\$|€|£|¥|₺|₩|₹|₦|₽|₫|₪|₴|₡|₲|₱|₵|₭|₸|R\$|C\$|A\$|CHF)"
)
SEPARATORS_PATTERN = re.compile(r"[\u00A0\u2000-\u200B'\s]+")
KMB_SUFFIX_PATTERN = re.compile(r"\s*([kKmMbB])\s*$")


def _normalize_whitespace_and_minus(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace("\u2212", "-", regex=False)
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(r"[\u2000-\u200B]", " ", regex=True)
    return s


def _drop_fully_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    is_na = df.isna()
    lowered = df.astype(str).apply(lambda s: s.str.strip().str.lower(), axis=0)
    is_token_blank = lowered.isin(_NULL_TOKENS_LOWER)
    is_blank = is_na | is_token_blank
    keep_mask = ~is_blank.all(axis=1)
    return df.loc[keep_mask].reset_index(drop=True)


def _handle_percent(series: pd.Series):
    s = series
    mask = s.str.contains("%", na=False)
    s = s.str.replace("%", "", regex=False)
    return s, mask


def _detect_negatives(series: pd.Series):
    s = series
    mask_paren = s.str.match(r"^\(.*\)$", na=False)
    s = s.mask(mask_paren, s.str.replace(r"^[\(](.*)[\)]$", r"\1", regex=True))
    mask_trail = s.str.endswith("-", na=False)
    s = s.mask(mask_trail, s.str[:-1])
    negative = mask_paren | mask_trail
    return s, negative


def _extract_kmb_multiplier(series: pd.Series):
    matches = series.str.extract(KMB_SUFFIX_PATTERN.pattern)
    suffix = matches[0].fillna("")
    mult = pd.Series(1.0, index=series.index, dtype=float)
    mult = mult.mask(suffix.str.lower() == "k", 1e3)
    mult = mult.mask(suffix.str.lower() == "m", 1e6)
    mult = mult.mask(suffix.str.lower() == "b", 1e9)
    cleaned = series.str.replace(KMB_SUFFIX_PATTERN.pattern, "", regex=True)
    return cleaned, mult


def _strip_currency_and_separators(series: pd.Series) -> pd.Series:
    s = series.str.replace(CURRENCY_PATTERN.pattern, "", regex=True)
    s = s.str.replace(SEPARATORS_PATTERN.pattern, "", regex=True)
    return s


def _normalize_decimal_thousands(series: pd.Series) -> pd.Series:
    s = series
    has_dot = s.str.contains(r"\.", na=False)
    has_comma = s.str.contains(",", na=False)
    both = has_dot & has_comma
    last_dot = s.str.rfind(".")
    last_comma = s.str.rfind(",")
    decimal_is_dot = both & (last_dot > last_comma)
    decimal_is_comma = both & (last_comma > last_dot)
    out = s.copy()
    mask = decimal_is_dot
    out = out.mask(mask, out.where(~mask, out.str.replace(",", "", regex=False)))
    mask = decimal_is_comma
    tmp = out.where(~mask, out.str.replace(".", "", regex=False))
    tmp = tmp.where(~mask, tmp.str.replace(",", ".", regex=False))
    out = out.mask(mask, tmp)
    only_comma = has_comma & (~has_dot)
    looks_decimal_comma = only_comma & s.str.contains(r",\d{1,2}$", na=False)
    out = out.mask(
        looks_decimal_comma,
        out.where(~looks_decimal_comma, out.str.replace(",", ".", regex=False)),
    )
    mask = only_comma & (~looks_decimal_comma)
    out = out.mask(mask, out.where(~mask, out.str.replace(",", "", regex=False)))
    only_dot = has_dot & (~has_comma)
    looks_decimal_dot = only_dot & s.str.contains(r"\.\d+$", na=False)
    mask = only_dot & (~looks_decimal_dot)
    out = out.mask(mask, out.where(~mask, out.str.replace(".", "", regex=False)))
    return out


def normalize_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        try:
            return series.astype(float)
        except Exception:
            pass
    s = series.astype(object)
    s = s.where(~s.isna(), None)
    s = _normalize_whitespace_and_minus(s).str.strip()
    lower = s.str.lower()
    s = s.mask(lower.isin(_NULL_TOKENS_LOWER), np.nan)
    s = s.fillna("")
    s, percent_mask = _handle_percent(s)
    s, negative_mask = _detect_negatives(s)
    s, kmb_multiplier = _extract_kmb_multiplier(s)
    s = _strip_currency_and_separators(s)
    s = _normalize_decimal_thousands(s)
    nums = pd.to_numeric(s, errors="coerce")
    nums = nums * kmb_multiplier
    nums = nums.mask(negative_mask, -nums)
    nums = nums.mask(percent_mask, nums / 100.0)
    nums = nums.replace([np.inf, -np.inf], np.nan)
    return nums.astype(float)


def is_numeric_candidate(
    series: pd.Series, sample_n: int = 30, threshold: float = 0.7
) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    s_str = _normalize_whitespace_and_minus(s.astype(str)).str.strip()
    s_str = s_str.mask(s_str.str.lower().isin(_NULL_TOKENS_LOWER), np.nan).dropna()
    if s_str.empty:
        return False
    if len(s_str) > sample_n:
        s_str = s_str.sample(sample_n, random_state=42)
    parsed = normalize_numeric_series(s_str)
    return bool(parsed.notna().mean() >= threshold)


def _is_year_like(cell: object) -> bool:
    try:
        s = str(cell).strip()
        if not re.fullmatch(r"\d{4}", s):
            return False
        y = int(s)
        return 1900 <= y <= 2100
    except Exception:
        return False


def _non_empty_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.ne("")


def _row_non_empty_ratio(df: pd.DataFrame, row_idx: int) -> float:
    row = df.iloc[row_idx]
    mask = _non_empty_mask(row)
    return float(mask.mean()) if len(mask) else 0.0


def _row_year_signal(df: pd.DataFrame, row_idx: int):
    row = df.iloc[row_idx]
    years = []
    for val in row.iloc[1:]:
        if _is_year_like(val):
            years.append(int(str(val).strip()))
    inc = all(x < y for x, y in zip(years, years[1:])) if len(years) >= 2 else False
    return len(years), inc


def _type_consistency_score(
    df: pd.DataFrame, header_row: int, lookahead: int = 5
) -> float:
    start = header_row + 1
    end = min(df.shape[0], start + lookahead)
    if start >= end:
        return 0.0
    body = df.iloc[start:end]
    if body.empty:
        return 0.0
    score = 0.0
    col0 = body.iloc[:, 0].astype(str).str.strip()
    has_alpha = col0.str.contains(r"[A-Za-z]", regex=True, na=False)
    not_numeric = normalize_numeric_series(col0).isna()
    score += (has_alpha | not_numeric).mean()
    if df.shape[1] > 1:
        numeric_cols = []
        for j in range(1, df.shape[1]):
            ser = body.iloc[:, j]
            parsed = normalize_numeric_series(ser)
            numeric_cols.append(parsed.notna().mean())
        if numeric_cols:
            score += float(np.mean(numeric_cols))
    return score


def detect_header_row(
    df: pd.DataFrame, top_n: int = 10, lookahead: int = 5, verbose: bool = False
) -> int:
    top_n = min(top_n, df.shape[0])
    best_idx = 0
    best_score = float("-inf")
    for r in range(top_n):
        density = _row_non_empty_ratio(df, r)
        year_count, inc = _row_year_signal(df, r)
        type_score = _type_consistency_score(df, r, lookahead)
        penalty = (1.0 - density) * 0.5
        score = (
            year_count * 3.0
            + (1.5 if inc else 0.0)
            + density
            + type_score * 2.0
            - penalty
        )
        if score > best_score:
            best_score = score
            best_idx = r
    if verbose:
        print(
            f"[info] header detection: chose row {best_idx} with score {best_score:.2f}"
        )
    return best_idx


def _dedupe_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            out.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            out.append(h)
    return out


def build_headers(df: pd.DataFrame, header_row: int, verbose: bool = False):
    row = df.iloc[header_row].tolist()
    headers: List[str] = []
    for i, val in enumerate(row):
        if pd.isna(val):
            headers.append(f"Column_{i+1}")
        else:
            s = re.sub(r"\s+", " ", str(val).strip())
            headers.append(s or f"Column_{i+1}")
    if header_row + 1 < df.shape[0]:
        next_row_raw = df.iloc[header_row + 1]
        # Handle NaN values properly before converting to string
        next_row = next_row_raw.fillna("").astype(str).str.strip()
        # Filter out empty strings and "nan" strings that come from NaN conversion
        non_empty_mask = (next_row != "") & (next_row.str.lower() != "nan")
        non_empty_ratio = non_empty_mask.mean() if len(next_row) else 0.0
        alpha_ratio = next_row.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
        year_count = sum(_is_year_like(x) for x in next_row.iloc[1:])
        if non_empty_ratio > 0.7 and alpha_ratio > 0.5 and year_count == 0:
            flat = []
            for h, n in zip(headers, next_row.tolist()):
                n_clean = re.sub(r"\s+", " ", n) if n else ""
                flat.append(f"{h} | {n_clean}" if n_clean else h)
            headers = flat
            header_row += 1
    headers = _dedupe_headers(headers)
    return headers, header_row + 1


__all__ = [
    "detect_header_row",
    "build_headers",
    "_drop_fully_blank_rows",
    "is_numeric_candidate",
    "normalize_numeric_series",
    "_NULL_TOKENS_LOWER",
    "_normalize_whitespace_and_minus",
]
