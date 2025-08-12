from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .type_inference import TypeInferencer
from .data_profiler import DataProfiler

# Reuse cleaning logic
from .cleaning_utils import (
    detect_header_row,
    build_headers,
    _drop_fully_blank_rows,
    is_numeric_candidate,
    normalize_numeric_series,
    _NULL_TOKENS_LOWER,
    _normalize_whitespace_and_minus,
)

# ---------------------------------------------------------------------------
# Internal helpers for structured cleaning report (augment existing cleaners)
# ---------------------------------------------------------------------------


def _load_raw(file_path: str) -> Tuple[pd.DataFrame, str]:
    ext = Path(file_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        # Only first sheet per requirements
        df_raw = pd.read_excel(file_path, sheet_name=0, header=None, dtype=object)
        return df_raw, "excel"
    elif ext == ".csv":
        # Load raw with no header for structural normalization
        df_raw = pd.read_csv(file_path, header=None, dtype=object, engine="python")
        return df_raw, "csv"
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _clean_structured(
    df_raw: pd.DataFrame, file_kind: str, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform structural + cell-level cleaning and build a report.

    We replicate core logic from existing cleaners to capture a cleaning report.
    """
    report: Dict[str, Any] = {
        "header_row": None,
        "flattened_headers": False,
        "renamed_columns": {},
        "numeric_columns": [],
        "rows_before": int(df_raw.shape[0]),
        "rows_after": None,
        "null_token_mappings": {},
        "file_kind": file_kind,
    }

    if df_raw.empty:
        report["rows_after"] = 0
        return pd.DataFrame(), report

    df_work = _drop_fully_blank_rows(df_raw)

    # Optional: treat first row as data (no header row in file)
    if config.get("treat_first_row_as_data"):
        headers = [f"Column_{i+1}" for i in range(df_work.shape[1])]
        body = df_work.reset_index(drop=True)
        report["header_row"] = None
    else:
        # Header detection
        header_row = detect_header_row(df_work, verbose=False)
        report["header_row"] = header_row
        headers, start_row = build_headers(df_work, header_row, verbose=False)
        # build_headers returns start_row = header_row + (1 or 2) already
        body = df_work.iloc[start_row:].reset_index(drop=True)

    if body.empty:
        cleaned = pd.DataFrame(columns=headers)
        report["rows_after"] = 0
        return cleaned, report

    # Remove possible repeated header lines
    # (Using simple heuristic: rows fully matching headers are dropped)
    def _norm_row(row):
        return ["" if pd.isna(v) else str(v).strip() for v in row]

    if not config.get("treat_first_row_as_data"):
        hdr_norm = [str(h).strip() for h in headers]
        keep_mask = []
        for _, r in body.iterrows():
            keep_mask.append(_norm_row(r) != hdr_norm)
        body = body.loc[keep_mask].reset_index(drop=True)

    df = body.copy()
    df.columns = headers

    # First column rename heuristic
    if df.shape[1] > 0:
        first_col = df.columns[0]
        col0_text_ratio = (
            df[first_col]
            .astype(str)
            .str.contains(r"[A-Za-z]", regex=True, na=False)
            .mean()
            if len(df) > 0
            else 0.0
        )
        if df.shape[1] > 1:
            other_numeric = [is_numeric_candidate(df[c]) for c in df.columns[1:]]
            if col0_text_ratio >= 0.5 and sum(other_numeric) >= max(
                1, int(0.6 * (df.shape[1] - 1))
            ):
                df.rename(columns={first_col: "Metric"}, inplace=True)
                report["renamed_columns"][first_col] = "Metric"

    # Identify numeric columns
    numeric_cols = []
    for c in df.columns:
        try:
            if is_numeric_candidate(df[c]):
                numeric_cols.append(c)
        except Exception:
            pass
    report["numeric_columns"] = numeric_cols

    # Normalize numeric columns
    for c in numeric_cols:
        df[c] = normalize_numeric_series(df[c])

    # Text cols + null token mapping counts
    null_token_counts: Dict[str, int] = {}
    for c in df.columns:
        if c not in numeric_cols:
            s = _normalize_whitespace_and_minus(df[c].astype(object)).str.strip()
            lower = s.str.lower()
            mask = lower.isin(_NULL_TOKENS_LOWER)
            null_token_counts[c] = int(mask.sum())
            df[c] = s.mask(mask, np.nan)
    report["null_token_mappings"] = {
        k: v for k, v in null_token_counts.items() if v > 0
    }

    # Final drop of fully blank rows
    df = _drop_fully_blank_rows(df)
    report["rows_after"] = int(df.shape[0])
    return df, report


def _build_llm_payload(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    type_info: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    mode: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_info = profile.get("dataset_info", {})
    columns_profile = profile.get("columns", {})

    # Extract full ordered list of metric labels (first column) if it was renamed to 'Metric'.
    metrics_list = None
    if "Metric" in df.columns:
        raw_metrics = df["Metric"].dropna().astype(str).map(lambda s: s.strip())
        # Keep duplicates / ordering exactly as appears in cleaned data.
        if not raw_metrics.empty:
            metrics_list = raw_metrics.tolist()

    if mode == "schema_only":
        payload: Dict[str, Any] = {
            "dataset": {
                "rows": dataset_info.get("total_rows", df.shape[0]),
                "columns": dataset_info.get("total_columns", df.shape[1]),
                "column_names": list(df.columns),
            },
            # Top-level placeholder for metric list will be inserted just after dataset (insertion order) if available.
            "columns": {
                c: {"type": columns_profile.get(c, {}).get("detected_type", "unknown")}
                for c in df.columns
            },
            "mode": mode,
            "version": "v1",
        }
        if metrics_list:
            # Insert immediately after dataset by reconstructing ordered dict (Python 3.7+ preserves insertion order)
            ordered: Dict[str, Any] = {}
            ordered["dataset"] = payload["dataset"]
            # Use singular 'Metric' to align with column name conveying it's the content of that column
            ordered["Metric"] = metrics_list
            # Remainder
            for k in [k for k in payload.keys() if k != "dataset"]:
                ordered[k] = payload[k]
            return ordered
        return payload

    # Full mode
    sample_size = int(config.get("sample_size", 10))
    sample_rows = (
        df.sample(min(sample_size, len(df)), random_state=None).to_dict(
            orient="records"
        )
        if len(df)
        else []
    )

    # Assemble compact column summaries
    col_summaries = {}
    for col, cprof in columns_profile.items():
        stats = cprof.get("statistics", {}) or {}
        # Minimal numeric stats subset
        minimal_stats_keys = [
            "min",
            "max",
            "mean",
            "median",
            "std",
            "min_date",
            "max_date",
            "date_range_days",
            "categories",
            "most_common",
            "mean_length",
            "max_length",
        ]
        slim_stats = {k: stats[k] for k in minimal_stats_keys if k in stats}
        top_vals = cprof.get("most_frequent_values", [])[:5]
        col_summaries[col] = {
            "type": cprof.get("detected_type"),
            "null_pct": round(float(cprof.get("null_percentage", 0.0)), 4),
            "unique_pct": round(float(cprof.get("unique_percentage", 0.0)), 4),
            "top_values": top_vals,
            "stats": slim_stats,
        }

    payload: Dict[str, Any] = {
        "dataset": {
            "rows": dataset_info.get("total_rows", df.shape[0]),
            "columns": dataset_info.get("total_columns", df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {k: str(v) for k, v in dataset_info.get("dtypes", {}).items()},
        },
        # Metric list will be inserted after dataset if present.
        "columns": col_summaries,
        "quality": profile.get("quality_metrics", {}),
        "sample_rows": sample_rows,
        "cleaning_report": cleaning_report,
        "mode": mode,
        "version": "v1",
    }
    if metrics_list:
        ordered: Dict[str, Any] = {}
        ordered["dataset"] = payload["dataset"]
        ordered["Metric"] = metrics_list
        for k in [k for k in payload.keys() if k != "dataset"]:
            ordered[k] = payload[k]
        payload = ordered
    return payload


def run_processing_pipeline(
    file_path: str, *, mode: str = "full", config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Primary orchestrator: load -> clean -> infer types -> profile -> build payload.

    Parameters
    ----------
    file_path : str
        Path to CSV or Excel file (first sheet only for Excel).
    mode : str
        'full' or 'schema_only'.
    config : dict, optional
        Additional configuration (sample_size, etc.).

    Returns
    -------
    dict with keys: cleaned_df, payload, cleaning_report, profile, type_info
    """
    if mode not in ("full", "schema_only"):
        raise ValueError("mode must be 'full' or 'schema_only'")
    cfg = config or {}

    df_raw, file_kind = _load_raw(file_path)

    cleaned_df, cleaning_report = _clean_structured(df_raw, file_kind, cfg)

    # Type inference after cleaning
    type_inferencer = TypeInferencer()
    type_info = type_inferencer.infer_types(cleaned_df) if not cleaned_df.empty else {}

    # Profile (using existing profiler, but expects type_info)
    profiler = DataProfiler()
    profile = (
        profiler.profile_dataframe(cleaned_df, type_info)
        if not cleaned_df.empty
        else {
            "dataset_info": {
                "total_rows": 0,
                "total_columns": 0,
                "memory_usage_mb": 0.0,
                "dtypes": {},
                "null_counts": {},
                "null_percentages": {},
            },
            "columns": {},
            "quality_metrics": {},
        }
    )

    payload = _build_llm_payload(
        cleaned_df, profile, type_info, cleaning_report, mode, cfg
    )

    return {
        "cleaned_df": cleaned_df,
        "payload": payload,
        "cleaning_report": cleaning_report,
        "profile": profile,
        "type_info": type_info,
    }
