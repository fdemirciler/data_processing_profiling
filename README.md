# Unified Data Processing Pipeline

Unified, LLM-friendly ingestion pipeline that loads a raw CSV / Excel file, applies light cleaning, infers types, profiles data quality, and returns both JSON metadata for LLMs and cleaned datasets for analysis scripts.

## What It Does

* **Load**: CSV (no header assumption) or first sheet of Excel
* **Clean**: header detection (or synthesize), blank row removal, numeric normalization (%, K/M/B, currency), null token mapping, first-column heuristic rename to `Metric`
* **Infer**: currency, percentage, date, id, integer/float, categorical, text (confidence-ordered)
* **Profile**: per-column stats + quality metrics (completeness, uniqueness, validity, consistency + overall)
* **Return**: Three outputs for LLM-orchestrator workflows

## Pipeline Outputs

The pipeline returns a dictionary with **three key components**:

1. **`cleaned_df`** *(pandas.DataFrame)* - Full cleaned dataset for analysis scripts
2. **`payload`** *(dict)* - JSON-serializable metadata + samples for LLM consumption  
3. **`cleaning_report`** *(dict)* - Transparency on transformations applied

### Intended Workflow
```
1. Pipeline processes file → {cleaned_df, payload, cleaning_report, ...}
2. LLM receives payload JSON → writes analysis script based on metadata/samples
3. Orchestrator executes script on cleaned_df → captures results
4. LLM interprets results → provides final answer to user
```

### Programmatic vs CLI Usage
- **Programmatic**: Returns all components including cleaned DataFrame
- **CLI**: Outputs only JSON payload (for LLM integration)

## Key Entry Points

**Programmatic API:**
```python
from data_processing import run_processing_pipeline

result = run_processing_pipeline('data.csv', mode='full')
# result['cleaned_df']      → pandas DataFrame for analysis scripts  
# result['payload']         → JSON metadata + samples for LLM
# result['cleaning_report'] → transformation details
```

**Command Line Interface:**
```powershell
python -m data_processing.cli <file> [--mode full|schema_only] [--json] [--output out.json] [--sample-size N] [--treat-first-row-as-data] [--suppress-warnings]
```

## Examples

### CLI Usage
```powershell
python -m data_processing.cli .\BS.csv --mode full --json --output payload.json
```

### Programmatic Usage
```python
from data_processing import run_processing_pipeline

# Process file and get all outputs
result = run_processing_pipeline('BS.csv', mode='full')

# For LLM: JSON metadata + samples
llm_input = result['payload']

# For analysis script: cleaned DataFrame  
cleaned_data = result['cleaned_df']
print(f"Dataset shape: {cleaned_data.shape}")
print(f"Columns: {list(cleaned_data.columns)}")
```

Sample summary output:
```
Rows: 120  Columns: 8
Columns: id, date, revenue, margin, region, segment, users, notes
Mode: full  Version: 1.0.0
  - revenue: type=float null%=0.00 unique%=95.83
  - margin: type=float null%=1.67 unique%=42.50
  - region: type=categorical null%=0.00 unique%=12.50
```

## JSON Payload Structure (full mode)

Actual key order may vary slightly; below reflects current implementation (Python 3.7+ preserves insertion order as built).

```jsonc
{
  "dataset": {
    "rows": 120,
    "columns": 8,
    "column_names": ["id", "date", "revenue", "margin", "region", "segment", "users", "notes"],
    "dtypes": { "id": "int64", "date": "object", "revenue": "float64", "margin": "float64", "users": "float64" }
  },
  "Metric": [
    "Assets", "Cash & Equivalents", "Short-Term Investments", "..." // present only if a first-column rename to Metric occurred
  ],
  "columns": {
    "revenue": {
      "type": "float",
      "null_pct": 0.0,
      "unique_pct": 95.83,
      "top_values": [ { "value": "123.45", "count": 1, "percentage": 0.83 } ],
      "stats": { "min": 12.5, "max": 9983.2, "mean": 182.7, "median": 120.4, "std": 55.3 }
    },
    "date": {
      "type": "date",
      "null_pct": 0.0,
      "unique_pct": 100.0,
      "top_values": [],
      "stats": { "min_date": "2024-01-01", "max_date": "2024-04-05", "date_range_days": 95 }
    }
  },
  "quality": {
    "completeness_score": 99.2,
    "uniqueness_score": 88.5,
    "validity_score": 97.0,
    "consistency_score": 100.0,
    "overall_score": 96.3
  },
  "sample_rows": [ { "id": 1, "date": "2024-01-01", "revenue": 123.45, "margin": 0.32, "users": 1200 } ],
  "cleaning_report": {
    "header_row": 0,
    "renamed_columns": { "Column_1": "Metric" },
    "numeric_columns": ["revenue", "margin", "users"],
    "rows_before": 121,
    "rows_after": 120,
    "null_token_mappings": {},
    "file_kind": "csv"
  },
  "mode": "full",
  "version": "v1"
}
```

### Key Notes
* `Metric` (singular) is an optional top-level list mirroring the cleaned first column values when that column was heuristically renamed to `Metric`. Duplicates & order are preserved exactly as in the cleaned dataframe.
* `columns` -> per-column summary: detected type, null & unique percentages, up to 5 top values, and a minimal stats subset (varies by type).
* `quality` -> dataset-level quality metrics (lightweight heuristics, bounded 0–100).
* `cleaning_report` -> transparency on structural transformations (header row index, renames, numeric columns detected, null token replacements, row counts).
* `version` -> current payload schema version (`v1`).

### schema_only Mode
Returns a trimmed payload:

```jsonc
{
  "dataset": { "rows": 120, "columns": 8, "column_names": ["id", "date", ...] },
  "Metric": ["Assets", "Cash & Equivalents", "..."] // optional as above
  "columns": { "id": { "type": "integer" }, "date": { "type": "date" }, ... },
  "mode": "schema_only",
  "version": "v1"
}
```

Omitted: detailed stats, quality, sample_rows, cleaning_report.

## Configuration Flags

| Flag                        | Description                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| `--mode`                    | `full` (default) or `schema_only`                                          |
| `--sample-size N`           | Max sample rows in `sample_rows` (full mode)                               |
| `--treat-first-row-as-data` | Bypass header detection; generate `Column_1..n`; disables rename heuristic |
| `--suppress-warnings`       | Suppress pandas/date parsing warnings                                      |
| `--json`                    | Print JSON payload to stdout                                               |
| `--output path`             | Write JSON payload to file                                                 |

## Cleaning Logic (Summary)

Implemented in `data_processing/cleaning_utils.py`:
* Header detection: heuristic scoring of candidate rows (alpha density, blanks)
* Renaming heuristic: if first column is text-dominant and others are mostly numeric → rename to `Metric`
* Synthetic headers: if `--treat-first-row-as-data` set
* Numeric normalization: `%` to 0–1 float, currency symbols & commas stripped, `K/M/B` scaled, dash/minus normalization
* Null token mapping: common tokens (e.g. `na`, `n/a`, `null`, `-`, empty) converted to `NaN`

## Type Inference

`TypeInferencer` applies prioritized detectors (currency -> percentage -> date -> id -> numeric -> categorical -> text) with confidence thresholds.

## Profiling

`DataProfiler` generates:
* Per-column: detected type, null %, unique %, most frequent values (value/count/percentage), minimal statistics subset per type.
* Dataset-level quality: completeness (non-null ratio), uniqueness (average distinct proportion), validity (simple rule-of-thumb checks by type), consistency (light heuristic), overall weighted score.

## Testing

Run tests:
```powershell
pytest -q
```

