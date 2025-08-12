import pandas as pd
from pathlib import Path
import sys
from pathlib import Path as _P

# Ensure project root (containing the 'data_processing' package directory) is on sys.path
_project_root = _P(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_processing import run_processing_pipeline


def test_pipeline_full_mode(tmp_path: Path):
    # Create a small mixed CSV
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie", "Dana"],
            "Age": [30, 25, 35, 40],
            "Score %": ["95%", "88%", "91%", "85%"],
            "Joined": ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-05"],
        }
    )
    csv_path = tmp_path / "people.csv"
    # Write without header to exercise header detection logic
    df.to_csv(csv_path, index=False, header=False)

    result = run_processing_pipeline(str(csv_path), mode="full")
    payload = result["payload"]

    # Basic structure assertions (header detection may treat first row as header -> 3 data rows)
    assert payload["dataset"]["rows"] in {3, 4}
    assert payload["dataset"]["columns"] >= 4
    assert (
        "Name" in payload["dataset"]["column_names"]
        or "Metric" in payload["dataset"]["column_names"]
    )
    assert len(payload["sample_rows"]) <= 4

    # Column summaries should exist for at least one numeric-ish column
    cols = payload["columns"]
    assert any(
        cinfo.get("type") in {"integer", "float", "percentage", "currency"}
        for cinfo in cols.values()
    )
    # Metric list (singular key) should be present because first column renamed to 'Metric'
    assert (
        "Metric" in payload and len(payload["Metric"]) >= payload["dataset"]["rows"] - 1
    )


def test_pipeline_schema_only(tmp_path: Path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    csv_path = tmp_path / "simple.csv"
    df.to_csv(csv_path, index=False, header=False)

    result = run_processing_pipeline(str(csv_path), mode="schema_only")
    payload = result["payload"]
    assert payload["mode"] == "schema_only"
    assert "columns" in payload
    # schema_only mode intentionally excludes sample_rows
    assert "sample_rows" not in payload


def test_percentage_and_numeric_normalization(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Label": ["Row1", "Row2", "Row3"],
            "Revenue": ["$1,200", "$2,500", "$3,750"],
            "Margin %": ["15%", "20%", "18%"],
            "Users": ["1.2K", "2.5K", "3K"],
        }
    )
    csv_path = tmp_path / "metrics.csv"
    df.to_csv(csv_path, index=False, header=False)

    # Use treat_first_row_as_data so percentage row isn't consumed as header candidate
    result = run_processing_pipeline(
        str(csv_path), mode="full", config={"treat_first_row_as_data": True}
    )
    payload = result["payload"]
    cols = payload["columns"]

    # Revenue column should parse to large numbers (> 1000 for this fixture)
    revenue_like = None
    for name, c in cols.items():
        stats = c.get("stats", {})
        if all(k in stats for k in ("min", "max")):
            try:
                mn, mx = float(stats["min"]), float(stats["max"])
            except Exception:
                continue
            if mx > 3000 and mn >= 1000:  # our revenue values 1200..3750
                revenue_like = c
                break
    assert (
        revenue_like is not None
    ), "Revenue-like numeric column with expected magnitude not found"

    # Users column with K suffix should scale to thousands and have integer-ish max of 3000
    users_like = None
    for name, c in cols.items():
        stats = c.get("stats", {})
        if all(k in stats for k in ("min", "max")):
            try:
                mn, mx = float(stats["min"]), float(stats["max"])
            except Exception:
                continue
            # Users: 1200..3000 (3K becomes 3000)
            if 1000 <= mn <= 1500 and 2500 <= mx <= 3200:
                users_like = c
                break
    assert (
        users_like is not None
    ), "Users-like column (K suffix normalized) not identified"


def test_treat_first_row_as_data(tmp_path: Path):
    # First row should not be consumed as header when flag set
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    csv_path = tmp_path / "numbers.csv"
    df.to_csv(csv_path, index=False, header=False)

    result_default = run_processing_pipeline(str(csv_path), mode="full")
    rows_default = result_default["payload"]["dataset"][
        "rows"
    ]  # header may reduce rows

    result_flag = run_processing_pipeline(
        str(csv_path), mode="full", config={"treat_first_row_as_data": True}
    )
    rows_flag = result_flag["payload"]["dataset"]["rows"]

    # With the flag, we should retain all original rows
    assert rows_flag == 3
    assert rows_flag >= rows_default
    # No 'Metric' column in this case -> top-level Metric list key absent
    assert "Metric" not in result_flag["payload"]
