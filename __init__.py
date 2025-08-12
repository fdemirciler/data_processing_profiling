"""Data processing package providing cleaning, type inference, profiling and an orchestrated pipeline.

Public entry point:
    run_processing_pipeline(file_path: str, *, mode: str = "full", config: Optional[dict] = None)

Modes:
    full         -> full metadata + column summaries + sample rows
    schema_only  -> only dataset + column type schema (lightweight for LLM)
"""

from typing import Optional, Dict, Any
from .pipeline import run_processing_pipeline  # noqa: F401

__all__ = ["run_processing_pipeline"]
