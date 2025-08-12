import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class DataProfiler:
    """Profiling engine producing per-column stats and quality metrics."""

    def profile_dataframe(
        self, df: pd.DataFrame, type_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive profile for dataframe."""

        profile = {
            "dataset_info": self._get_dataset_info(df),
            "columns": {},
            "quality_metrics": {},
        }

        for column in df.columns:
            if column in type_info:
                profile["columns"][column] = self._profile_column(
                    df[column], type_info[column]
                )

        profile["quality_metrics"] = self._calculate_quality_metrics(df, type_info)

        return profile

    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
        }

    def _profile_column(
        self, series: pd.Series, type_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed profile for single column."""

        profile = {
            "name": series.name,
            "detected_type": type_info.get("detected_type", "unknown"),
            "original_dtype": str(series.dtype),
            "null_count": series.isnull().sum(),
            "null_percentage": series.isnull().sum() / len(series) * 100,
            "unique_count": series.nunique(),
            "unique_percentage": series.nunique() / len(series) * 100,
            "most_frequent_values": self._get_most_frequent_values(series),
            "completeness_score": (len(series) - series.isnull().sum())
            / len(series)
            * 100,
            "uniqueness_score": self._calculate_uniqueness_score(series),
        }

        # Type-specific statistics
        detected_type = type_info.get("detected_type", "unknown")

        if detected_type in ["integer", "float", "currency", "percentage"]:
            profile["statistics"] = self._get_numeric_statistics(series)
        elif detected_type == "date":
            profile["statistics"] = self._get_date_statistics(series)
        elif detected_type == "categorical":
            profile["statistics"] = self._get_categorical_statistics(series)
        elif detected_type == "text":
            profile["statistics"] = self._get_text_statistics(series)
        else:
            profile["statistics"] = {}

        return profile

    def _get_most_frequent_values(
        self, series: pd.Series, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get most frequent values in series."""

        value_counts = series.value_counts()

        return [
            {
                "value": str(value),
                "count": int(count),
                "percentage": float(count / len(series) * 100),
            }
            for value, count in value_counts.head(top_n).items()
        ]

    def _calculate_uniqueness_score(self, series: pd.Series) -> float:
        """Calculate uniqueness score for series."""

        unique_ratio = series.nunique() / len(series)

        # Score based on uniqueness ratio
        if unique_ratio == 1.0:
            return 100.0  # Perfectly unique
        elif unique_ratio >= 0.9:
            return 90.0
        elif unique_ratio >= 0.7:
            return 70.0
        elif unique_ratio >= 0.5:
            return 50.0
        else:
            return unique_ratio * 100  # Scale lower values

    def _get_numeric_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get numeric statistics for series."""

        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No valid numeric values"}

        stats = {
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "mean": float(clean_series.mean()),
            "median": float(clean_series.median()),
            "std": float(clean_series.std()),
        }

        return stats

    def _get_date_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get date statistics for series."""

        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No valid date values"}

        # Ensure datetime dtype; attempt conversion if necessary
        if not pd.api.types.is_datetime64_any_dtype(clean_series):
            converted = pd.to_datetime(clean_series, errors="coerce")
            converted = converted.dropna()
            if len(converted) == 0:
                return {"error": "Values not parseable as dates"}
            clean_series = converted

        min_v = clean_series.min()
        max_v = clean_series.max()
        try:
            range_days = int((max_v - min_v).days)
        except Exception:
            range_days = None

        stats = {
            "min_date": str(min_v),
            "max_date": str(max_v),
        }
        if range_days is not None:
            stats["date_range_days"] = int(range_days)

        return stats

    def _get_categorical_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get categorical statistics for series."""

        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No valid categorical values"}

        value_counts = clean_series.value_counts()

        stats = {
            "categories": int(len(value_counts)),
            "most_common": (
                str(value_counts.index[0]) if len(value_counts) > 0 else None
            ),
        }

        return stats

    def _get_text_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get text statistics for series."""

        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {"error": "No valid text values"}

        # Text analysis
        text_lengths = clean_series.astype(str).str.len()

        stats = {
            "mean_length": float(text_lengths.mean()),
            "max_length": int(text_lengths.max()),
        }

        return stats

    def _calculate_quality_metrics(
        self, df: pd.DataFrame, type_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""

        metrics = {
            "completeness_score": 0.0,
            "uniqueness_score": 0.0,
            "validity_score": 0.0,
            "consistency_score": 0.0,
            "overall_score": 0.0,
        }

        # Completeness score
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        metrics["completeness_score"] = (total_cells - null_cells) / total_cells * 100

        # Uniqueness score
        uniqueness_scores = []
        for column in df.columns:
            if column in type_info:
                uniqueness_scores.append(self._calculate_uniqueness_score(df[column]))

        if uniqueness_scores:
            metrics["uniqueness_score"] = float(np.mean(uniqueness_scores))

        # Validity score (based on type detection confidence)
        validity_scores = []
        for column, type_data in type_info.items():
            if column in df.columns:
                confidence = type_data.get("confidence_score", 0.5)
                validity_scores.append(confidence * 100)

        if validity_scores:
            metrics["validity_score"] = float(np.mean(validity_scores))

        # Consistency score (based on data consistency)
        metrics["consistency_score"] = self._calculate_consistency_score(df)

        # Overall score (weighted average)
        weights = {
            "completeness": 0.3,
            "uniqueness": 0.2,
            "validity": 0.3,
            "consistency": 0.2,
        }

        overall_score = (
            metrics["completeness_score"] * weights["completeness"]
            + metrics["uniqueness_score"] * weights["uniqueness"]
            + metrics["validity_score"] * weights["validity"]
            + metrics["consistency_score"] * weights["consistency"]
        )

        metrics["overall_score"] = max(0, min(100, overall_score))

        return metrics

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""

        consistency_issues = 0
        total_checks = 0

        # Check for consistent data types across columns
        for column in df.columns:
            total_checks += 1

            # Check for mixed data types
            if df[column].dtype == "object":
                types = {type(v) for v in df[column].dropna().tolist()}
                if len(types) > 1:
                    consistency_issues += 1

        consistency_score = max(
            0, 100 - (consistency_issues / max(total_checks, 1) * 100)
        )

        return consistency_score

    # Removed correlation/outlier detection and aggregate quality scoring for simplicity.
