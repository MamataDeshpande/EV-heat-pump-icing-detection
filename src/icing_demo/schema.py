"""Model input schema and simple data validation."""

import numpy as np
import pandas as pd

FEATURES = ("suction_pressure_bar", "high_pressure_bar", "compressor_speed_pct")
LABEL = "icing_label"
GROUP = "recording_id"


def validate_dataframe(df: pd.DataFrame, *, labeled: bool = True) -> pd.DataFrame:
    """Check the expected columns, finite signals, and recording-level labels."""
    required = list(FEATURES) + ([LABEL, GROUP] if labeled else [])
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("No rows were supplied")
    for name in FEATURES:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {name} contains missing or invalid values")
    if labeled:
        classes = set(df[LABEL].unique())
        if not classes.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 (non-iced) or 1 (iced): {classes}")
        if df[GROUP].isna().any():
            raise ValueError("recording_id must be present for every sample")
        if df.groupby(GROUP)[LABEL].nunique().max() > 1:
            raise ValueError("Each synthetic recording must contain one class")
    return df
