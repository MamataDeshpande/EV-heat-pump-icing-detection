"""Optional MF4 ingestion with user-provided, authorized generic channel names.

No manufacturer signal identifiers or real vehicle data are included.
Install optional support using: pip install '.[mf4]'
"""

from pathlib import Path
from typing import Mapping

import pandas as pd

from .schema import FEATURES


def read_mf4(
    path: str | Path,
    channel_map: Mapping[str, str],
    sample_period_s: float = 1.0,
) -> pd.DataFrame:
    """Load selected MF4 signals and resample onto a common time base."""
    if set(channel_map) != set(FEATURES):
        raise ValueError(f"channel_map must map exactly these generic fields: {FEATURES}")
    if sample_period_s <= 0:
        raise ValueError("sample_period_s must be positive")
    try:
        from asammdf import MDF
    except ImportError as exc:
        raise RuntimeError("Install MF4 support using pip install '.[mf4]'") from exc

    with MDF(str(path)) as mdf:
        selected = mdf.filter(list(channel_map.values()))
        aligned = selected.resample(sample_period_s)
        df = aligned.to_dataframe()
    inverse = {source: generic for generic, source in channel_map.items()}
    df = df.rename(columns=inverse)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing requested channels: {missing}")
    result = df.loc[:, list(FEATURES)].reset_index()
    return result.rename(columns={result.columns[0]: "time_s"})
