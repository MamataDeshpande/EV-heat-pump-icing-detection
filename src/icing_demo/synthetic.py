"""Generate completely synthetic, illustrative refrigeration-cycle time series.

Numbers are NOT vehicle measurements, manufacturer calibrations or validated
refrigerant physics. Only an ML/data-processing teaching example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_recordings(
    *, per_class: int = 20, samples_per_recording: int = 120, seed: int = 42
) -> pd.DataFrame:
    """Create independent labeled recordings with some non-iced restart transients."""
    if per_class < 3 or samples_per_recording < 30:
        raise ValueError("Require at least 3 recordings per class and 30 samples per recording")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for label in (0, 1):
        for j in range(per_class):
            recording_id = f"SYN_{'ICED' if label else 'CLEAR'}_{j:03d}"
            ambient = float(rng.uniform(-8, 5))
            humidity = float(rng.uniform(72, 95))
            base_speed = float(rng.uniform(35, 70))
            low_offset = float(rng.normal(0, 0.30))
            high_offset = float(rng.normal(0, 0.70))
            restart_recording = label == 0 and j % 4 == 0

            for t in range(samples_per_recording):
                speed = base_speed + 4 * np.sin(t / 17) + rng.normal(0, 1.2)
                p_low = 3.6 + 0.035 * ambient + 0.011 * speed + low_offset
                p_high = 8.4 + 0.055 * speed + 0.08 * ambient + high_offset
                if label:
                    p_low -= 1.25
                    p_high -= 1.8
                restart = bool(restart_recording and 20 <= t < 35)
                if restart:
                    p_low -= 1.0
                    p_high -= 1.6
                    speed += 8.0
                p_low = max(0.3, p_low + rng.normal(0, 0.10))
                p_high = max(1.5, p_high + rng.normal(0, 0.27))
                rows.append({
                    "recording_id": recording_id,
                    "time_s": t,
                    "suction_pressure_bar": round(float(p_low), 4),
                    "high_pressure_bar": round(float(p_high), 4),
                    "compressor_speed_pct": round(float(np.clip(speed, 0, 100)), 4),
                    "icing_label": label,
                    "ambient_temp_c_context_only": round(ambient, 2),
                    "humidity_pct_context_only": round(humidity, 1),
                    "restart_transient_context_only": restart,
                })
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic heat-pump recordings")
    parser.add_argument("--out", default="data/synthetic_recordings.csv")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_synthetic_recordings(
        per_class=args.per_class, samples_per_recording=args.samples, seed=args.seed
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    print(f"Saved {len(df)} synthetic samples to {destination}")


if __name__ == "__main__":
    main()
