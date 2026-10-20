"""Plot synthetic recording data and icing predictions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction(csv_file: Path, dest: Path, recording: str | None = None) -> str:
    df = pd.read_csv(csv_file)
    if "recording_id" not in df or "predicted_icing" not in df:
        raise ValueError("Need recording_id and predicted_icing columns")
    recording = recording or str(df["recording_id"].iloc[0])
    sub = df.loc[df["recording_id"] == recording].sort_values("time_s")
    if sub.empty:
        raise ValueError(f"No recording {recording} found")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(sub["time_s"], sub["suction_pressure_bar"], label="Suction pressure")
    axes[0].plot(sub["time_s"], sub["high_pressure_bar"], label="High pressure")
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].legend()
    axes[1].step(sub["time_s"], sub["predicted_icing"], where="post", label="Predicted class")
    if "icing_label" in sub.columns:
        axes[1].step(sub["time_s"], sub["icing_label"], where="post",
                     label="Synthetic truth", linestyle="--")
    axes[1].set_xlabel("Synthetic elapsed time (s)")
    axes[1].set_ylabel("Icing class (0 or 1)")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    fig.suptitle(f"Icing demo — {recording} (synthetic measurements)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return recording


def main():
    parser = argparse.ArgumentParser(description="Plot synthetic pressure and icing predictions")
    parser.add_argument("--data", default="outputs/model/held_out_predictions.csv")
    parser.add_argument("--recording", default=None)
    parser.add_argument("--out", default="outputs/example_prediction.png")
    args = parser.parse_args()
    selected = plot_prediction(Path(args.data), Path(args.out), args.recording)
    print(f"Plotted recording {selected}: {args.out}")


if __name__ == "__main__":
    main()
