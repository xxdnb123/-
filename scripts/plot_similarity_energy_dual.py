
 #!/usr/bin/env python
import argparse
import glob
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_dual_axis(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    similarities: List[float] = data.get("similarities", [])
    energies: List[float] = data.get("amber_energies", [])
    valid_flags: List[bool] = data.get("valid_flags", [True] * len(similarities))

    if len(energies) != len(similarities):
        # pad/truncate energies to match length
        energies = (energies + [None] * len(similarities))[:len(similarities)]

    xs = list(range(len(similarities)))
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Similarity to final frame", color="tab:blue")
    ax1.plot(xs, similarities, color="tab:blue", lw=1.8, label="Similarity")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("AMBER Energy (kcal/mol)", color="tab:orange")
    energy_x = []
    energy_y = []
    for i, (ok, val) in enumerate(zip(valid_flags, energies)):
        if ok and val is not None:
            energy_x.append(i)
            energy_y.append(val)
    if energy_x:
        ax2.plot(energy_x, energy_y, color="tab:orange", lw=1.5, label="Energy")
        ax2.scatter(energy_x, energy_y, color="tab:orange", s=18)
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    title = data.get("title") or os.path.basename(json_path)
    plt.title(title)
    fig.tight_layout()

    out_png = os.path.splitext(json_path)[0] + "_dual.png"
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[Plot] {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Plot similarity + energy dual-axis charts for JSON directory")
    parser.add_argument("json_dir", help="Directory containing *.json files (each holds similarity and energy info)")
    args = parser.parse_args()

    json_dir = os.path.abspath(args.json_dir)
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"Directory not found: {json_dir}")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        raise RuntimeError(f"No JSON files found under {json_dir}")

    for jp in json_files:
        try:
            plot_dual_axis(jp)
        except Exception as e:
            print(f"[WARN] Failed to plot {jp}: {e}")

if __name__ == "__main__":
    main()
