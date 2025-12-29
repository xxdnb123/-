#!/usr/bin/env python
import argparse
import csv
import os
from typing import List

import numpy as np
import torch

from xxd_tools.chem_eval import trajectory_energy, plot_energy


def _to_numpy_traj(traj_list: List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(traj_list)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr


def _to_numpy_cat(traj_list: List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(traj_list)
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64)
    return arr


def recompute_for_sample(result_path: str, data_id: int, bond_k: float, max_iters: int, do_minimize: bool):
    result_file = os.path.join(result_path, f"result_{data_id}.pt")
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")

    pkg = torch.load(result_file, map_location="cpu")
    idx2Z = pkg.get("idx2Z")
    if idx2Z is None:
        raise ValueError("idx2Z is missing in result file; please regenerate the sampling output with the updated script.")

    pos_traj_list = pkg["pred_ligand_pos_traj"]
    v_traj_list = pkg["pred_ligand_v_traj"]

    plot_dir = os.path.join(result_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    stats = []
    for i_sample, (pos_traj, v_traj) in enumerate(zip(pos_traj_list, v_traj_list)):
        pos_arr = _to_numpy_traj(pos_traj)
        cat_arr = _to_numpy_cat(v_traj)

        energy_pack = trajectory_energy(
            pos_traj=pos_arr,
            v_traj=cat_arr,
            idx2Z=idx2Z,
            do_minimize=do_minimize,
            bond_k=bond_k,
            maxIters=max_iters,
        )

        fig_name = f"data{data_id}_sample{i_sample}.png"
        fig_path = os.path.join(plot_dir, fig_name)
        title = f"Data {data_id} • Sample {i_sample} • Final valid={energy_pack['final_valid']}"
        plot_energy(
            energy_pack["energies"],
            energy_pack["valid_flags"],
            energy_pack.get("amber_energies"),
            fig_path,
            title=title,
            metadata=dict(
                data_id=data_id,
                sample=i_sample,
                final_valid=energy_pack["final_valid"],
            ),
        )

        valid_ratio = sum(energy_pack["valid_flags"]) / len(energy_pack["valid_flags"])
        final_energy = energy_pack["energies"][-1] if energy_pack["energies"] else None
        amber_final = energy_pack.get("amber_energies", [None])[-1] if energy_pack.get("amber_energies") else None
        stats.append(dict(sample=i_sample, valid_ratio=valid_ratio, final_energy=final_energy, amber_final=amber_final))

    csv_path = os.path.join(result_path, f"energy_overview_data{data_id}.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["data_id", "sample", "valid_ratio", "final_energy", "amber_final"])
        for s in stats:
            writer.writerow([
                data_id,
                s["sample"],
                f"{s['valid_ratio']:.4f}",
                "" if s["final_energy"] is None else f"{s['final_energy']:.6f}",
                "" if s["amber_final"] is None else f"{s['amber_final']:.6f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="Recompute chem-eval with AMBER energy support")
    parser.add_argument("--result_path", required=True, help="Directory that contains result_<data_id>.pt")
    parser.add_argument("--data_id", type=int, required=True, help="Data index that was sampled")
    parser.add_argument("--bond_k", type=float, default=1.25)
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--disable_minimize", action="store_true", help="Skip minimization during validation")
    args = parser.parse_args()

    recompute_for_sample(
        result_path=args.result_path,
        data_id=args.data_id,
        bond_k=args.bond_k,
        max_iters=args.max_iters,
        do_minimize=not args.disable_minimize,
    )


if __name__ == "__main__":
    main()
