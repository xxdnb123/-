#!/usr/bin/env python
import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from xxd_tools import chem_eval  # type: ignore
from xxd_tools.chem_eval import validate_frame_and_smiles  # type: ignore


def to_numpy(arr_list):
    arr = np.asarray(arr_list)
    if arr.dtype != np.float64 and arr.dtype != np.int64:
        arr = arr.astype(np.float64 if arr.ndim == 3 else np.int64)
    return arr


def export_mols(result_path: str, out_dir: str, sample_ids: Optional[List[int]], bond_k: float, include_invalid: bool, do_minimize: bool):
    pkg = torch.load(result_path, map_location="cpu")
    idx2Z = pkg.get("idx2Z")
    if idx2Z is None:
        raise RuntimeError("idx2Z not found in result file; please regenerate sampling outputs with the latest script.")

    pos_traj_all = pkg["pred_ligand_pos_traj"]
    v_traj_all = pkg["pred_ligand_v_traj"]

    total_samples = len(pos_traj_all)
    samples = sample_ids if sample_ids is not None else list(range(total_samples))

    os.makedirs(out_dir, exist_ok=True)

    exported = 0
    for sid in samples:
        if sid < 0 or sid >= total_samples:
            print(f"[WARN] sample {sid} out of range (0-{total_samples-1}), skip.")
            continue
        pos_traj = to_numpy(pos_traj_all[sid])
        v_traj = np.asarray(v_traj_all[sid], dtype=np.int64)

        sample_dir = os.path.join(out_dir, f"sample_{sid:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        T = pos_traj.shape[0]
        for t in range(T):
            pos = pos_traj[t]
            cat = v_traj[t]
            Z = [idx2Z[int(c)] for c in cat]

            chem_eval._CTX_IS_FINAL_FRAME = (t == T - 1)
            if chem_eval._is_strict():
                chem_eval._CTX_LIGHT_PATH = False
                chem_eval._CTX_MINIMIZE_THIS_FRAME = do_minimize
            elif chem_eval._is_fast():
                chem_eval._CTX_LIGHT_PATH = not chem_eval._CTX_IS_FINAL_FRAME
                chem_eval._CTX_MINIMIZE_THIS_FRAME = do_minimize and chem_eval._CTX_IS_FINAL_FRAME
            else:
                chem_eval._CTX_LIGHT_PATH = not chem_eval._CTX_IS_FINAL_FRAME
                interval = max(1, chem_eval._BALANCED_MIN_INTERVAL)
                chem_eval._CTX_MINIMIZE_THIS_FRAME = do_minimize and (chem_eval._CTX_IS_FINAL_FRAME or (t % interval == 0))

            res, mol = validate_frame_and_smiles(Z, pos, do_minimize=do_minimize, bond_k=bond_k, maxIters=200)
            if mol is None:
                if include_invalid:
                    print(f"[WARN] sample {sid} frame {t} invalid (reason: {res.reason}), skipped (no mol).")
                continue
            if not res.ok and not include_invalid:
                continue

            mol_block = Chem.MolToMolBlock(mol)
            outfile = os.path.join(sample_dir, f"frame_{t:04d}.mol")
            with open(outfile, "w", encoding="utf-8") as fout:
                fout.write(mol_block)
            exported += 1
    print(f"Exported {exported} mol files to {out_dir}")


def parse_sample_ids(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return vals


def main():
    parser = argparse.ArgumentParser(description="Export trajectory frames to .mol files from result_<id>.pt")
    parser.add_argument("result_file", help="Path to result_<data_id>.pt produced by sample_diffusion")
    parser.add_argument("output_dir", help="Directory to store exported mol files")
    parser.add_argument("--samples", help="Sample indices to export, e.g. '0,2,5-7'; default: all")
    parser.add_argument("--bond_k", type=float, default=1.25, help="Bond threshold scaling factor")
    parser.add_argument("--include-invalid", action="store_true", help="Also dump frames even if validation fails (if Mol exists)")
    parser.add_argument("--no-minimize", action="store_true", help="Disable minimization (default: follow strict/balanced logic)")
    args = parser.parse_args()

    sample_ids = parse_sample_ids(args.samples)
    export_mols(
        args.result_file,
        args.output_dir,
        sample_ids,
        args.bond_k,
        include_invalid=args.include_invalid,
        do_minimize=not args.no_minimize,
    )


if __name__ == "__main__":
    main()
