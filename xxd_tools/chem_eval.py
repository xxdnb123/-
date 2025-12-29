# utils/chem_eval.py
# -*- coding: utf-8 -*-
"""
对扩散生成的配体轨迹做：
1) 帧级别几何/价态快速检查（坐标->Mol）
2) 可选最小化（MMFF 优先，UFF 兜底）并生成规范 SMILES
3) 轨迹级：收集每帧的力场能量与可选 AMBER 能量
4) 画一张时间步能量曲线图（若最终帧无效则不画）

依赖：numpy、rdkit、matplotlib 和标准库。
外部接口保持不变：
- get_idx2Z_from_featurizer
- validate_frame_and_smiles
- trajectory_energy
- plot_energy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import os
import math
import warnings

# ================= 依赖自检（缺啥给出安装提示） =================

_HAVE_NUMPY = True
_HAVE_RDKIT = True
_HAVE_MPL = True
_HAVE_AMBER = True
_AMBER_BACKEND_READY = True

# numpy
try:
    import numpy as np
except Exception:
    _HAVE_NUMPY = False
    np = None  # type: ignore

# rdkit（按功能模块分级）
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem, rdmolops  # type: ignore
except Exception:
    _HAVE_RDKIT = False
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    DataStructs = None  # type: ignore
    rdMolDescriptors = None  # type: ignore
    rdmolops = None  # type: ignore

# --- RDKit 版本兼容：几何->键推断 ---
try:
    from rdkit.Chem import rdDetermineBonds as _rdDB  # type: ignore
except Exception:
    _rdDB = None

# --- RDKit 标准化（可用则启用轻量标准化） ---
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize as _rdStd  # type: ignore
except Exception:
    _rdStd = None

# matplotlib（仅在绘图接口需要）
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互环境安全
    import matplotlib.pyplot as plt
except Exception:
    _HAVE_MPL = False
    matplotlib = None  # type: ignore
    plt = None  # type: ignore

# AMBER (OpenMM + OpenFF，可选)
try:
    import openmm  # type: ignore
    from openmm import app as _omm_app  # type: ignore
    from openmm import unit as _omm_unit  # type: ignore
    from openff.toolkit.topology import Molecule as _offMolecule  # type: ignore
    from openmmforcefields.generators import GAFFTemplateGenerator as _GAFFTemplateGenerator  # type: ignore
except Exception:
    _HAVE_AMBER = False
    _AMBER_BACKEND_READY = False
    openmm = None  # type: ignore
    _omm_app = None  # type: ignore
    _omm_unit = None  # type: ignore
    _offMolecule = None  # type: ignore
    _GAFFTemplateGenerator = None  # type: ignore


def _ensure_dependencies(for_plot: bool = False):
    missing = []
    if not _HAVE_NUMPY:
        missing.append("numpy")
    if not _HAVE_RDKIT:
        missing.append("rdkit")
    if for_plot and not _HAVE_MPL:
        missing.append("matplotlib")

    if missing:
        msgs = []
        if "numpy" in missing:
            msgs.append("numpy 未安装：可执行\n  pip install -U numpy")
        if "rdkit" in missing:
            msgs.append(
                "RDKit 未安装：优先使用 conda 安装（推荐）\n"
                "  conda install -c conda-forge rdkit\n"
                "或尝试（部分平台可用）\n"
                "  pip install rdkit-pypi"
            )
        if "matplotlib" in missing:
            msgs.append("matplotlib 未安装：可执行\n  pip install -U matplotlib")
        raise RuntimeError("依赖缺失，功能无法运行：\n- " + "\n- ".join(msgs))


# ================= 性能控制（不改外部接口，用环境变量开关） =================
# CHEM_EVAL_SPEED: fast | balanced | strict
_SPEED_MODE = os.environ.get("CHEM_EVAL_SPEED", "balanced").strip().lower()
# 在 balanced 模式下，每隔多少帧做一次完整最小化检查（最后一帧一定做）
_BALANCED_MIN_INTERVAL = int(os.environ.get("CHEM_EVAL_MIN_EVERY", "10"))
# fast 模式下是否对非末帧做轻量 sanitize（更稳但略慢）
_FAST_LIGHT_SANITIZE = os.environ.get("CHEM_EVAL_FAST_SAN", "1") not in ("0", "false", "False")

def _is_fast():
    return _SPEED_MODE == "fast"

def _is_strict():
    return _SPEED_MODE == "strict"

def _is_balanced():
    return _SPEED_MODE == "balanced"


# ================= 化学评估主逻辑 =================

# ========= 常量：共价半径、最大价、默认元素序 =========

COV_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}
MAX_VALENCE = {1:1, 6:4, 7:4, 8:2, 9:1, 15:5, 16:6, 17:1, 35:1, 53:1}
COMMON_Z_ORDER = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C,N,O,F,P,S,Cl,Br,I

# ========= 合理性阈值 =========
ENERGY_PER_HA_WARN = 12.0
ENERGY_PER_HA_FAIL = 20.0
BOND_LEN_RANGE = {
    (6, 6): (1.20, 1.70),
    (6, 7): (1.20, 1.60),
    (6, 8): (1.20, 1.60),
    (6, 9): (1.20, 1.55),
    (6, 16): (1.60, 1.95),
    (6, 17): (1.60, 2.00),
    (6, 35): (1.70, 2.10),
    (6, 53): (1.85, 2.25),
    (7, 7): (1.20, 1.60),
    (7, 8): (1.20, 1.55),
    (8, 8): (1.20, 1.60),
}
MAX_FRAGMENTS_ALLOWED = 1
MIN_RING_SIZE_WARN = 3
MAX_SMALL_RINGS_WARN = 3
ALLOW_UNASSIGNED_CHIRAL = 2

# ========= 运行时上下文（不改签名，通过模块变量传递“是否末帧”等） =========
_CTX_IS_FINAL_FRAME = False
_CTX_MINIMIZE_THIS_FRAME = True
_CTX_LIGHT_PATH = False  # 非末帧 + fast/balanced 的轻量路径

# ========= 数据结构 =========

@dataclass
class FrameCheckResult:
    ok: bool
    reason: str
    n_atoms: int
    n_bonds: int
    smiles: str = ""
    energy: Optional[float] = None

# ========= 工具：类别索引 -> 原子序数 =========

def get_idx2Z_from_featurizer(ligand_featurizer, num_classes: int) -> List[int]:
    """
    尝试从 featurizer 拿“类别->原子序数(Z)”映射；
    拿不到则退化到 COMMON_Z_ORDER（超出部分按碳处理）。
    """
    try_names = [
        "atomic_numbers", "allowed_z", "allowed_atomic_numbers",
        "allowed_atoms", "atom_types"
    ]
    idx2Z = None
    for name in try_names:
        if hasattr(ligand_featurizer, name):
            val = getattr(ligand_featurizer, name)
            if isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
                idx2Z = list(val)
                break
            if isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
                sym2z = {"H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17,"Br":35,"I":53}
                try:
                    idx2Z = [sym2z[s] for s in val]
                    break
                except KeyError:
                    pass

    if idx2Z is None:
        if num_classes <= len(COMMON_Z_ORDER):
            idx2Z = COMMON_Z_ORDER[:num_classes]
        else:
            idx2Z = COMMON_Z_ORDER + [6] * (num_classes - len(COMMON_Z_ORDER))
    return idx2Z

# ========= 几何建模（坐标 -> Mol / SMILES）=========

def _bond_thresh(zi: int, zj: int, k: float) -> float:
    ri = COV_RADII.get(int(zi), 0.77)
    rj = COV_RADII.get(int(zj), 0.77)
    return k * (ri + rj)

def _vectorized_overlap_check(pos, tol: float = 0.6) -> Optional[str]:
    """向量化原子重叠检查。"""
    if np is None:
        return None
    n = pos.shape[0]
    if n < 2:
        return None
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    if iu[0].size == 0:
        return None
    d2u = d2[iu]
    if np.any(d2u < tol * tol):
        mask = (d2 < tol * tol)
        mask[np.eye(n, dtype=bool)] = False
        ij = np.argwhere(mask)
        i, j = (int(ij[0,0]), int(ij[0,1])) if ij.size else (0,1)
        return f"atom overlap ({i},{j}) d={float(np.sqrt(d2[i,j])):.2f}Å"
    return None

def _build_bonds_distance(Z: List[int], pos, bond_k: float) -> List[Tuple[int,int]]:
    """元素依赖阈值 + 全局缩放 k 的向量化连边，返回单键边列表。"""
    if np is None:
        return []
    n = len(Z)
    if n < 2:
        return []
    pos = np.asarray(pos, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))  # [n,n]
    rad = np.vectorize(lambda z: COV_RADII.get(int(z), 0.77))(np.array(Z))
    thr = bond_k * (rad[:, None] + rad[None, :])  # [n,n]
    iu = np.triu_indices(n, k=1)
    sel = d[iu] <= thr[iu]
    ii = iu[0][sel]
    jj = iu[1][sel]
    return list(zip(ii.tolist(), jj.tolist()))

def _coords_to_mol(Z: List[int], pos, bond_k: float = 1.20) -> Tuple[Optional["Chem.Mol"], str]:
    """
    若可用，优先 DetermineBonds；但在 fast 非末帧/ balanced 非采样帧走“轻量路径”：
      - 跳过 DetermineBonds，直接距离阈值连边（更快）
    失败回退时，重建“无键分子”再连边，避免重复加边。
    """
    _ensure_dependencies()
    assert Chem is not None

    n = len(Z)
    if n == 0:
        return None, "no atoms"

    # 更严的过密筛：fast 模式上调到 0.8 Å，早点淘汰问题帧
    ov_tol = 0.8 if _CTX_LIGHT_PATH else 0.6
    ov = _vectorized_overlap_check(pos, tol=ov_tol)
    if ov is not None:
        return None, ov

    def _make_atom_only_with_conf(Z_local, pos_local):
        em0 = Chem.EditableMol(Chem.Mol())
        for z_ in Z_local:
            em0.AddAtom(Chem.Atom(int(z_)))
        m0 = em0.GetMol()
        conf0 = Chem.Conformer(m0.GetNumAtoms())
        for ii in range(len(Z_local)):
            x, y, z = map(float, pos_local[ii])
            conf0.SetAtomPosition(ii, Chem.rdGeometry.Point3D(x, y, z))
        m0.RemoveAllConformers()
        m0.AddConformer(conf0, assignId=True)
        return m0

    mol = _make_atom_only_with_conf(Z, pos)

    use_determine_bonds = (_rdDB is not None) and (not _CTX_LIGHT_PATH or _is_strict())
    if use_determine_bonds:
        try:
            _rdDB.DetermineBonds(mol, charge=0)
        except Exception:
            # 失败：丢弃已有 mol，重建一个“无键”分子，再走距离阈值
            mol = _make_atom_only_with_conf(Z, pos)
            rwm = Chem.RWMol(mol)
            for i, j in _build_bonds_distance(Z, pos, bond_k):
                if rwm.GetBondBetweenAtoms(i, j) is None:
                    rwm.AddBond(i, j, Chem.BondType.SINGLE)
            mol = rwm.GetMol()
    else:
        # 轻量路径：直接距离阈值连单键
        rwm = Chem.RWMol(mol)
        for i, j in _build_bonds_distance(Z, pos, bond_k):
            if rwm.GetBondBetweenAtoms(i, j) is None:
                rwm.AddBond(i, j, Chem.BondType.SINGLE)
        mol = rwm.GetMol()

    # 快速度数检查
    for idx, znum in enumerate(Z):
        deg = mol.GetAtomWithIdx(idx).GetDegree()
        maxv = MAX_VALENCE.get(int(znum), 4)
        if deg > maxv:
            return None, f"valence exceeded at atom {idx} (Z={znum}, deg={deg}, max={maxv})"

    # 轻量路径：仅做属性 sanitize，严格/末帧做完整 sanitize
    if _CTX_LIGHT_PATH and _FAST_LIGHT_SANITIZE:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol, "ok"
        except Exception:
            pass  # 退到完整 sanitize 再试一次

    # 严格路径或轻量失败：完整 sanitize
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"sanitize failed: {str(e)}"
    return mol, "ok"

def _standardize_mol(m: "Chem.Mol") -> "Chem.Mol":
    """轻量标准化：移除小碎片、规范化、再离子化（若可用）。"""
    if _rdStd is None or m is None:
        return m
    try:
        chooser = _rdStd.LargestFragmentChooser(preferOrganic=True)
        m = chooser.choose(m)
        normalizer = _rdStd.Normalizer()
        reionizer = _rdStd.Reionizer()
        m = normalizer.normalize(m)
        m = reionizer.reionize(m)
        te = _rdStd.TautomerEnumerator()
        m = te.Canonicalize(m)
        return m
    except Exception:
        return m

def _minimize_and_smiles(mol: "Chem.Mol", pos, maxIters: int = 200, light: bool = False) -> Tuple[Optional["Chem.Mol"], Optional[float], str, str]:
    """
    严格/末帧：加氢 + MMFF 优先（UFF 兜底）+ 标准化
    轻量：不加氢，直接 UFF on heavy（快很多），失败就退回无最小化 SMILES
    """
    _ensure_dependencies()
    assert Chem is not None and AllChem is not None

    try:
        # 写入构象
        n_in = pos.shape[0]
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(n_in):
            x, y, z = map(float, pos[i])
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

        if light:
            # 轻量：不加氢，直接 UFF
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                ff.Initialize()
                n_steps = ff.Minimize(maxIts=maxIters // 2)
                energy = float(ff.CalcEnergy())
                mol_noH = Chem.Mol(mol)  # 已经是无氢
                # 轻量不做标准化，直接 SMILES
                try:
                    Chem.SanitizeMol(mol_noH, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    Chem.SanitizeMol(mol_noH)
                smi = Chem.MolToSmiles(mol_noH, canonical=True)
                return mol, energy, f"min_steps={n_steps}", smi
            except Exception:
                # 回退：无最小化，直接出 SMILES（可能失败）
                try:
                    mol_noH = Chem.Mol(mol)
                    Chem.SanitizeMol(mol_noH)
                    smi = Chem.MolToSmiles(mol_noH, canonical=True)
                    return mol, None, "light_no_min", smi
                except Exception as e2:
                    return None, None, f"light path error: {str(e2)}", ""

        # 严格：加氢 + MMFF / UFF
        molH = Chem.AddHs(mol, addCoords=True)
        ff = None
        try:
            props = AllChem.MMFFGetMoleculeProperties(molH, mmffVariant='MMFF94s')
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(molH, props)
        except Exception:
            ff = None

        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(molH)

        ff.Initialize()
        n_steps = ff.Minimize(maxIts=maxIters)
        energy = float(ff.CalcEnergy())

        mol_noH = Chem.RemoveHs(molH)
        Chem.SanitizeMol(mol_noH)
        mol_std = _standardize_mol(mol_noH)
        smi = Chem.MolToSmiles(mol_std, canonical=True)
        return molH, energy, f"min_steps={n_steps}", smi

    except Exception as e:
        return None, None, f"minimize error: {str(e)}", ""

def _quick_energy_estimate(mol: "Chem.Mol") -> Optional[float]:
    """不给最小化也估算一个 UFF 能量（仅用于 fallback）。"""
    _ensure_dependencies()
    if Chem is None or AllChem is None:
        return None
    try:
        m = Chem.Mol(mol)
        ff = AllChem.UFFGetMoleculeForceField(m)
        if ff is None:
            return None
        ff.Initialize()
        return float(ff.CalcEnergy())
    except Exception:
        return None

# ========= 最小化后的后验检查 =========

def _post_min_checks(molH: "Chem.Mol", energy: Optional[float]) -> Tuple[bool, str]:
    """
    - 组件数、手性定义数、环分布
    - 键长范围（常见元素对）
    - 能量密度（按重原子计）
    返回 (ok, reason)
    """
    _ensure_dependencies()
    assert Chem is not None

    if molH is None:
        return False, "no molecule after minimization"

    try:
        mol_noH = Chem.RemoveHs(molH)
    except Exception:
        mol_noH = Chem.Mol(molH)

    n_atoms = mol_noH.GetNumAtoms()
    if n_atoms == 0:
        return False, "no atoms after RemoveHs"

    # 组件数
    frags = Chem.GetMolFrags(mol_noH, asMols=False, sanitizeFrags=False)
    n_frags = len(frags)
    if n_frags > MAX_FRAGMENTS_ALLOWED:
        return False, f"too many fragments ({n_frags})"

    # 手性
    try:
        rdmolops.AssignAtomChiralTagsFromStructure(mol_noH)
    except Exception:
        pass
    undef_chiral = 0
    for a in mol_noH.GetAtoms():
        if a.HasProp('_ChiralityPossible') and a.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            undef_chiral += 1
    if undef_chiral > ALLOW_UNASSIGNED_CHIRAL:
        return False, f"too many undefined stereocenters ({undef_chiral})"

    # 小环
    ri = mol_noH.GetRingInfo()
    small_rings = 0
    for atom_ring in ri.AtomRings():
        sz = len(atom_ring)
        if sz <= MIN_RING_SIZE_WARN:
            small_rings += 1
    if small_rings > MAX_SMALL_RINGS_WARN:
        return False, f"too many small rings ({small_rings})"

    # 键长
    if molH.GetNumConformers() > 0:
        conf = molH.GetConformer()
        ref_mol = mol_noH
        for b in ref_mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            ai = ref_mol.GetAtomWithIdx(i)
            aj = ref_mol.GetAtomWithIdx(j)
            zi = int(ai.GetAtomicNum()); zj = int(aj.GetAtomicNum())
            key = (min(zi, zj), max(zi, zj))
            if key in BOND_LEN_RANGE:
                pi = conf.GetAtomPosition(i); pj = conf.GetAtomPosition(j)
                d = math.dist((pi.x, pi.y, pi.z), (pj.x, pj.y, pj.z))
                lo, hi = BOND_LEN_RANGE[key]
                if not (lo <= d <= hi):
                    return False, f"abnormal bond length {zi}-{zj} d={d:.2f}Å not in [{lo:.2f},{hi:.2f}]"

    # 能量密度
    if energy is not None:
        heavy = sum(1 for a in mol_noH.GetAtoms() if a.GetAtomicNum() > 1)
        heavy = max(heavy, 1)
        e_per = float(energy) / heavy
        if e_per >= ENERGY_PER_HA_FAIL:
            return False, f"energy/HA too high ({e_per:.1f})"
    return True, "ok"

# ========= AMBER 能量（可选） =========

_AMBER_WARNING_EMITTED = False

def _amber_backend_available() -> bool:
    return bool(_HAVE_AMBER and openmm is not None and _offMolecule is not None and _GAFFTemplateGenerator is not None)

def _compute_amber_energy(mol: "Chem.Mol") -> Optional[float]:
    """尝试用 OpenMM + GAFF 评估 AMBER 势能；失败返回 None。"""
    if not _amber_backend_available():
        return None
    try:
        off_mol = _offMolecule.from_rdkit(mol, allow_undefined_stereo=True)  # type: ignore
        if not off_mol.conformers:
            return None
        topology = off_mol.to_topology().to_openmm()
        positions = off_mol.conformers[0].to_openmm()
        ff = _omm_app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")  # type: ignore
        gaff = _GAFFTemplateGenerator(molecules=[off_mol])  # type: ignore
        ff.registerTemplateGenerator(gaff.generator)
        system = ff.createSystem(topology, nonbondedMethod=_omm_app.NoCutoff)  # type: ignore
        integrator = openmm.LangevinIntegrator(300 * _omm_unit.kelvin, 1.0 / _omm_unit.picosecond, 0.002 * _omm_unit.picoseconds)  # type: ignore
        context = openmm.Context(system, integrator)  # type: ignore
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(_omm_unit.kilocalorie_per_mole)  # type: ignore
        del context, integrator, system
        return float(energy)
    except Exception:
        return None

def _amber_energy_with_fallback(mol: "Chem.Mol", fallback_energy: Optional[float]) -> Optional[float]:
    """优先 AMBER；不可用则仅返回 fallback，且只警告一次。"""
    global _AMBER_WARNING_EMITTED
    val = _compute_amber_energy(mol)
    if val is not None:
        return val
    if fallback_energy is not None and not _AMBER_WARNING_EMITTED:
        warnings.warn("AMBER backend unavailable; falling back to existing力场能量", RuntimeWarning)
        _AMBER_WARNING_EMITTED = True
    return fallback_energy

# ========= 对外接口（保持不变） =========

def validate_frame_and_smiles(
    Z: List[int], pos,
    do_minimize: bool = True, bond_k: float = 1.20, maxIters: int = 200
) -> Tuple[FrameCheckResult, Optional["Chem.Mol"]]:
    """
    单帧验证：
    - 轻量/严格两套路径自动选择（由运行时上下文决定）
    - 可选最小化
    - 后验合理性检查（严格路径）
    - 产出 SMILES（若失败则空）
    """
    _ensure_dependencies()
    mol, msg = _coords_to_mol(Z, pos, bond_k=bond_k)
    if mol is None:
        return FrameCheckResult(False, msg, len(Z), 0, ""), None

    energy = None
    smiles = ""
    mol_out = mol

    # 是否对该帧做最小化
    minimize_now = do_minimize and _CTX_MINIMIZE_THIS_FRAME

    if minimize_now:
        # 轻量路径 -> 轻量最小化；严格路径 -> 严格最小化
        mol2, energy, mmsg, smiles = _minimize_and_smiles(
            mol, pos, maxIters=maxIters, light=_CTX_LIGHT_PATH
        )
        if mol2 is None:
            return FrameCheckResult(False, f"minimize failed: {mmsg}", len(Z), mol.GetNumBonds(), ""), mol
        if not _CTX_LIGHT_PATH:  # 严格路径才做后验检查
            ok2, why2 = _post_min_checks(mol2, energy)
            if not ok2:
                return FrameCheckResult(False, f"post_min_checks failed: {why2}", len(Z), mol2.GetNumBonds(), ""), mol2
        mol_out = mol2
    else:
        # 无最小化：尽量快速给 SMILES；轻量路径不做标准化
        try:
            m2 = Chem.RemoveHs(mol) if mol.HasSubstructMatch(Chem.MolFromSmarts("[#1]")) else mol  # 快速去氢（若有）
        except Exception:
            m2 = mol
        try:
            if _CTX_LIGHT_PATH:
                Chem.SanitizeMol(m2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                smiles = Chem.MolToSmiles(m2, canonical=True)
            else:
                Chem.SanitizeMol(m2)
                smiles = Chem.MolToSmiles(_standardize_mol(m2), canonical=True)
        except Exception:
            smiles = ""

    return FrameCheckResult(True, "ok", len(Z), mol_out.GetNumBonds(), smiles, energy), mol_out

# ========= 轨迹级：能量序列 =========

def trajectory_energy(
    pos_traj,   # np.ndarray [T, N, 3]
    v_traj,     # np.ndarray [T, N] (int)
    idx2Z: List[int],
    do_minimize: bool = True,
    bond_k: float = 1.20,
    maxIters: int = 200
) -> Dict:
    """
    对一个样本的完整轨迹：
    - 每帧：验证/最小化并拿到力场能量（可选 AMBER）
    - 返回：能量序列 + 帧有效性 + 失败原因
    """
    _ensure_dependencies()
    assert np is not None

    T = pos_traj.shape[0]
    assert v_traj.shape[0] == T
    valid_flags: List[bool] = []
    reasons: List[str] = []
    energies: List[Optional[float]] = []
    amber_energies: List[Optional[float]] = []

    for t in range(T):
        # 设置运行时上下文（不改对外接口）
        global _CTX_IS_FINAL_FRAME, _CTX_MINIMIZE_THIS_FRAME, _CTX_LIGHT_PATH
        _CTX_IS_FINAL_FRAME = (t == T - 1)

        if _is_strict():
            _CTX_LIGHT_PATH = False
            _CTX_MINIMIZE_THIS_FRAME = do_minimize
        elif _is_fast():
            _CTX_LIGHT_PATH = not _CTX_IS_FINAL_FRAME
            _CTX_MINIMIZE_THIS_FRAME = do_minimize and _CTX_IS_FINAL_FRAME  # 只有最后一帧最小化
        else:  # balanced
            _CTX_LIGHT_PATH = not _CTX_IS_FINAL_FRAME
            # 每 N 帧做一次完整最小化，最后一帧必做
            _CTX_MINIMIZE_THIS_FRAME = do_minimize and (_CTX_IS_FINAL_FRAME or (t % max(1, _BALANCED_MIN_INTERVAL) == 0))

        pos = pos_traj[t]
        v = v_traj[t]
        Z = [idx2Z[int(c)] for c in v]

        res, mol_out = validate_frame_and_smiles(
            Z, pos, do_minimize=do_minimize, bond_k=bond_k, maxIters=maxIters
        )
        valid_flags.append(res.ok)
        reasons.append(res.reason)
        energies.append(res.energy if res.ok else None)
        if res.ok and mol_out is not None:
            fallback_energy = res.energy
            if fallback_energy is None:
                fallback_energy = _quick_energy_estimate(mol_out)
            if fallback_energy is None:
                fallback_energy = 0.0
            amber_val = _amber_energy_with_fallback(mol_out, fallback_energy)
        else:
            amber_val = None
        amber_energies.append(amber_val)

    return dict(
        energies=energies,
        amber_energies=amber_energies,
        valid_flags=valid_flags,
        reasons=reasons,
        final_valid=bool(valid_flags[-1])
    )

# ========= 能量曲线图 =========

def plot_energy(
    energies: List[Optional[float]],
    valid_flags: List[bool],
    amber_energies: Optional[List[Optional[float]]] = None,
    out_path: str = "",
    title: str = "",
    dpi: int = 140,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    保存“时间步 vs 能量”的曲线图。
    约束：若最终帧无效 (valid_flags[-1] 为 False)，**不画图**（不创建/覆盖文件，直接返回）。
    """
    _ensure_dependencies(for_plot=True)
    if not valid_flags or not valid_flags[-1]:
        return

    dirn = os.path.dirname(out_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    xs = list(range(len(energies)))
    y_main = [
        float(e) if (e is not None and (i >= len(valid_flags) or valid_flags[i])) else float("nan")
        for i, e in enumerate(energies)
    ]

    plt.figure(figsize=(8, 4.5))  # type: ignore
    plt.plot(xs, y_main, linewidth=2, label='Force field energy')  # type: ignore

    invalid_x = [i for i, ok in enumerate(valid_flags) if not ok or energies[i] is None]
    if invalid_x:
        invalid_y = [0.0 for _ in invalid_x]
        plt.scatter(invalid_x, invalid_y, marker='x', s=40, label='Invalid')  # type: ignore

    amber_vals = amber_energies if amber_energies is not None else [None] * len(energies)
    amber_x = [i for i, (ok, val) in enumerate(zip(valid_flags, amber_vals)) if ok and val is not None]
    amber_y = [float(amber_vals[i]) for i in amber_x]
    if amber_x:
        plt.scatter(amber_x, amber_y, marker='o', s=55, color='tab:orange', label='AMBER energy', zorder=3)  # type: ignore
    if plt.gca().get_legend_handles_labels()[0]:  # type: ignore
        plt.legend(loc='best')  # type: ignore
    plt.xlabel("Timestep (t)")  # type: ignore
    plt.ylabel("Energy (kcal/mol)")  # type: ignore
    plt.grid(True, alpha=0.3)  # type: ignore
    if title:
        plt.title(title)  # type: ignore
    plt.tight_layout()  # type: ignore
    plt.savefig(out_path, dpi=dpi)  # type: ignore
    plt.close()  # type: ignore

    base, _ = os.path.splitext(out_path)
    json_path = base + ".json"
    payload: Dict[str, Any] = dict(
        title=title,
        energies=[float(x) if x is not None else None for x in energies],
        valid_flags=[bool(x) for x in valid_flags],
        amber_energies=[float(x) if x is not None else None for x in amber_vals],
    )
    if metadata:
        payload["metadata"] = metadata
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
