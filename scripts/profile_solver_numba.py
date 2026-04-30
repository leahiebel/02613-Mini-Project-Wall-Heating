#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import sys
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_DIR = Path("/dtu/projects/02613_2025/data/modified_swiss_dwellings")


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        return func


def read_building_ids(ids_file: Path) -> list[str]:
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing file: {ids_file}")
    building_ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not building_ids:
        raise ValueError(f"Subset file is empty: {ids_file}")
    return building_ids


def load_module(module_path: Path) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Missing solver module: {module_path}")

    spec = spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_data(load_dir: Path, bid: str):
    import numpy as np

    size = 512
    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(load_dir / f"{bid}_domain.npy")
    interior_mask = np.load(load_dir / f"{bid}_interior.npy").astype(bool)
    return u, interior_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a Jacobi solver from any simulate-like module on a fixed building subset "
            "using kernprof."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path containing floorplan data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        required=True,
        help="Path to the file containing the fixed profiling building IDs",
    )
    parser.add_argument(
        "--solver-module",
        type=Path,
        required=True,
        help="Path to the solver module that defines the function to profile",
    )
    parser.add_argument(
        "--solver-function",
        default="jacobi",
        help="Solver function name inside the module (default: jacobi)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20_000,
        help="Maximum Jacobi iterations (default: 20000)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Convergence tolerance (default: 1e-4)",
    )
    return parser.parse_args()


@profile
def run_profile(
    data_dir: Path,
    building_ids: list[str],
    solver_module: Path,
    solver_function: str,
    max_iter: int,
    atol: float,
) -> None:  # noqa: F821
    import time

    print(f"[INFO] Loaded {len(building_ids)} buildings", flush=True)

    module = load_module(solver_module)
    if not hasattr(module, solver_function):
        raise AttributeError(f"{solver_module} has no function named '{solver_function}'")

    solver = getattr(module, solver_function)

    # -------------------------
    # 🔥 WARM-UP (JIT compile)
    # -------------------------
    print("[INFO] Warm-up start", flush=True)
    u0, interior_mask = load_data(data_dir, building_ids[0])
    solver(u0, interior_mask, max_iter=10, atol=atol)
    print("[INFO] Warm-up done", flush=True)

    # -------------------------
    # ⏱ TIMED RUN
    # -------------------------
    print("[INFO] Timed run start", flush=True)

    t_global = time.perf_counter()

    print("DEBUG: entering loop", len(building_ids), flush=True)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(data_dir, bid)

        t0 = time.perf_counter()
        solver(u0, interior_mask, max_iter=max_iter, atol=atol)
        t1 = time.perf_counter()

        print(f"[TIMING] {bid}: {t1 - t0:.4f} s", flush=True)

    t_global_end = time.perf_counter()

    total = t_global_end - t_global
    print(f"[TIMING] TOTAL: {total:.4f} s", flush=True)
    print(f"[TIMING] AVG: {total / len(building_ids):.4f} s", flush=True)

def main() -> None:
    args = parse_args()
    building_ids = read_building_ids(args.ids_file)
    run_profile(
        data_dir=args.data_dir,
        building_ids=building_ids,
        solver_module=args.solver_module,
        solver_function=args.solver_function,
        max_iter=args.max_iter,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()