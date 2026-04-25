#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simulate import jacobi, load_data  # noqa: E402

DEFAULT_DATA_DIR = Path("/dtu/projects/02613_2025/data/modified_swiss_dwellings")


def read_building_ids(data_dir: Path) -> list[str]:
    ids_file = data_dir / "building_ids.txt"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing file: {ids_file}")
    return ids_file.read_text(encoding="utf-8").splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile reference jacobi function from src/simulate.py using kernprof"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path containing floorplan data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--num-buildings",
        type=int,
        default=1,
        help="How many floorplans to profile (default: 1)",
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
def run_profile(data_dir: Path, building_ids: list[str], max_iter: int, atol: float) -> None:  # noqa: F821
    for bid in building_ids:
        u0, interior_mask = load_data(str(data_dir), bid)
        jacobi(u0, interior_mask, max_iter=max_iter, atol=atol)


def main() -> None:
    args = parse_args()
    if args.num_buildings <= 0:
        raise ValueError("--num-buildings must be > 0")

    building_ids = read_building_ids(args.data_dir)[: args.num_buildings]
    run_profile(args.data_dir, building_ids, args.max_iter, args.atol)


if __name__ == "__main__":
    main()
