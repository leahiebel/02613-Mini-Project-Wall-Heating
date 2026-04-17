#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_DIR = Path("/dtu/projects/02613_2025/data/modified_swiss_dwellings")


def read_building_ids(data_dir: Path) -> list[str]:
    ids_file = data_dir / "building_ids.txt"
    if not ids_file.exists():
        raise FileNotFoundError(f"Missing file: {ids_file}")
    return ids_file.read_text(encoding="utf-8").splitlines()


def resolve_ids(all_ids: list[str], ids: list[str] | None, num: int) -> list[str]:
    if ids:
        missing = [bid for bid in ids if bid not in all_ids]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Unknown building id(s): {missing_str}")
        return ids
    return all_ids[:num]


def load_data(data_dir: Path, building_id: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    domain_path = data_dir / f"{building_id}_domain.npy"
    interior_path = data_dir / f"{building_id}_interior.npy"

    if not domain_path.exists() or not interior_path.exists():
        raise FileNotFoundError(
            f"Missing data for {building_id}: {domain_path.name} and/or {interior_path.name}"
        )

    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    domain = np.load(domain_path)
    interior_mask = np.load(interior_path).astype(bool)
    u[1:-1, 1:-1] = domain
    return u, interior_mask


def jacobi(u: np.ndarray, interior_mask: np.ndarray, max_iter: int, atol: float = 1e-6) -> np.ndarray:
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1]
        )
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u: np.ndarray, interior_mask: np.ndarray) -> dict[str, float]:
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = float(u_interior.mean())
    std_temp = float(u_interior.std())
    pct_above_18 = float(np.sum(u_interior > 18) / u_interior.size * 100)
    pct_below_15 = float(np.sum(u_interior < 15) / u_interior.size * 100)
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def plot_result(building_id: str, u: np.ndarray, interior_mask: np.ndarray, out_path: Path) -> None:
    room = u[1:-1, 1:-1]
    room_for_plot = np.where(interior_mask, room, np.nan)
    stats = summary_stats(u, interior_mask)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(room_for_plot, origin="lower", cmap="inferno", vmin=5, vmax=25)
    ax.set_title(
        (
            f"Building {building_id} - steady-state temperature\n"
            f"mean={stats['mean_temp']:.2f} C, std={stats['std_temp']:.2f} C"
        )
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("temperature [C]")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reference simulation for selected floorplans and visualize results."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path containing building_ids.txt and data arrays (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/simulation_viz"),
        help="Directory where PNG files are written (default: outputs/simulation_viz)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Explicit building IDs to visualize, e.g. --ids 10000 10001",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=2,
        help="If --ids is not given, use the first N IDs from building_ids.txt (default: 2)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Maximum Jacobi iterations (default: 20000)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute convergence tolerance (default: 1e-4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num <= 0:
        raise ValueError("--num must be > 0")

    all_ids = read_building_ids(args.data_dir)
    selected_ids = resolve_ids(all_ids, args.ids, args.num)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for bid in selected_ids:
        u0, interior_mask = load_data(args.data_dir, bid)
        u = jacobi(u0, interior_mask, args.max_iter, args.atol)
        out_path = args.out_dir / f"{bid}_simulation.png"
        plot_result(bid, u, interior_mask, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
