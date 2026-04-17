#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def load_floorplan(data_dir: Path, building_id: str) -> tuple[np.ndarray, np.ndarray]:
    domain_path = data_dir / f"{building_id}_domain.npy"
    interior_path = data_dir / f"{building_id}_interior.npy"

    if not domain_path.exists() or not interior_path.exists():
        raise FileNotFoundError(
            f"Missing data for {building_id}: {domain_path.name} and/or {interior_path.name}"
        )

    domain = np.load(domain_path)
    interior_mask = np.load(interior_path).astype(bool)

    if domain.shape != interior_mask.shape:
        raise ValueError(
            f"Shape mismatch for {building_id}: domain={domain.shape}, interior={interior_mask.shape}"
        )
    return domain, interior_mask


def build_class_map(domain: np.ndarray, interior_mask: np.ndarray) -> np.ndarray:
    # 0=outside/other, 1=cold wall, 2=warm wall, 3=interior-update point
    class_map = np.zeros_like(domain, dtype=np.uint8)
    class_map[domain == 5] = 1
    class_map[domain == 25] = 2
    class_map[interior_mask] = 3
    return class_map


def plot_floorplan(building_id: str, domain: np.ndarray, interior_mask: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].imshow(domain, cmap="inferno", origin="lower")
    axes[0].set_title(f"{building_id} - domain values")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("temperature/init value")

    im1 = axes[1].imshow(interior_mask, cmap="gray_r", origin="lower")
    axes[1].set_title("interior mask")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_ticks([0, 1])
    cbar1.set_ticklabels(["non-interior", "interior"])

    class_map = build_class_map(domain, interior_mask)
    class_colors = ["#111111", "#2c7bb6", "#d7191c", "#fdae61"]
    class_cmap = mcolors.ListedColormap(class_colors)
    class_norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    im2 = axes[2].imshow(class_map, cmap=class_cmap, norm=class_norm, origin="lower")
    axes[2].set_title("type map")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_ticks([0, 1, 2, 3])
    cbar2.set_ticklabels(["outside/other", "cold wall (5)", "warm wall (25)", "interior"])

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize input data (domain/interior) for selected floorplans."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path containing building_ids.txt and *_domain.npy/*_interior.npy (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/input_viz"),
        help="Directory where PNG files are written (default: outputs/input_viz)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Explicit building IDs to visualize, e.g. --ids 00001 00042 01000",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="If --ids is not given, visualize the first N IDs from building_ids.txt (default: 3)",
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
        domain, interior_mask = load_floorplan(args.data_dir, bid)
        out_path = args.out_dir / f"{bid}_input.png"
        plot_floorplan(bid, domain, interior_mask, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
