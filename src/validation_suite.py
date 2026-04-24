from __future__ import annotations
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import csv
import inspect
import types

import numpy as np


STAT_KEYS = ("mean_temp", "std_temp", "pct_above_18", "pct_below_15")
EXPECTED_COLUMNS = ("building_id", *STAT_KEYS)


@dataclass
class SolverRun:
    rows: list[dict[str, float | str]]
    grids: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


def load_module(module_path: Path) -> types.ModuleType:
    spec = spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_validation_ids(ids_file: Path) -> list[str]:
    if not ids_file.exists():
        raise FileNotFoundError(f"Validation subset file not found: {ids_file}")

    ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Validation subset file is empty: {ids_file}")
    return ids


def load_floorplan(data_dir: Path, building_id: str) -> tuple[np.ndarray, np.ndarray]:
    size = 512
    domain_path = data_dir / f"{building_id}_domain.npy"
    interior_path = data_dir / f"{building_id}_interior.npy"

    if not domain_path.exists() or not interior_path.exists():
        raise FileNotFoundError(
            f"Missing data for {building_id}: {domain_path.name} and/or {interior_path.name}"
        )

    u = np.zeros((size + 2, size + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(domain_path)
    interior_mask = np.load(interior_path).astype(bool)
    return u, interior_mask


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


def call_solver(
    solver_fn: object,
    u0: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float,
) -> np.ndarray:
    signature = inspect.signature(solver_fn)
    parameters = signature.parameters

    if len(parameters) >= 4:
        return solver_fn(u0, interior_mask, max_iter, atol)
    if len(parameters) == 3:
        return solver_fn(u0, interior_mask, max_iter)
    raise TypeError(
        "Solver function must accept either (u, interior_mask, max_iter) or "
        "(u, interior_mask, max_iter, atol)."
    )


def run_solver(
    data_dir: Path,
    building_ids: list[str],
    module_path: Path,
    solver_name: str,
    max_iter: int,
    atol: float,
) -> SolverRun:
    module = load_module(module_path)
    if not hasattr(module, solver_name):
        raise AttributeError(f"{module_path} has no function named '{solver_name}'")

    solver_fn = getattr(module, solver_name)
    rows: list[dict[str, float | str]] = []
    grids: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}

    for bid in building_ids:
        u0, interior_mask = load_floorplan(data_dir, bid)
        u = call_solver(solver_fn, u0, interior_mask, max_iter, atol)
        stats = summary_stats(u, interior_mask)
        rows.append({"building_id": bid, **stats})
        grids[bid] = u
        masks[bid] = interior_mask

    return SolverRun(rows=rows, grids=grids, masks=masks)


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPECTED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def validate_row_schema(rows: list[dict[str, float | str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        keys = tuple(row.keys())
        if keys != EXPECTED_COLUMNS:
            errors.append(
                "Unexpected columns for building "
                f"{row.get('building_id', '<unknown>')}: got {keys}, expected {EXPECTED_COLUMNS}"
            )
    return errors


def compare_stats(
    reference_rows: list[dict[str, float | str]],
    candidate_rows: list[dict[str, float | str]],
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    mismatches: list[str] = []
    if len(reference_rows) != len(candidate_rows):
        mismatches.append(
            f"Different number of rows: reference={len(reference_rows)}, candidate={len(candidate_rows)}"
        )
        return mismatches

    for ref_row, cand_row in zip(reference_rows, candidate_rows):
        ref_bid = str(ref_row["building_id"])
        cand_bid = str(cand_row["building_id"])

        if ref_bid != cand_bid:
            mismatches.append(f"Mismatched building_id order: reference={ref_bid}, candidate={cand_bid}")
            continue

        for key in STAT_KEYS:
            ref_value = float(ref_row[key])
            cand_value = float(cand_row[key])
            if not np.isclose(ref_value, cand_value, atol=abs_tol, rtol=rel_tol):
                delta = abs(ref_value - cand_value)
                mismatches.append(
                    f"{ref_bid}::{key} differs: reference={ref_value:.10f}, "
                    f"candidate={cand_value:.10f}, abs_delta={delta:.10f}"
                )

    return mismatches


def save_visual_checks(
    out_dir: Path,
    building_ids: list[str],
    reference: SolverRun,
    candidate: SolverRun,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for --make-viz") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    for bid in building_ids:
        mask = reference.masks[bid]
        ref_room = np.where(mask, reference.grids[bid][1:-1, 1:-1], np.nan)
        cand_room = np.where(mask, candidate.grids[bid][1:-1, 1:-1], np.nan)
        diff = np.abs(ref_room - cand_room)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

        ref_im = axes[0].imshow(ref_room, origin="lower", cmap="inferno", vmin=5, vmax=25)
        axes[0].set_title(f"{bid} - reference")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(ref_im, ax=axes[0], fraction=0.046, pad=0.04)

        cand_im = axes[1].imshow(cand_room, origin="lower", cmap="inferno", vmin=5, vmax=25)
        axes[1].set_title(f"{bid} - candidate")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(cand_im, ax=axes[1], fraction=0.046, pad=0.04)

        diff_im = axes[2].imshow(diff, origin="lower", cmap="magma")
        axes[2].set_title(f"{bid} - abs difference")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(diff_im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.savefig(out_dir / f"{bid}_comparison.png", dpi=170)
        plt.close(fig)


def build_report(
    building_ids: list[str],
    mismatch_lines: list[str],
    schema_errors: list[str],
    reference_csv: Path,
    candidate_csv: Path,
) -> str:
    lines: list[str] = []
    lines.append("Validation subset run")
    lines.append(f"Buildings tested: {', '.join(building_ids)}")
    lines.append(f"Reference CSV: {reference_csv}")
    lines.append(f"Candidate CSV: {candidate_csv}")

    if schema_errors:
        lines.append("CSV/schema errors:")
        lines.extend(f"- {msg}" for msg in schema_errors)

    if mismatch_lines:
        lines.append("Statistic mismatches:")
        lines.extend(f"- {msg}" for msg in mismatch_lines)
    else:
        lines.append("Statistic mismatches: none")

    return "\n".join(lines) + "\n"
