#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation_suite import (  # noqa: E402
    build_report,
    compare_stats,
    read_validation_ids,
    run_solver,
    save_visual_checks,
    validate_row_schema,
    write_csv,
)


DEFAULT_DATA_DIR = Path("/dtu/projects/02613_2025/data/modified_swiss_dwellings")
DEFAULT_IDS_FILE = PROJECT_ROOT / "description" / "validation_subset_ids.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed validation subset and compare a candidate solver against the "
            "reference implementation in src/simulate.py"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path containing building_ids.txt and floorplan arrays (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=DEFAULT_IDS_FILE,
        help=f"Validation building IDs file (default: {DEFAULT_IDS_FILE})",
    )
    parser.add_argument(
        "--reference-module",
        type=Path,
        default=PROJECT_ROOT / "src" / "simulate.py",
        help="Path to the reference module (default: src/simulate.py)",
    )
    parser.add_argument(
        "--candidate-module",
        type=Path,
        default=PROJECT_ROOT / "src" / "simulate_numba_cpu.py",
        help="Path to the candidate module being validated",
    )
    parser.add_argument(
        "--reference-solver",
        default="jacobi",
        help="Reference solver function name (default: jacobi)",
    )
    parser.add_argument(
        "--candidate-solver",
        default="jacobi",
        help="Candidate solver function name (default: jacobi)",
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
        help="Solver convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--compare-abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for summary-stat comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--compare-rel-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for summary-stat comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for validation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--make-viz",
        action="store_true",
        help="Generate side-by-side comparison images for the first 2 validation buildings",
    )
    parser.add_argument(
        "--viz-count",
        type=int,
        default=2,
        help="How many IDs from the subset to visualize when --make-viz is enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_iter <= 0:
        raise ValueError("--max-iter must be > 0")
    if args.viz_count <= 0:
        raise ValueError("--viz-count must be > 0")

    validation_ids = read_validation_ids(args.ids_file)

    reference = run_solver(
        data_dir=args.data_dir,
        building_ids=validation_ids,
        module_path=args.reference_module,
        solver_name=args.reference_solver,
        max_iter=args.max_iter,
        atol=args.atol,
    )
    candidate = run_solver(
        data_dir=args.data_dir,
        building_ids=validation_ids,
        module_path=args.candidate_module,
        solver_name=args.candidate_solver,
        max_iter=args.max_iter,
        atol=args.atol,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_csv = args.output_dir / "reference_stats.csv"
    candidate_csv = args.output_dir / "candidate_stats.csv"
    write_csv(reference.rows, reference_csv)
    write_csv(candidate.rows, candidate_csv)

    schema_errors = validate_row_schema(reference.rows) + validate_row_schema(candidate.rows)
    mismatches = compare_stats(
        reference_rows=reference.rows,
        candidate_rows=candidate.rows,
        abs_tol=args.compare_abs_tol,
        rel_tol=args.compare_rel_tol,
    )

    if args.make_viz:
        viz_ids = validation_ids[: min(args.viz_count, len(validation_ids))]
        save_visual_checks(
            out_dir=args.output_dir / "visual_checks",
            building_ids=viz_ids,
            reference=reference,
            candidate=candidate,
        )

    report = build_report(
        building_ids=validation_ids,
        mismatch_lines=mismatches,
        schema_errors=schema_errors,
        reference_csv=reference_csv,
        candidate_csv=candidate_csv,
    )
    report_path = args.output_dir / "validation_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)

    if schema_errors or mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
