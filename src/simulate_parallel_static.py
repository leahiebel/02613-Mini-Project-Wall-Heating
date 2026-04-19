from os.path import join
import sys
import math
import time
import multiprocessing as mp
import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1]
        )
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }

def split_into_chunks(items, n_workers):
    """Split building IDs into nearly equal fixed chunks."""
    chunk_size = math.ceil(len(items) / n_workers)
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    return chunks


def process_chunk(building_ids, load_dir, max_iter, atol):
    """Process one fixed chunk of floorplans."""
    results = []
    for bid in building_ids:
        u0, interior_mask = load_data(load_dir, bid)
        u = jacobi(u0, interior_mask, max_iter, atol)
        stats = summary_stats(u, interior_mask)
        results.append((bid, stats))
    return results



if __name__ == "__main__":
    # Load data
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    if len(sys.argv) < 3:
        n_workers = 1
    else:
        n_workers = int(sys.argv[2])

    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Static scheduling: split building IDs into fixed chunks
    chunks = split_into_chunks(building_ids, n_workers)

    t0 = time.perf_counter()

    # Process chunks
    if n_workers == 1:
        chunk_results = [process_chunk(chunks[0], LOAD_DIR, MAX_ITER, ABS_TOL)]
    else:
        with mp.Pool(processes=n_workers) as pool:
            chunk_results = pool.starmap(
                process_chunk,
                [(chunk, LOAD_DIR, MAX_ITER, ABS_TOL) for chunk in chunks]
            )
    
    elapsed = time.perf_counter() - t0

    # Flatten results from all workers
    all_results = []
    for chunk in chunk_results:
        all_results.extend(chunk)

    # Keep original building order
    stats_by_id = {bid: stats for bid, stats in all_results}

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))  # CSV header
    for bid in building_ids:
        stats = stats_by_id[bid]
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    
    print(f"ELAPSED_SECONDS={elapsed:.6f}", file=sys.stderr)