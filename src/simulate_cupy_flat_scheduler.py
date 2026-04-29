from os.path import join
import sys
import time
import numpy as np
import cupy as cp


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u0, interior_mask, max_iter, atol=1e-6, check_interval=100):
    u = cp.asarray(u0)
    mask = cp.asarray(interior_mask)

    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[mask]

        # Periodic check to avoid constant CPU-GPU synchronization overhead
        if i % check_interval == 0:
            delta = cp.abs(u[1:-1, 1:-1][mask] - u_new_interior).max()
            if delta < atol:
                u[1:-1, 1:-1][mask] = u_new_interior
                break

        # Normal update
        u[1:-1, 1:-1][mask] = u_new_interior

    return cp.asnumpy(u)


def summary_stats(u, interior_mask):
    # u is a numpy array again, this function is unchanged
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


if __name__ == "__main__":
    # Load data
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype="bool")

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    start_time = time.perf_counter()

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    end_time = time.perf_counter()

    total_sim_time = end_time - start_time
    print(
        f"\n[TIMING] Total simulation time for {N} floorplans: {total_sim_time:.4f} seconds"
    )
    print(f"[TIMING] Average time per floorplan: {total_sim_time / N:.4f} seconds\n")

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
