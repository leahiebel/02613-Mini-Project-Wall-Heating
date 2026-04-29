from os.path import join
import sys
import time
import numpy as np
from numba import cuda


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# Fused Numba Kernel
@cuda.jit
def jacobi_flag_kernel(u, u_new, interior_mask, flag, atol, do_check):
    i, j = cuda.grid(2)
    ny, nx = u.shape

    if 1 <= i < ny - 1 and 1 <= j < nx - 1:
        if interior_mask[i - 1, j - 1]:
            # Calculate new average
            val = 0.25 * (u[i, j - 1] + u[i, j + 1] + u[i - 1, j] + u[i + 1, j])
            u_new[i, j] = val

            # Dirty Flag Check
            if do_check == 1:
                diff = abs(val - u[i, j])
                if diff >= atol:
                    # If even one thread exceeds tolerance, flag the whole grid as dirty
                    flag[0] = 1
        else:
            u_new[i, j] = u[i, j]


def jacobi(u0, interior_mask, max_iter, atol=1e-4):
    d_u = cuda.to_device(u0)
    d_u_new = cuda.to_device(u0)
    d_mask = cuda.to_device(interior_mask)

    # Allocate a 1-element integer array for our dirty flag
    d_flag = cuda.to_device(np.zeros(1, dtype=np.int32))

    threadsperblock = (16, 16)
    blockspergrid_x = (u0.shape[0] + 15) // 16
    blockspergrid_y = (u0.shape[1] + 15) // 16
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Flat Scheduler
    check_interval = 100

    for it in range(max_iter):
        # Check according to the flat scheduler
        do_check = 1 if (it % check_interval == 0) else 0

        if do_check == 1:
            # Reset the flag to 0 before launching the checking kernel
            d_flag.copy_to_device(np.zeros(1, dtype=np.int32))

        # Launch the asynchronous kernel
        jacobi_flag_kernel[blockspergrid, threadsperblock](
            d_u, d_u_new, d_mask, d_flag, atol, do_check
        )

        if do_check == 1:
            # Read the flag back to the CPU
            flag_val = d_flag.copy_to_host()[0]

            if flag_val == 0:
                d_u = d_u_new
                break

        # Swap pointers for the next iteration
        d_u, d_u_new = d_u_new, d_u

    return d_u.copy_to_host()


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

    all_u = np.empty_like(all_u0)

    start_time = time.perf_counter()

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    end_time = time.perf_counter()

    total_sim_time = end_time - start_time
    print(
        f"\n[TIMING] Task 11 (Dirty Flag) simulation time for {N} floorplans: {total_sim_time:.4f} seconds",
        file=sys.stderr,
    )
    print(
        f"[TIMING] Average time per floorplan: {total_sim_time / N:.4f} seconds\n",
        file=sys.stderr,
    )

    # Print summary statistics in CSV format
    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
