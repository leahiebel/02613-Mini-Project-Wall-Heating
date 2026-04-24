# 02613 Mini Project — Wall Heating

This project studies a wall-heating concept by solving a steady-state heat equation on many building floorplans. The goal is to understand whether heating the interior walls of a building can produce room temperatures that are warm enough and sufficiently uniform.

## Documentation precedence

Use repository documentation in this order:

2. `README.md` — project overview and practical usage
3. `description/tasks.md` — assignment deliverables
4. `description/project.md` — domain background and mathematical model

## Project background

The project is based on the **Modified Swiss Dwellings** dataset, which contains **4571 building floorplans**. Each floorplan encodes a building layout with two types of walls:

- **inside walls**: treated as heated walls at a fixed high temperature
- **load-bearing walls**: left unheated and kept cold

The idea is to place heating elements inside the interior walls instead of using conventional radiators or underfloor heating. The question is whether this produces acceptable indoor temperatures across a large and varied set of real building geometries.

### Physical assumptions

To keep the model tractable, the problem is formulated in two dimensions.

- inside walls are fixed at **25°C**
- load-bearing walls are fixed at **5°C**

The task is to compute the resulting steady-state temperature field inside the rooms.

### Mathematical model

Let $u(x, y)$ denote the temperature at position $(x, y)$. The steady-state field satisfies Laplace’s equation:

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$

with Dirichlet boundary conditions:

- $u(x, y) = 5$ on load-bearing walls
- $u(x, y) = 25$ on inside walls

### Numerical method

The continuous problem is discretized on a $S \times S$ grid. For each interior room point, the temperature is updated as the average of its four neighbors:

$$
u[i, j] \leftarrow \frac{1}{4}(u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
$$

This is the Jacobi method.

Important details:

- interior room points are updated
- wall points remain fixed at their prescribed temperatures
- points outside the building are not updated
- iteration continues until convergence or a maximum iteration count is reached

### Input data

Each building is stored as a $514 \times 514$ simulation grid with two NumPy files:

- `{building_id}_domain.npy` — initial temperature grid
	- load-bearing walls are set to 5
	- inside walls are set to 25
	- all other points start at 0
- `{building_id}_interior.npy` — binary mask for interior room points
	- `1` means the point should be updated
	- `0` means the point is a wall or outside the building

## What the simulation produces

After the Jacobi iterations, heat diffuses from the warm walls into the rooms while the cold load-bearing walls keep nearby regions cooler. The result is a smooth temperature field over the building interior.

For each floorplan, the simulation reports four summary statistics:

1. mean room temperature
2. standard deviation of room temperature
3. percentage of room area above 18°C
4. percentage of room area below 15°C

These quantities are used to judge whether wall heating is a viable heating strategy.

## Repository structure

- `README.md` — project overview and usage guide
- `description/project.md` — domain background and mathematical model
- `description/tasks.md` — assignment tasks and deliverables
- `src/` — simulation implementation
- `scripts/` — batch-job templates for CPU and GPU runs
- `outputs/` — CSV outputs from simulation runs
- `job_outputs/` — batch job stdout and stderr files

## Running the simulation

The main reference implementation is [src/simulate.py](src/simulate.py). It loads the floorplans, runs the Jacobi solver, and prints CSV-formatted summary statistics.

## Visualize input floorplans

Use [scripts/visualize_inputs.py](scripts/visualize_inputs.py) to inspect the raw inputs for selected buildings. The script saves PNG images with three panels per building:

- domain values
- interior mask
- type map (outside/cold wall/warm wall/interior)

Generate images for the first 3 building IDs:

```bash
python scripts/visualize_inputs.py --num 3
```

Generate images for specific IDs:

```bash
python scripts/visualize_inputs.py --ids 00001 00042 01000
```

Use a custom data location or output folder:

```bash
python scripts/visualize_inputs.py \
	--data-dir /dtu/projects/02613_2025/data/modified_swiss_dwellings \
	--out-dir outputs/input_viz
```

Images are written to `outputs/input_viz/` by default.

## Visualize simulation results

Use [scripts/visualize_simulation_results.py](scripts/visualize_simulation_results.py) to run the reference Jacobi solver and save steady-state temperature maps.

Generate result images for 2 floorplans (good starting point for task 3):

```bash
python scripts/visualize_simulation_results.py --num 2
```

Generate result images for specific IDs:

```bash
python scripts/visualize_simulation_results.py --ids 10000 10009
```

Control solver settings and output location:

```bash
python scripts/visualize_simulation_results.py \
	--num 2 \
	--max-iter 20000 \
	--atol 1e-4 \
	--out-dir outputs/simulation_viz
```

Images are written to `outputs/simulation_viz/` by default.

## Standardized profiling subset

Use the profiling job when you want to compare solver implementations under the same
conditions on the cluster.

- Fixed profiling IDs are stored in `description/profile_subset_ids.txt`.
- The subset contains 20 buildings and is used for every profiling run.
- The profiling job accepts any solver module/function pair that follows the Jacobi
	signature used by the reference implementation.

Submit the profiling job with optional overrides for the solver file and function name:

```bash
bsub < scripts/CPU_profile_job.sh
bsub < scripts/CPU_profile_job.sh src/simulate_numba_cpu.py jacobi
```

This writes the raw `.lprof` file and a readable `.txt` report into `outputs/profiling/`.

### CPU batch run

Submit the CPU job from the project root:

```bash
bsub < scripts/template_CPU.sh
```

This writes batch logs to `job_outputs/` and CSV output to `outputs/`.

### GPU batch run

Submit the GPU job from the project root:

```bash
bsub < scripts/template_GPU.sh
```

### Output files

- batch stdout and stderr: `job_outputs/`
- simulation CSV files: `outputs/`

