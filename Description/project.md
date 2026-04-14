# Wall Heating Project Background

> Canonical project practices and conventions are defined in `../copilot-instructions.md`.

## Overview

This project evaluates a fictional experimental building heating concept called **Wall Heating**. Instead of using conventional radiators or floor heating, the idea is to place heating elements inside the **interior walls** of a building so that the walls themselves radiate heat into the rooms. To avoid structural issues, **load-bearing walls are left unheated** and therefore remain cold.

The purpose of the project is to assess how effective this heating strategy is across a large and diverse set of building layouts.

## Dataset

The analysis is based on the **Modified Swiss Dwellings** dataset, which contains **4571 building floor plans**. These floor plans indicate both:

- **inside walls**, which are treated as heated walls,
- **load-bearing walls**, which remain cold.

This makes it possible to evaluate the Wall Heating idea on many different real building geometries.

## Physical Assumptions

To simplify the analysis, the heating process is modeled in **two dimensions**.

The temperature assumptions are:

- **Inside walls:** fixed at **25°C**
- **Load-bearing walls:** fixed at **5°C**

The goal is to determine the resulting **steady-state temperature distribution** inside the rooms.

## Mathematical Model

Let `u(x, y)` denote the temperature at position `(x, y)`.

The steady-state temperature field is modeled using **Laplace's equation**:

```text
∂²u/∂x² + ∂²u/∂y² = 0
```

with **Dirichlet boundary conditions**:

- `u(x, y) = 5` on load-bearing walls
- `u(x, y) = 25` on inside walls

So the task is to solve for the temperature distribution in the rooms given fixed wall temperatures.

## Numerical Approach

The continuous problem is discretized on an `S × S` square grid.

If `u[i, j]` denotes the temperature at grid point `(i, j)`, then for each **interior point** the temperature is updated as the average of its four neighbors:

```text
u[i, j] ← 1/4 (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
```

This iterative scheme is the **Jacobi method**.

Important details:

- **Interior room points** are updated.
- **Wall points** remain fixed at their prescribed temperatures.
- **Points outside the building** are not updated.
- Iteration continues until either:
  - a maximum number of iterations is reached, or
  - the solution has converged and the grid no longer changes significantly.

## Simulation Inputs

Each building floor plan has been converted into a **514 × 514 simulation grid**.

For each building, two NumPy files are provided:

### 1. `{building_id}_domain.npy`
Contains the initial temperature grid:

- load-bearing walls are set to **5**,
- inside walls are set to **25**,
- all other points are set to **0** initially.

### 2. `{building_id}_interior.npy`
Contains a binary mask indicating which grid points are interior room points:

- `1` means the point is inside a room and should be updated,
- `0` means the point is either on a wall or outside the building and should not be updated.

## Interpretation of the Simulation

After applying the Jacobi iterations, heat diffuses from the warm inside walls into the rooms, while the colder load-bearing walls keep nearby regions cooler. The resulting temperature field varies smoothly across the building interior.

This lets us evaluate whether Wall Heating produces room temperatures that are both sufficiently warm and reasonably uniform.

## Evaluation Quantities

For each building, the simulation is summarized using four quantities:

1. **Mean room temperature**  
   Measures the overall average temperature inside the rooms.

2. **Standard deviation of room temperature**  
   Measures how uniform or variable the temperature is across the rooms.

3. **Percentage of room area above 18°C**  
   This is important because areas below 18°C have increased risk of mold, so this percentage should ideally be high.

4. **Percentage of room area below 15°C**  
   This is important because areas below 15°C are considered too cold for human comfort, so this percentage should ideally be low.

## Project Goal

The overall goal is to determine whether Wall Heating is a viable heating strategy by simulating the steady-state temperature distribution for many buildings and analyzing the resulting temperature statistics.
