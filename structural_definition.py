"""
structural_definition.py
------------------------
Structural problem definition file using the upgraded ACOR-MINLP optimizer.

This file contains two versions of the same 2D 3-bar truss problem:
1. Continuous sizing: each bar area is a continuous variable.
2. Discrete sizing: each bar area is selected from a standard area catalogue.

Run examples:
    python structural_definition.py continuous
    python structural_definition.py discrete

Default mode is discrete because it better demonstrates MINLP behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from optimization_main import ResearchMOACOR_MINLP_FEA, Solution, find_knee_point

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Problem constants
# ============================================================

E = 210e9
RHO = 7850.0
ALLOWABLE_STRESS = 250e6
ALLOWABLE_DISPLACEMENT = 0.005

NODES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0],
])
ELEMENTS = [(0, 1), (1, 2), (2, 0)]

# Standard section catalogue, in m^2. These are representative educational values.
AVAILABLE_AREAS = np.array([
    0.00010,
    0.00015,
    0.00020,
    0.00030,
    0.00040,
    0.00060,
    0.00080,
    0.00100,
    0.00150,
    0.00200,
    0.00300,
    0.00400,
    0.00600,
    0.00800,
    0.01000,
])


def decode_areas(x: np.ndarray, mode: str) -> np.ndarray:
    """Convert optimizer design variables into physical cross-sectional areas."""
    if mode == "discrete":
        indices = np.clip(np.rint(x).astype(int), 0, len(AVAILABLE_AREAS) - 1)
        return AVAILABLE_AREAS[indices]
    return np.asarray(x, dtype=float)


# ============================================================
# FEA function
# ============================================================

def truss_fea_from_areas(areas: np.ndarray) -> Dict:
    """
    Finite Element Analysis for a 2D 3-bar truss.

    Returns weight, maximum displacement, maximum stress, full displacement vector,
    element stresses, member lengths, and axial forces.
    """
    dof = 2 * len(NODES)
    K = np.zeros((dof, dof))
    lengths = []

    for element_index, (i, j) in enumerate(ELEMENTS):
        xi, yi = NODES[i]
        xj, yj = NODES[j]
        length = float(np.hypot(xj - xi, yj - yi))
        c = (xj - xi) / length
        s = (yj - yi) / length
        lengths.append(length)

        area = max(float(areas[element_index]), 1e-12)
        local_stiffness = (E * area / length) * np.array([
            [ c * c,  c * s, -c * c, -c * s],
            [ c * s,  s * s, -c * s, -s * s],
            [-c * c, -c * s,  c * c,  c * s],
            [-c * s, -s * s,  c * s,  s * s],
        ])

        element_dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(element_dofs, element_dofs)] += local_stiffness

    F = np.zeros(dof)
    F[5] = -10000.0

    fixed_dofs = [0, 1, 2, 3]
    free_dofs = [i for i in range(dof) if i not in fixed_dofs]
    U = np.zeros(dof)

    try:
        U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
    except np.linalg.LinAlgError:
        U[:] = 1e10

    stresses = []
    axial_forces = []
    for element_index, (i, j) in enumerate(ELEMENTS):
        xi, yi = NODES[i]
        xj, yj = NODES[j]
        length = lengths[element_index]
        c = (xj - xi) / length
        s = (yj - yi) / length
        element_dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        strain = np.array([-c, -s, c, s]) @ U[element_dofs] / length
        stress = E * strain
        stresses.append(stress)
        axial_forces.append(stress * areas[element_index])

    lengths = np.array(lengths)
    stresses = np.array(stresses)
    axial_forces = np.array(axial_forces)

    return {
        "areas": areas.copy(),
        "weight": float(RHO * np.sum(areas * lengths)),
        "max_displacement": float(np.max(np.abs(U))),
        "max_stress": float(np.max(np.abs(stresses))),
        "displacements": U,
        "stresses": stresses,
        "axial_forces": axial_forces,
        "lengths": lengths,
    }


def make_truss_fea(mode: str):
    def truss_fea(x: np.ndarray) -> Dict:
        areas = decode_areas(x, mode)
        return truss_fea_from_areas(areas)
    return truss_fea


# ============================================================
# Objective functions and constraints
# ============================================================

def objective_function(x: np.ndarray, fea: Dict) -> List[float]:
    """Minimize structural weight and maximum displacement."""
    return [fea["weight"], fea["max_displacement"]]


def displacement_constraint(x: np.ndarray, fea: Dict) -> float:
    """Global serviceability constraint: max displacement <= allowable displacement."""
    return fea["max_displacement"] - ALLOWABLE_DISPLACEMENT


def make_member_stress_constraint(member_index: int):
    """Create a separate stress constraint for each element."""
    def stress_constraint(x: np.ndarray, fea: Dict) -> float:
        return abs(float(fea["stresses"][member_index])) - ALLOWABLE_STRESS
    stress_constraint.__name__ = f"stress_constraint_member_{member_index + 1}"
    return stress_constraint


def make_simple_buckling_constraint(member_index: int):
    """
    Educational Euler buckling constraint for compression members.

    Assumption: circular solid equivalent section.
    A = pi r^2, I = pi r^4 / 4 = A^2 / (4 pi)
    Compression axial force is treated as negative.
    Constraint form: compression_force - Pcr <= 0
    """
    effective_length_factor = 1.0

    def buckling_constraint(x: np.ndarray, fea: Dict) -> float:
        axial_force = float(fea["axial_forces"][member_index])
        compression = max(0.0, -axial_force)
        area = max(float(fea["areas"][member_index]), 1e-12)
        length = float(fea["lengths"][member_index])
        inertia = area**2 / (4.0 * np.pi)
        p_cr = (np.pi**2 * E * inertia) / ((effective_length_factor * length) ** 2)
        return compression - p_cr

    buckling_constraint.__name__ = f"buckling_constraint_member_{member_index + 1}"
    return buckling_constraint


# ============================================================
# Optimizer setup
# ============================================================

def build_optimizer(mode: str = "discrete") -> ResearchMOACOR_MINLP_FEA:
    if mode not in {"continuous", "discrete"}:
        raise ValueError("mode must be 'continuous' or 'discrete'")

    if mode == "discrete":
        bounds = [(0, len(AVAILABLE_AREAS) - 1)] * len(ELEMENTS)
        variable_types = ["integer"] * len(ELEMENTS)
    else:
        bounds = [(0.0001, 0.01)] * len(ELEMENTS)
        variable_types = ["continuous"] * len(ELEMENTS)

    inequality_constraints = [
        *[make_member_stress_constraint(i) for i in range(len(ELEMENTS))],
        displacement_constraint,
        *[make_simple_buckling_constraint(i) for i in range(len(ELEMENTS))],
    ]

    # Three stress constraints, one displacement constraint, three buckling constraints.
    # Scaling avoids stress/buckling values dominating displacement numerically.
    constraint_scales = [ALLOWABLE_STRESS] * len(ELEMENTS) + [ALLOWABLE_DISPLACEMENT] + [1.0e4] * len(ELEMENTS)

    return ResearchMOACOR_MINLP_FEA(
        objective_function=objective_function,
        fea_function=make_truss_fea(mode),
        bounds=bounds,
        variable_types=variable_types,
        inequality_constraints=inequality_constraints,
        equality_constraints=[],
        constraint_scales=constraint_scales,
        n_ants=60,
        n_iterations=120,
        archive_size=45,
        pareto_archive_size=80,
        q=0.25,
        xi_initial=0.85,
        xi_final=0.05,
        penalty_initial=10.0,
        penalty_final=1e6,
        feasibility_tolerance=1e-8,
        equality_tolerance=1e-5,
        mutation_probability=0.06,
        local_search_frequency=15,
        local_search_steps=4,
        restart_patience=30,
        restart_fraction=0.40,
        duplicate_x_tolerance=1e-10,
        duplicate_objective_tolerance=1e-10,
        evaluation_backend="thread",
        n_workers=4,
        random_seed=10,
        verbose=True,
    )


# ============================================================
# Reporting and plotting helpers
# ============================================================

def solution_areas(solution: Solution, mode: str) -> np.ndarray:
    return decode_areas(solution.x, mode)


def print_solution(label: str, solution: Solution, mode: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print("Design variables:", solution.x)
    print("Physical areas [m^2]:", solution_areas(solution, mode))
    print("Objectives [Weight, Max displacement]:", solution.objectives)
    print("Raw violation:", solution.violation)
    print("Normalized violation:", solution.norm_violation)
    print("Feasible:", solution.feasible)
    if solution.fea_results is not None:
        print("Member stresses [Pa]:", solution.fea_results["stresses"])
        print("Axial forces [N]:", solution.fea_results["axial_forces"])


def print_final_solutions(solutions: List[Solution], mode: str, limit: int = 10) -> None:
    print("\nFinal Pareto / near-feasible solutions")
    print("=====================================")
    for i, solution in enumerate(solutions[:limit], start=1):
        print_solution(f"Solution {i}", solution, mode)

    knee = find_knee_point(solutions)
    if knee is not None:
        print_solution("Recommended knee/compromise solution", knee, mode)

    feasible = [solution for solution in solutions if solution.feasible]
    if feasible:
        lightest = min(feasible, key=lambda solution: solution.objectives[0])
        stiffest = min(feasible, key=lambda solution: solution.objectives[1])
        print_solution("Lightest feasible solution", lightest, mode)
        print_solution("Stiffest feasible solution", stiffest, mode)


def plot_results(optimizer: ResearchMOACOR_MINLP_FEA, solutions: List[Solution], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feasible_solutions = [solution for solution in solutions if solution.feasible]
    plot_set = feasible_solutions if feasible_solutions else solutions

    weights = [solution.objectives[0] for solution in plot_set]
    displacements = [solution.objectives[1] for solution in plot_set]

    plt.figure(figsize=(8, 5))
    plt.scatter(weights, displacements)
    plt.xlabel("Weight")
    plt.ylabel("Maximum displacement")
    plt.title("Pareto Front: Weight vs Maximum Displacement")
    plt.grid(True)
    plt.savefig(output_dir / "pareto_front.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(optimizer.history["pareto_size"], label="Feasible Pareto archive")
    plt.plot(optimizer.history["near_feasible_size"], label="Near-feasible archive")
    plt.xlabel("Iteration")
    plt.ylabel("Archive size")
    plt.title("Archive Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "archive_growth.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(np.maximum(optimizer.history["best_norm_violation"], 1e-30))
    plt.xlabel("Iteration")
    plt.ylabel("Best normalized violation")
    plt.title("Constraint Violation History")
    plt.grid(True)
    plt.savefig(output_dir / "constraint_violation.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(optimizer.history["spacing_metric"])
    plt.xlabel("Iteration")
    plt.ylabel("Pareto spacing metric")
    plt.title("Pareto Distribution Quality")
    plt.grid(True)
    plt.savefig(output_dir / "spacing_metric.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "discrete"
    optimizer = build_optimizer(mode)
    solutions = optimizer.optimize()

    output_dir = Path(f"results_{mode}")
    output_dir.mkdir(exist_ok=True)
    optimizer.export_pareto_csv(str(output_dir / "pareto_results.csv"))
    optimizer.export_history_csv(str(output_dir / "optimization_history.csv"))

    print_final_solutions(solutions, mode)
    plot_results(optimizer, solutions, output_dir)
    print(f"\nSaved CSV files and plots in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
