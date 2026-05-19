"""
benchmark_tests.py
------------------
Verification tests for ResearchMOACOR_MINLP_FEA.

The purpose of this file is not to prove global optimality, but to verify that
important optimizer mechanisms work correctly:
1. Basic continuous convergence: Sphere.
2. Curved valley behavior: Rosenbrock.
3. Multimodal search: Rastrigin.
4. Constraint handling: constrained quadratic.
5. Mixed-integer repair/sampling: simple MINLP.
6. Multi-objective Pareto archive: ZDT1-style test.

Run:
    python benchmark_tests.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from optimization_main import ResearchMOACOR_MINLP_FEA, Solution, find_knee_point

logging.basicConfig(level=logging.WARNING, format="%(message)s")

OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def sphere_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    return [float(np.sum(x**2))]


def rosenbrock_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    value = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    return [float(value)]


def rastrigin_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    n = len(x)
    value = 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))
    return [float(value)]


def constrained_quadratic_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    return [float(x[0] ** 2 + x[1] ** 2)]


def constrained_quadratic_g1(x: np.ndarray, fea: Optional[Dict] = None) -> float:
    # x1 + x2 >= 1 written as g(x) <= 0.
    return float(1.0 - x[0] - x[1])


def minlp_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    # optimum x = [2.5, 3, 1]
    return [float((x[0] - 2.5) ** 2 + (x[1] - 3.0) ** 2 + (x[2] - 1.0) ** 2)]


def zdt1_objective(x: np.ndarray, fea: Optional[Dict] = None) -> List[float]:
    f1 = float(x[0])
    g = 1.0 + 9.0 * float(np.sum(x[1:])) / (len(x) - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return [f1, f2]


def build_optimizer(
    objective_function: Callable,
    bounds: Sequence[tuple],
    variable_types: Sequence[str],
    seed: int,
    n_ants: int = 40,
    n_iterations: int = 80,
    constraints: Optional[List[Callable]] = None,
    constraint_scales: Optional[List[float]] = None,
    pareto_archive_size: int = 80,
) -> ResearchMOACOR_MINLP_FEA:
    return ResearchMOACOR_MINLP_FEA(
        objective_function=objective_function,
        bounds=bounds,
        variable_types=variable_types,
        inequality_constraints=constraints or [],
        constraint_scales=constraint_scales,
        n_ants=n_ants,
        n_iterations=n_iterations,
        archive_size=35,
        pareto_archive_size=min(pareto_archive_size, 50),
        q=0.25,
        xi_initial=0.85,
        xi_final=0.05,
        penalty_initial=10.0,
        penalty_final=1e6,
        feasibility_tolerance=1e-8,
        mutation_probability=0.06,
        local_search_frequency=20,
        local_search_steps=4,
        restart_patience=40,
        restart_fraction=0.40,
        evaluation_backend="serial",
        n_workers=1,
        random_seed=seed,
        verbose=False,
    )


def best_solution(solutions: List[Solution]) -> Solution:
    feasible = [solution for solution in solutions if solution.feasible]
    candidates = feasible if feasible else solutions
    return min(candidates, key=lambda solution: float(np.sum(solution.penalized_objectives)))


def run_single_objective_test(name: str, optimizer: ResearchMOACOR_MINLP_FEA, known_f: float) -> Dict:
    solutions = optimizer.optimize()
    best = best_solution(solutions)
    optimizer.export_history_csv(str(OUTPUT_DIR / f"{name}_history.csv"))
    return {
        "test": name,
        "best_f": float(best.objectives[0]),
        "known_f": known_f,
        "absolute_error": abs(float(best.objectives[0]) - known_f),
        "feasible": bool(best.feasible),
        "norm_violation": float(best.norm_violation),
        "best_x": best.x.tolist(),
        "pareto_size": len(optimizer.pareto_archive),
    }


def run_zdt1_test() -> Dict:
    optimizer = build_optimizer(
        objective_function=zdt1_objective,
        bounds=[(0.0, 1.0)] * 30,
        variable_types=["continuous"] * 30,
        seed=6,
        n_ants=50,
        n_iterations=120,
        pareto_archive_size=80,
    )
    solutions = optimizer.optimize()
    archive = optimizer.pareto_archive if optimizer.pareto_archive else solutions
    knee = find_knee_point(archive)
    optimizer.export_pareto_csv(str(OUTPUT_DIR / "zdt1_pareto.csv"))
    optimizer.export_history_csv(str(OUTPUT_DIR / "zdt1_history.csv"))

    if archive:
        f1 = [solution.objectives[0] for solution in archive]
        f2 = [solution.objectives[1] for solution in archive]
        plt.figure(figsize=(8, 5))
        plt.scatter(f1, f2)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("ZDT1 Pareto Front Approximation")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "zdt1_pareto.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "test": "zdt1_multi_objective",
        "best_f": None,
        "known_f": None,
        "absolute_error": None,
        "feasible": True,
        "norm_violation": 0.0,
        "best_x": knee.x.tolist() if knee is not None else [],
        "pareto_size": len(archive),
    }


def write_summary(rows: List[Dict]) -> None:
    csv_path = OUTPUT_DIR / "benchmark_summary.csv"
    fieldnames = [
        "test",
        "best_f",
        "known_f",
        "absolute_error",
        "feasible",
        "norm_violation",
        "pareto_size",
        "best_x",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    tests = []

    tests.append(run_single_objective_test(
        "sphere_10d",
        build_optimizer(
            sphere_objective,
            bounds=[(-5.12, 5.12)] * 10,
            variable_types=["continuous"] * 10,
            seed=1,
            n_ants=40,
            n_iterations=80,
        ),
        known_f=0.0,
    ))

    tests.append(run_single_objective_test(
        "rosenbrock_5d",
        build_optimizer(
            rosenbrock_objective,
            bounds=[(-2.0, 2.0)] * 5,
            variable_types=["continuous"] * 5,
            seed=2,
            n_ants=45,
            n_iterations=100,
        ),
        known_f=0.0,
    ))

    tests.append(run_single_objective_test(
        "rastrigin_10d",
        build_optimizer(
            rastrigin_objective,
            bounds=[(-5.12, 5.12)] * 10,
            variable_types=["continuous"] * 10,
            seed=3,
            n_ants=45,
            n_iterations=100,
        ),
        known_f=0.0,
    ))

    tests.append(run_single_objective_test(
        "constrained_quadratic",
        build_optimizer(
            constrained_quadratic_objective,
            bounds=[(0.0, 2.0), (0.0, 2.0)],
            variable_types=["continuous", "continuous"],
            seed=4,
            n_ants=40,
            n_iterations=80,
            constraints=[constrained_quadratic_g1],
            constraint_scales=[1.0],
        ),
        known_f=0.5,
    ))

    tests.append(run_single_objective_test(
        "simple_minlp",
        build_optimizer(
            minlp_objective,
            bounds=[(0.0, 5.0), (0.0, 5.0), (0.0, 1.0)],
            variable_types=["continuous", "integer", "binary"],
            seed=5,
            n_ants=40,
            n_iterations=80,
        ),
        known_f=0.0,
    ))

    tests.append(run_zdt1_test())
    write_summary(tests)

    print("Benchmark summary")
    print("=================")
    for row in tests:
        print(row)
    print(f"\nSaved benchmark outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
