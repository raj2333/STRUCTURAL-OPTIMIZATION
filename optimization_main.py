"""
optimization_main.py
--------------------
Reusable multi-objective ACOR-MINLP optimizer with optional FEA integration.

Main upgrades in this version:
1. Separate feasibility_tolerance and equality_tolerance.
2. Stronger optimizer parameter validation.
3. Near-duplicate filtering for Pareto and near-feasible archives.
4. Integer-variable archive-probability sampling instead of only Gaussian rounding.
5. Mutation for continuous, integer, and binary variables.
6. Adaptive coordinate local search with shrinking step size.
7. Pareto spacing metric and knee-point selection.
8. Optional logging-style progress output.
9. Utility functions for exporting archive and selecting engineering solutions.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[float]
LOGGER = logging.getLogger(__name__)


@dataclass
class Solution:
    x: np.ndarray
    objectives: np.ndarray
    violation: float
    norm_violation: float
    feasible: bool
    penalized_objectives: np.ndarray
    fea_results: Optional[Dict] = None
    crowding: float = 0.0
    rank: int = 10**9


# ============================================================
# Pareto and ranking utilities
# ============================================================

def dominates_objectives(a: Solution, b: Solution) -> bool:
    """Pareto dominance for minimization objectives only."""
    return bool(np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives))


def constrained_dominates(a: Solution, b: Solution) -> bool:
    """
    Feasibility-first constrained dominance.

    Rule:
    1. Feasible dominates infeasible.
    2. Among feasible solutions, use objective Pareto dominance.
    3. Among infeasible solutions, smaller normalized violation dominates.
    """
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    if a.feasible and b.feasible:
        return dominates_objectives(a, b)
    return a.norm_violation < b.norm_violation


def non_dominated_filter(solutions: List[Solution], constrained: bool = True) -> List[Solution]:
    if not solutions:
        return []

    output: List[Solution] = []
    dominance_rule = constrained_dominates if constrained else dominates_objectives

    for i, solution in enumerate(solutions):
        dominated = False
        for j, other in enumerate(solutions):
            if i != j and dominance_rule(other, solution):
                dominated = True
                break
        if not dominated:
            output.append(solution)
    return output


def assign_crowding_distance(front: List[Solution]) -> List[Solution]:
    n = len(front)
    if n == 0:
        return front
    if n <= 2:
        for solution in front:
            solution.crowding = math.inf
        return front

    distances = np.zeros(n, dtype=float)
    objectives = np.array([solution.objectives for solution in front], dtype=float)
    n_objectives = objectives.shape[1]

    for objective_index in range(n_objectives):
        order = np.argsort(objectives[:, objective_index])
        distances[order[0]] = math.inf
        distances[order[-1]] = math.inf

        objective_min = objectives[order[0], objective_index]
        objective_max = objectives[order[-1], objective_index]
        span = objective_max - objective_min
        if span <= 1e-30:
            continue

        for k in range(1, n - 1):
            if math.isinf(distances[order[k]]):
                continue
            distances[order[k]] += (
                objectives[order[k + 1], objective_index]
                - objectives[order[k - 1], objective_index]
            ) / span

    for i, solution in enumerate(front):
        solution.crowding = float(distances[i])
    return front


def remove_near_duplicates(
    solutions: Iterable[Solution],
    x_tolerance: float = 1e-8,
    objective_tolerance: float = 1e-8,
) -> List[Solution]:
    """Remove solutions that are almost identical in design or objective space."""
    unique: List[Solution] = []
    for candidate in solutions:
        duplicate = False
        for accepted in unique:
            same_x = np.linalg.norm(candidate.x - accepted.x) <= x_tolerance
            same_f = np.linalg.norm(candidate.objectives - accepted.objectives) <= objective_tolerance
            if same_x or same_f:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def fast_fronts(solutions: List[Solution]) -> List[List[Solution]]:
    """Simple non-dominated front construction using constrained dominance."""
    remaining = solutions[:]
    fronts: List[List[Solution]] = []
    rank = 0

    while remaining:
        front = non_dominated_filter(remaining, constrained=True)
        for solution in front:
            solution.rank = rank
        fronts.append(assign_crowding_distance(front))
        front_ids = {id(solution) for solution in front}
        remaining = [solution for solution in remaining if id(solution) not in front_ids]
        rank += 1
    return fronts


def sort_by_constrained_fronts(solutions: List[Solution]) -> List[Solution]:
    fronts = fast_fronts(solutions)
    ordered: List[Solution] = []
    for front in fronts:
        ordered.extend(sorted(front, key=lambda solution: solution.crowding, reverse=True))
    return ordered


def pareto_spacing_metric(solutions: List[Solution]) -> float:
    """
    Spacing metric for a Pareto archive. Lower is more uniformly distributed.
    Returns 0.0 when the archive has fewer than 3 solutions.
    """
    if len(solutions) < 3:
        return 0.0

    objectives = np.array([solution.objectives for solution in solutions], dtype=float)
    mins = objectives.min(axis=0)
    spans = objectives.max(axis=0) - mins
    spans[spans <= 1e-30] = 1.0
    normalized = (objectives - mins) / spans

    nearest_distances = []
    for i in range(len(normalized)):
        distances = np.linalg.norm(normalized[i] - normalized[np.arange(len(normalized)) != i], axis=1)
        nearest_distances.append(float(np.min(distances)))

    nearest = np.array(nearest_distances)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def find_knee_point(solutions: List[Solution]) -> Optional[Solution]:
    """
    Select a practical compromise solution from a Pareto set.

    The method normalizes each objective between 0 and 1 and returns the solution
    with the minimum Euclidean distance to the ideal point. This is a simple and
    defensible knee/compromise choice for engineering reports.
    """
    if not solutions:
        return None

    feasible = [solution for solution in solutions if solution.feasible]
    candidates = feasible if feasible else solutions

    objectives = np.array([solution.objectives for solution in candidates], dtype=float)
    mins = objectives.min(axis=0)
    spans = objectives.max(axis=0) - mins
    spans[spans <= 1e-30] = 1.0
    normalized = (objectives - mins) / spans
    distances = np.linalg.norm(normalized, axis=1)
    return candidates[int(np.argmin(distances))]


# ============================================================
# Main optimizer
# ============================================================

class ResearchMOACOR_MINLP_FEA:
    """
    Multi-objective ACOR-like optimizer for constrained MINLP structural problems.

    ACOR interpretation:
    - Archive = pheromone memory.
    - Rank-based weights = pheromone intensity.
    - Continuous variables = Gaussian kernel sampling around archived solutions.
    - Integer variables = empirical archive probability sampling with mutation.
    - Binary variables = Bernoulli sampling from archive frequency.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray, Optional[Dict]], Sequence[float]],
        bounds: Sequence[Tuple[float, float]],
        variable_types: Sequence[str],
        fea_function: Optional[Callable[[np.ndarray], Dict]] = None,
        inequality_constraints: Optional[List[Callable[[np.ndarray, Optional[Dict]], float]]] = None,
        equality_constraints: Optional[List[Callable[[np.ndarray, Optional[Dict]], float]]] = None,
        constraint_scales: Optional[Sequence[float]] = None,
        n_ants: int = 100,
        n_iterations: int = 300,
        archive_size: int = 60,
        pareto_archive_size: int = 150,
        q: float = 0.25,
        xi_initial: float = 0.85,
        xi_final: float = 0.05,
        penalty_initial: float = 10.0,
        penalty_final: float = 1e6,
        feasibility_tolerance: float = 1e-8,
        equality_tolerance: float = 1e-5,
        mutation_probability: float = 0.05,
        local_search_frequency: int = 25,
        local_search_steps: int = 3,
        restart_patience: int = 50,
        restart_fraction: float = 0.45,
        duplicate_x_tolerance: float = 1e-8,
        duplicate_objective_tolerance: float = 1e-8,
        evaluation_backend: str = "thread",  # "serial", "thread", or "process"
        n_workers: int = 4,
        random_seed: int = 1,
        verbose: bool = True,
    ):
        self.objective_function = objective_function
        self.fea_function = fea_function
        self.bounds = np.array(bounds, dtype=float)
        self.variable_types = list(variable_types)
        self.inequality_constraints = inequality_constraints or []
        self.equality_constraints = equality_constraints or []

        self.n_variables = len(self.bounds)
        self.n_constraints = len(self.inequality_constraints) + len(self.equality_constraints)
        self.constraint_scales = self._prepare_constraint_scales(constraint_scales)

        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.archive_size = int(archive_size)
        self.pareto_archive_size = int(pareto_archive_size)
        self.q = float(q)
        self.xi_initial = float(xi_initial)
        self.xi_final = float(xi_final)
        self.penalty_initial = float(penalty_initial)
        self.penalty_final = float(penalty_final)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.equality_tolerance = float(equality_tolerance)
        self.mutation_probability = float(mutation_probability)
        self.local_search_frequency = int(local_search_frequency)
        self.local_search_steps = int(local_search_steps)
        self.restart_patience = int(restart_patience)
        self.restart_fraction = float(restart_fraction)
        self.duplicate_x_tolerance = float(duplicate_x_tolerance)
        self.duplicate_objective_tolerance = float(duplicate_objective_tolerance)
        self.evaluation_backend = evaluation_backend
        self.n_workers = int(n_workers)
        self.verbose = verbose

        self._validate_inputs()

        self.rng = np.random.default_rng(random_seed)
        self.archive: List[Solution] = []
        self.pareto_archive: List[Solution] = []
        self.near_feasible_archive: List[Solution] = []

        self.history = {
            "best_violation": [],
            "best_norm_violation": [],
            "pareto_size": [],
            "near_feasible_size": [],
            "spread_indicator": [],
            "spacing_metric": [],
            "xi": [],
            "penalty": [],
        }

    def _validate_inputs(self) -> None:
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (lower, upper) pairs")
        if len(self.bounds) != len(self.variable_types):
            raise ValueError("bounds and variable_types must have the same length")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("each bound must satisfy upper > lower")
        for variable_type in self.variable_types:
            if variable_type not in {"continuous", "integer", "binary"}:
                raise ValueError(f"Unsupported variable type: {variable_type}")
        if self.evaluation_backend not in {"serial", "thread", "process"}:
            raise ValueError("evaluation_backend must be 'serial', 'thread', or 'process'")
        if self.n_ants <= 0:
            raise ValueError("n_ants must be positive")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if self.archive_size < 2:
            raise ValueError("archive_size must be at least 2")
        if self.pareto_archive_size < 2:
            raise ValueError("pareto_archive_size must be at least 2")
        if not (0.0 < self.q <= 1.0):
            raise ValueError("q must satisfy 0 < q <= 1")
        if self.xi_initial <= 0 or self.xi_final <= 0:
            raise ValueError("xi_initial and xi_final must be positive")
        if self.penalty_initial <= 0 or self.penalty_final < self.penalty_initial:
            raise ValueError("penalties must satisfy 0 < penalty_initial <= penalty_final")
        if self.feasibility_tolerance < 0 or self.equality_tolerance < 0:
            raise ValueError("tolerances must be non-negative")
        if not (0.0 <= self.mutation_probability <= 1.0):
            raise ValueError("mutation_probability must be between 0 and 1")
        if not (0.0 <= self.restart_fraction < 1.0):
            raise ValueError("restart_fraction must satisfy 0 <= restart_fraction < 1")
        for i, variable_type in enumerate(self.variable_types):
            if variable_type in {"integer", "binary"}:
                low, high = self.bounds[i]
                if math.floor(high) < math.ceil(low):
                    raise ValueError(f"integer/binary variable {i} has no valid integer value")

    def _prepare_constraint_scales(self, scales: Optional[Sequence[float]]) -> np.ndarray:
        n_constraints = self.n_constraints
        if n_constraints == 0:
            return np.ones(1)
        if scales is None:
            return np.ones(n_constraints)

        array = np.array(scales, dtype=float)
        if len(array) != n_constraints:
            raise ValueError("constraint_scales must match the number of constraints")
        array[array <= 0] = 1.0
        return array

    def adaptive_xi(self, iteration: int, stagnation_ratio: float = 0.0) -> float:
        t = iteration / max(1, self.n_iterations - 1)
        base = self.xi_initial * (1.0 - t) + self.xi_final * t
        return min(1.5, base * (1.0 + 0.5 * stagnation_ratio))

    def adaptive_penalty(self, iteration: int) -> float:
        t = iteration / max(1, self.n_iterations - 1)
        return self.penalty_initial * (self.penalty_final / self.penalty_initial) ** t

    def random_solution(self) -> np.ndarray:
        x = np.zeros(self.n_variables, dtype=float)
        for i, (low, high) in enumerate(self.bounds):
            variable_type = self.variable_types[i]
            if variable_type == "continuous":
                x[i] = self.rng.uniform(low, high)
            elif variable_type == "integer":
                x[i] = self.rng.integers(math.ceil(low), math.floor(high) + 1)
            elif variable_type == "binary":
                x[i] = self.rng.integers(0, 2)
        return self.repair_solution(x)

    def repair_solution(self, x: ArrayLike) -> np.ndarray:
        x = np.array(x, dtype=float).copy()
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        for i, variable_type in enumerate(self.variable_types):
            if variable_type == "integer":
                x[i] = np.round(x[i])
            elif variable_type == "binary":
                x[i] = 1.0 if x[i] >= 0.5 else 0.0
        return x

    def raw_constraint_values(self, x: np.ndarray, fea: Optional[Dict]) -> np.ndarray:
        values = []
        for constraint in self.inequality_constraints:
            values.append(max(0.0, float(constraint(x, fea))))
        for constraint in self.equality_constraints:
            h_abs = abs(float(constraint(x, fea)))
            values.append(max(0.0, h_abs - self.equality_tolerance))
        if not values:
            return np.zeros(1)
        return np.array(values, dtype=float)

    def constraint_violation(self, x: np.ndarray, fea: Optional[Dict]) -> Tuple[float, float]:
        raw = self.raw_constraint_values(x, fea)
        if self.n_constraints == 0:
            return 0.0, 0.0
        normalized = raw / self.constraint_scales
        return float(np.sum(raw**2)), float(np.sum(normalized**2))

    def evaluate_solution(self, x: ArrayLike, iteration: int) -> Solution:
        x = self.repair_solution(x)
        fea = self.fea_function(x) if self.fea_function is not None else None
        objectives = np.array(self.objective_function(x, fea), dtype=float)
        violation, norm_violation = self.constraint_violation(x, fea)
        penalty = self.adaptive_penalty(iteration)
        penalized_objectives = objectives + penalty * norm_violation
        feasible = norm_violation <= self.feasibility_tolerance
        return Solution(
            x=x,
            objectives=objectives,
            violation=violation,
            norm_violation=norm_violation,
            feasible=feasible,
            penalized_objectives=penalized_objectives,
            fea_results=fea,
        )

    def initialize_archive(self) -> None:
        self.archive = [self.evaluate_solution(self.random_solution(), 0) for _ in range(self.archive_size)]
        self.archive = sort_by_constrained_fronts(self.archive)[: self.archive_size]
        self.update_pareto_archives(self.archive)

    def rank_weights(self) -> np.ndarray:
        """ACOR rank-based pheromone weights."""
        archive_length = len(self.archive)
        ranks = np.arange(1, archive_length + 1, dtype=float)
        denominator = self.q * archive_length * math.sqrt(2.0 * math.pi)
        weights = (1.0 / denominator) * np.exp(
            -((ranks - 1.0) ** 2) / (2.0 * (self.q * archive_length) ** 2)
        )
        weights /= np.sum(weights)
        return weights

    def _sample_integer_from_archive(self, archive_array: np.ndarray, j: int) -> float:
        low, high = self.bounds[j]
        possible_values = np.arange(math.ceil(low), math.floor(high) + 1)
        archived_values, counts = np.unique(archive_array[:, j].astype(int), return_counts=True)
        probability = np.ones(len(possible_values), dtype=float) * 1e-3
        for value, count in zip(archived_values, counts):
            index = np.where(possible_values == value)[0]
            if len(index) > 0:
                probability[index[0]] += count
        probability /= probability.sum()
        sampled = float(self.rng.choice(possible_values, p=probability))
        if self.rng.random() < self.mutation_probability:
            sampled += float(self.rng.choice([-1, 1]))
        return sampled

    def generate_new_solution(self, weights: np.ndarray, iteration: int, stagnation_ratio: float) -> np.ndarray:
        archive_length = len(self.archive)
        selected_index = self.rng.choice(archive_length, p=weights)
        selected = self.archive[selected_index].x
        archive_array = np.array([solution.x for solution in self.archive])
        xi = self.adaptive_xi(iteration, stagnation_ratio)

        x_new = np.zeros(self.n_variables, dtype=float)
        for j, variable_type in enumerate(self.variable_types):
            low, high = self.bounds[j]

            if variable_type == "binary":
                p_one = np.clip(np.mean(archive_array[:, j]), 0.05, 0.95)
                x_new[j] = 1.0 if self.rng.random() < p_one else 0.0
                if self.rng.random() < self.mutation_probability:
                    x_new[j] = 1.0 - x_new[j]

            elif variable_type == "integer":
                x_new[j] = self._sample_integer_from_archive(archive_array, j)

            else:
                average_distance = np.mean(np.abs(archive_array[:, j] - selected[j]))
                sigma = xi * average_distance
                sigma = max(sigma, 1e-4 * (high - low))
                x_new[j] = self.rng.normal(selected[j], sigma)
                if self.rng.random() < self.mutation_probability:
                    x_new[j] = self.rng.uniform(low, high)

        return self.repair_solution(x_new)

    def evaluate_population(self, xs: List[np.ndarray], iteration: int) -> List[Solution]:
        if self.evaluation_backend == "serial" or self.n_workers <= 1:
            return [self.evaluate_solution(x, iteration) for x in xs]

        if self.evaluation_backend == "thread":
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                return list(executor.map(partial(self.evaluate_solution, iteration=iteration), xs))

        try:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                return list(executor.map(partial(self.evaluate_solution, iteration=iteration), xs))
        except Exception as exc:
            if self.verbose:
                LOGGER.warning("Process backend failed; falling back to thread backend: %s", exc)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                return list(executor.map(partial(self.evaluate_solution, iteration=iteration), xs))

    def accept_better(self, candidate: Solution, current: Solution) -> bool:
        if constrained_dominates(candidate, current):
            return True
        if candidate.feasible == current.feasible:
            return np.sum(candidate.penalized_objectives) < np.sum(current.penalized_objectives)
        return False

    def local_search(self, start: Solution, iteration: int) -> Solution:
        """Adaptive coordinate search around an elite solution."""
        current = start
        base_steps = []
        for i, variable_type in enumerate(self.variable_types):
            low, high = self.bounds[i]
            if variable_type == "continuous":
                base_steps.append(0.05 * (high - low))
            else:
                base_steps.append(1.0)

        for shrink in range(max(1, self.local_search_steps)):
            improved_this_pass = False
            factor = 0.5**shrink
            for i, variable_type in enumerate(self.variable_types):
                candidates = []
                if variable_type == "continuous":
                    step = base_steps[i] * factor
                    for direction in (-1.0, 1.0):
                        x = current.x.copy()
                        x[i] += direction * step
                        candidates.append(x)
                elif variable_type == "integer":
                    for direction in (-1.0, 1.0):
                        x = current.x.copy()
                        x[i] += direction
                        candidates.append(x)
                elif variable_type == "binary":
                    x = current.x.copy()
                    x[i] = 1.0 - x[i]
                    candidates.append(x)

                evaluated = [self.evaluate_solution(candidate, iteration) for candidate in candidates]
                for candidate in evaluated:
                    if self.accept_better(candidate, current):
                        current = candidate
                        improved_this_pass = True
            if not improved_this_pass and shrink > 0:
                break
        return current

    def update_pareto_archives(self, candidates: List[Solution]) -> None:
        combined = remove_near_duplicates(
            self.pareto_archive + self.near_feasible_archive + candidates,
            x_tolerance=self.duplicate_x_tolerance,
            objective_tolerance=self.duplicate_objective_tolerance,
        )

        feasible = [solution for solution in combined if solution.feasible]
        if feasible:
            pareto = non_dominated_filter(feasible, constrained=False)
            pareto = remove_near_duplicates(
                pareto,
                x_tolerance=self.duplicate_x_tolerance,
                objective_tolerance=self.duplicate_objective_tolerance,
            )
            pareto = assign_crowding_distance(pareto)
            pareto = sorted(pareto, key=lambda solution: solution.crowding, reverse=True)
            self.pareto_archive = pareto[: self.pareto_archive_size]

        near = non_dominated_filter(combined, constrained=True)
        near = remove_near_duplicates(
            near,
            x_tolerance=self.duplicate_x_tolerance,
            objective_tolerance=self.duplicate_objective_tolerance,
        )
        near = assign_crowding_distance(near)
        near = sorted(
            near,
            key=lambda solution: (solution.feasible, -solution.norm_violation, solution.crowding),
            reverse=True,
        )
        self.near_feasible_archive = near[: self.pareto_archive_size]

    def spread_indicator(self) -> float:
        if len(self.pareto_archive) < 2:
            return 0.0
        objectives = np.array([solution.objectives for solution in self.pareto_archive])
        minimums = objectives.min(axis=0)
        maximums = objectives.max(axis=0)
        return float(np.sum(maximums - minimums))

    def restart_archive(self, iteration: int) -> None:
        keep_count = max(2, int(self.archive_size * (1.0 - self.restart_fraction)))
        kept = self.archive[:keep_count]
        new_random = [
            self.evaluate_solution(self.random_solution(), iteration)
            for _ in range(self.archive_size - keep_count)
        ]
        self.archive = sort_by_constrained_fronts(kept + new_random)[: self.archive_size]

    def optimize(self) -> List[Solution]:
        start_time = time.time()
        self.initialize_archive()

        no_improvement = 0
        best_spread = self.spread_indicator()
        best_feasible_count = len(self.pareto_archive)

        for iteration in range(self.n_iterations):
            stagnation_ratio = no_improvement / max(1, self.restart_patience)
            weights = self.rank_weights()

            ants = [self.generate_new_solution(weights, iteration, stagnation_ratio) for _ in range(self.n_ants)]
            evaluated = self.evaluate_population(ants, iteration)

            self.archive = sort_by_constrained_fronts(self.archive + evaluated)[: self.archive_size]

            if self.local_search_frequency > 0 and iteration % self.local_search_frequency == 0:
                elite = self.archive[0]
                improved = self.local_search(elite, iteration)
                self.archive = sort_by_constrained_fronts(self.archive + [improved])[: self.archive_size]

            self.update_pareto_archives(self.archive + evaluated)

            current_spread = self.spread_indicator()
            current_feasible_count = len(self.pareto_archive)
            improved_archive = (
                current_feasible_count > best_feasible_count
                or current_spread > best_spread * 1.001
            )

            if improved_archive:
                no_improvement = 0
                best_spread = max(best_spread, current_spread)
                best_feasible_count = max(best_feasible_count, current_feasible_count)
            else:
                no_improvement += 1

            if no_improvement >= self.restart_patience:
                self.restart_archive(iteration)
                no_improvement = 0
                if self.verbose:
                    LOGGER.info("Restart applied at iteration %s", iteration)

            best = self.archive[0]
            self.history["best_violation"].append(best.violation)
            self.history["best_norm_violation"].append(best.norm_violation)
            self.history["pareto_size"].append(len(self.pareto_archive))
            self.history["near_feasible_size"].append(len(self.near_feasible_archive))
            self.history["spread_indicator"].append(current_spread)
            self.history["spacing_metric"].append(pareto_spacing_metric(self.pareto_archive))
            self.history["xi"].append(self.adaptive_xi(iteration, stagnation_ratio))
            self.history["penalty"].append(self.adaptive_penalty(iteration))

            if self.verbose and (iteration % 20 == 0 or iteration == self.n_iterations - 1):
                LOGGER.info(
                    "Iter %4d | feasible Pareto: %3d | near archive: %3d | "
                    "best normalized violation: %.3e | spacing: %.3e | xi: %.4f",
                    iteration,
                    len(self.pareto_archive),
                    len(self.near_feasible_archive),
                    best.norm_violation,
                    self.history["spacing_metric"][-1],
                    self.history["xi"][-1],
                )

        if self.verbose:
            LOGGER.info("Optimization completed in %.2f seconds", time.time() - start_time)

        return self.pareto_archive if self.pareto_archive else self.near_feasible_archive

    def get_knee_solution(self) -> Optional[Solution]:
        archive = self.pareto_archive if self.pareto_archive else self.near_feasible_archive
        return find_knee_point(archive)

    def export_pareto_csv(self, filepath: str) -> None:
        archive = self.pareto_archive if self.pareto_archive else self.near_feasible_archive
        if not archive:
            return

        n_objectives = len(archive[0].objectives)
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            header = (
                [f"x{i + 1}" for i in range(self.n_variables)]
                + [f"f{i + 1}" for i in range(n_objectives)]
                + ["violation", "norm_violation", "feasible", "crowding", "rank"]
            )
            writer.writerow(header)
            for solution in archive:
                writer.writerow(
                    list(solution.x)
                    + list(solution.objectives)
                    + [
                        solution.violation,
                        solution.norm_violation,
                        solution.feasible,
                        solution.crowding,
                        solution.rank,
                    ]
                )

    def export_history_csv(self, filepath: str) -> None:
        if not self.history["best_norm_violation"]:
            return
        keys = list(self.history.keys())
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration"] + keys)
            for i in range(len(self.history["best_norm_violation"])):
                writer.writerow([i] + [self.history[key][i] for key in keys])
