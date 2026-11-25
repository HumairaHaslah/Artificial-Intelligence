"""
Genetic Algorithm for Delivery Route Optimisation
Depot: A | Delivery Locations: B, C, D, E, F, G
"""

import random
from typing import Dict, List, Tuple

# Symmetric distance matrix in kilometres (Depot = A, locations = B..G)
LOCATIONS: List[str] = ["A", "B", "C", "D", "E", "F", "G"]
DELIVERY_STOPS: List[str] = ["B", "C", "D", "E", "F", "G"]

DISTANCE_MATRIX: Dict[str, Dict[str, int]] = {
    "A": {"A": 0, "B": 4, "C": 8, "D": 6, "E": 7, "F": 9, "G": 5},
    "B": {"A": 4, "B": 0, "C": 5, "D": 7, "E": 3, "F": 8, "G": 6},
    "C": {"A": 8, "B": 5, "C": 0, "D": 4, "E": 6, "F": 3, "G": 7},
    "D": {"A": 6, "B": 7, "C": 4, "D": 0, "E": 5, "F": 6, "G": 8},
    "E": {"A": 7, "B": 3, "C": 6, "D": 5, "E": 0, "F": 4, "G": 6},
    "F": {"A": 9, "B": 8, "C": 3, "D": 6, "E": 4, "F": 0, "G": 5},
    "G": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 6, "F": 5, "G": 0},
}


def calculate_route_distance(chromosome: List[str]) -> int:
    """
    Calculate total round-trip distance (in km) for a chromosome (list of B..G),
    starting at depot 'A' and returning to depot 'A'.
    """
    full_route = ["A"] + chromosome + ["A"]
    total_distance = 0
    for i in range(len(full_route) - 1):
        origin = full_route[i]
        destination = full_route[i + 1]
        total_distance += DISTANCE_MATRIX[origin][destination]
    return total_distance


def fitness_score(chromosome: List[str]) -> float:
    """Higher fitness corresponds to a shorter total route distance."""
    dist = calculate_route_distance(chromosome)
    return 1.0 / dist if dist > 0 else float("inf")


def create_initial_population(pop_size: int) -> List[List[str]]:
    """Generate an initial population of valid random permutations of B..G."""
    population: List[List[str]] = []
    for _ in range(pop_size):
        individual = DELIVERY_STOPS.copy()
        random.shuffle(individual)
        population.append(individual)
    return population


def tournament_selection(population: List[List[str]], tournament_size: int = 3) -> List[str]:
    """Select one parent route using tournament selection based on fitness."""
    competitors = random.sample(population, k=min(tournament_size, len(population)))
    return max(competitors, key=fitness_score)


def order_crossover(parent1: List[str], parent2: List[str]) -> List[str]:
    """
    Order Crossover (OX): preserves a contiguous slice from parent1 and fills
    remaining locations in the order they appear in parent2 (no duplicates/omissions).
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child: List[str | None] = [None] * size
    child[start : end + 1] = parent1[start : end + 1]

    used = set(child[start : end + 1])
    fill_idx = 0
    for gene in parent2:
        if gene not in used:
            while child[fill_idx] is not None:
                fill_idx += 1
            child[fill_idx] = gene

    return [gene for gene in child if gene is not None]


def swap_mutation(chromosome: List[str], mutation_rate: float = 0.15) -> List[str]:
    """With probability mutation_rate, swap two locations to create a new valid route."""
    mutated = chromosome.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def run_genetic_algorithm(
    pop_size: int = 60,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    elitism_count: int = 1,
    seed: int = 42,
) -> Tuple[List[str], int]:
    """Run the Genetic Algorithm for a fixed number of generations."""
    random.seed(seed)
    population = create_initial_population(pop_size)

    for _ in range(generations):
        # Sort population from shortest distance (best) to longest
        population.sort(key=calculate_route_distance)

        # Elitism: carry over the best route(s) unchanged
        next_population: List[List[str]] = [
            ind.copy() for ind in population[:elitism_count]
        ]

        # Generate remaining offspring
        while len(next_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = swap_mutation(child, mutation_rate=mutation_rate)
            next_population.append(child)

        population = next_population

    best_chromosome = min(population, key=calculate_route_distance)
    best_distance = calculate_route_distance(best_chromosome)
    return best_chromosome, best_distance


def format_full_route(chromosome: List[str]) -> str:
    """Format chromosome as A -> ... -> A."""
    return " -> ".join(["A"] + chromosome + ["A"])


if __name__ == "__main__":
    best_chrom, best_km = run_genetic_algorithm(
        pop_size=60,
        generations=100,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_count=1,
        seed=42,
    )
    print("Best Route:", format_full_route(best_chrom))
    print(f"Total Distance: {best_km} km")