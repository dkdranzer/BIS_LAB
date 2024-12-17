import random
import numpy as np

# Problem Parameters
NUM_ENERGY_SOURCES = 3  # e.g., solar, wind, grid
DEMAND = 100            # Total energy demand (arbitrary units)
print("Dinesh kumar G-1BM22CS091")
# GEP Parameters
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

# Fitness Function
def fitness(solution):
    total_energy = sum(solution)
    if total_energy < DEMAND:
        return float('-inf')  # Penalize under-supply
    return -abs(total_energy - DEMAND)  # Minimize difference to demand

# Generate a random solution (chromosome)
def random_solution():
    return [random.randint(0, DEMAND) for _ in range(NUM_ENERGY_SOURCES)]

# Mutation operator
def mutate(solution):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, NUM_ENERGY_SOURCES - 1)
        solution[idx] = random.randint(0, DEMAND)
    return solution

# Crossover operator
def crossover(parent1, parent2):
    split = random.randint(1, NUM_ENERGY_SOURCES - 1)
    child = parent1[:split] + parent2[split:]
    return child

# Selection operator (tournament selection)
def select(population, fitnesses):
    best = None
    for _ in range(TOURNAMENT_SIZE):
        candidate = random.choice(population)
        if best is None or fitness(candidate) > fitness(best):
            best = candidate
    return best

# Main GEP loop
def gep_optimize():
    # Initialize population
    population = [random_solution() for _ in range(POPULATION_SIZE)]
    
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for each solution
        fitnesses = [fitness(sol) for sol in population]
        
        # Create next generation
        new_population = []
        for _ in range(POPULATION_SIZE):
            # Select parents
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            
            # Perform crossover
            child = crossover(parent1, parent2)
            
            # Apply mutation
            child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Track the best solution
        best_solution = max(population, key=fitness)
        best_fitness = fitness(best_solution)
        
        print(f"Generation {generation+1}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")
    
    return best_solution

# Run the GEP optimizer
best_solution = gep_optimize()
print(f"Optimized Energy Distribution: {best_solution}")
