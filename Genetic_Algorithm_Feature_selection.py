#Feature Selection in Machine Learning Models
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dinesh Kumar G-1BM22CS091")
# Parameters for the Genetic Algorithm
POPULATION_SIZE = 20
NUM_GENERATIONS = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 3

# Initialize population (binary encoding for features)
def initialize_population(num_features, population_size):
    return np.random.randint(2, size=(population_size, num_features))

# Fitness function: Model accuracy - Penalty for number of features used
def fitness_function(individual):
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) == 0:  # Avoid division by zero
        return 0
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_sel, y_train)
    predictions = model.predict(X_test_sel)
    accuracy = accuracy_score(y_test, predictions)
    # Penalize based on the number of selected features
    penalty = len(selected_features) / len(individual)
    return accuracy - 0.1 * penalty

# Selection: Tournament Selection
def tournament_selection(population, fitness_scores):
    tournament = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best = max(tournament, key=lambda idx: fitness_scores[idx])
    return population[best]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Mutation: Flip bit mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# Genetic Algorithm
def genetic_algorithm():
    num_features = X.shape[1]
    population = initialize_population(num_features, POPULATION_SIZE)
    
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(ind) for ind in population])
        
        # Track the best individual
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        print(f"Generation {generation}: Best Fitness = {fitness_scores[best_idx]:.4f}")
        
        # Create new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))
        
        population = np.array(new_population)
    
    # Return the best solution
    return best_individual, fitness_scores[best_idx]

# Run the Genetic Algorithm
best_solution, best_fitness = genetic_algorithm()
selected_features = np.where(best_solution == 1)[0]

print("\nBest Feature Subset Indices:", selected_features)
print("Best Fitness (Accuracy - Penalty):", best_fitness)
