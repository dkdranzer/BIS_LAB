# Hyperparameter Optimization for decision Trees
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("Dinesh Kumar G-1BM22CS091") 
# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# PSO Parameters
NUM_PARTICLES = 20
MAX_ITERATIONS = 10
INERTIA_WEIGHT = 0.7
COGNITIVE_WEIGHT = 1.5
SOCIAL_WEIGHT = 1.5
MIN_DEPTH = 1
MAX_DEPTH = 20

# Fitness function: Validation accuracy of the Decision Tree
def fitness_function(depth):
    depth = int(depth)
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy

# Initialize particles
particles = np.random.uniform(MIN_DEPTH, MAX_DEPTH, NUM_PARTICLES)  # Particle positions (depths)
velocities = np.random.uniform(-1, 1, NUM_PARTICLES)  # Particle velocities
personal_best_positions = particles.copy()
personal_best_scores = np.array([fitness_function(p) for p in particles])  # Fitness of personal bests
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]  # Best position globally

# PSO Algorithm
for iteration in range(MAX_ITERATIONS):
    for i in range(NUM_PARTICLES):
        # Update velocity
        r1, r2 = np.random.rand(), np.random.rand()  # Random factors
        cognitive_component = COGNITIVE_WEIGHT * r1 * (personal_best_positions[i] - particles[i])
        social_component = SOCIAL_WEIGHT * r2 * (global_best_position - particles[i])
        velocities[i] = INERTIA_WEIGHT * velocities[i] + cognitive_component + social_component

        # Update particle position
        particles[i] += velocities[i]
        particles[i] = np.clip(particles[i], MIN_DEPTH, MAX_DEPTH)  # Ensure depth stays within bounds

        # Evaluate fitness
        current_fitness = fitness_function(particles[i])
        if current_fitness > personal_best_scores[i]:  # Update personal best
            personal_best_positions[i] = particles[i]
            personal_best_scores[i] = current_fitness
    
    # Update global best
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    print(f"Iteration {iteration + 1}: Best Depth = {int(global_best_position)}, Best Fitness = {max(personal_best_scores):.4f}")

# Output the best result
optimal_depth = int(global_best_position)
optimal_score = max(personal_best_scores)
print(f"\nOptimal Depth: {optimal_depth}")
print(f"Validation Accuracy at Optimal Depth: {optimal_score:.4f}")
