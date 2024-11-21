import numpy as np
import random

# Define the wireless network problem (e.g., 5 access points and 4 available channels)
num_devices = 5   # Number of devices (access points)
num_channels = 4  # Number of available channels

# Interference matrix: shows the interference between devices (higher value means more interference)
# Lower values in the matrix mean less interference between devices
interference_matrix = np.array([
    [0, 3, 2, 4, 1],  # Device 1
    [3, 0, 5, 3, 2],  # Device 2
    [2, 5, 0, 4, 3],  # Device 3
    [4, 3, 4, 0, 6],  # Device 4
    [1, 2, 3, 6, 0]   # Device 5
])

# Fitness function to evaluate the quality of a channel assignment (lower interference is better)
def fitness(channel_assignment):
    total_interference = 0
    # Calculate total interference based on the assigned channels
    for i in range(num_devices):
        for j in range(i + 1, num_devices):
            if channel_assignment[i] == channel_assignment[j]:  # Same channel -> interference
                total_interference += interference_matrix[i][j]
    return total_interference  # Lower values are better (minimize interference)

# Lévy flight for generating new solutions
def levy_flight(beta=1.5, dim=1):
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# Cuckoo Search Algorithm for channel assignment optimization
class CuckooSearch:
    def __init__(self, fitness_function, num_devices, num_channels, num_nests, max_iter, pa=0.25):
        self.fitness_function = fitness_function
        self.num_devices = num_devices
        self.num_channels = num_channels
        self.num_nests = num_nests
        self.max_iter = max_iter
        self.pa = pa  # Probability of abandoning worst nests
        self.nests = np.random.randint(0, num_channels, (num_nests, num_devices))  # Initial random solutions
        self.fitness_values = np.apply_along_axis(self.fitness_function, 1, self.nests)
        self.best_nest = self.nests[np.argmin(self.fitness_values)]  # Best solution
        self.best_fitness = np.min(self.fitness_values)

    def search(self):
        for iteration in range(self.max_iter):
            # Generate new solutions via Lévy flights
            new_nests = np.copy(self.nests)
            for i in range(self.num_nests):
                step = levy_flight(dim=self.num_devices)
                new_nests[i] = np.clip(self.nests[i] + step, 0, self.num_channels - 1)  # Channel assignment within limits

            # Evaluate new solutions
            new_fitness_values = np.apply_along_axis(self.fitness_function, 1, new_nests)

            # Replace the worst nests with better ones
            for i in range(self.num_nests):
                if new_fitness_values[i] < self.fitness_values[i]:
                    self.nests[i] = new_nests[i]
                    self.fitness_values[i] = new_fitness_values[i]

            # Abandon worst nests
            worst_indices = np.argsort(self.fitness_values)[-int(self.pa * self.num_nests):]
            self.nests[worst_indices] = np.random.randint(0, self.num_channels, (len(worst_indices), self.num_devices))
            self.fitness_values[worst_indices] = np.apply_along_axis(self.fitness_function, 1, self.nests[worst_indices])

            # Update the best solution
            min_fitness_idx = np.argmin(self.fitness_values)
            min_fitness = self.fitness_values[min_fitness_idx]
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_nest = self.nests[min_fitness_idx]

            # Print progress (optional)
            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.best_fitness}")

        return self.best_nest, self.best_fitness

# Example Usage
if __name__ == "__main__":
    # Define problem parameters
    num_nests = 50  # Number of nests (solutions)
    max_iter = 100  # Number of iterations
    cs = CuckooSearch(fitness, num_devices, num_channels, num_nests, max_iter)
   
    best_solution, best_fitness = cs.search()
   
    print(f"Best channel assignment: {best_solution}")
    print(f"Best fitness (total interference): {best_fitness}")

