#Ant Colony Optimization (ACO) for vehicle routing
import numpy as np
import random
import matplotlib.pyplot as plt

# Define problem: Coordinates of cities (customers and depot)
# First city is the depot, other cities are customers
depot = np.array([0, 0])
customers = np.array([
    [2, 3], [4, 7], [6, 2], [8, 5], [3, 6]
])
num_customers = len(customers)
cities = np.vstack([depot, customers])

# Distance function (Euclidean distance)
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Distance matrix (distances between each pair of cities)
distance_matrix = np.array([[euclidean_distance(cities[i], cities[j]) for j in range(num_customers + 1)] for i in range(num_customers + 1)])

# ACO parameters
num_vehicles = 3  # Number of vehicles
num_ants = 10  # Number of ants
num_iterations = 10  # Number of iterations
alpha = 1  # Pheromone importance
beta = 2   # Heuristic importance (inverse of distance)
rho = 0.5  # Evaporation rate
initial_pheromone = 0.1
vehicle_capacity = 10  # Maximum capacity each vehicle can carry
demand = [4, 3, 6, 5, 7]  # Demand of each customer

# Initialize pheromone matrix (pheromone values on edges between cities)
pheromones = np.full((num_customers + 1, num_customers + 1), initial_pheromone)

# Helper function to calculate probabilities for moving from one city to another
def calculate_probabilities(current_city, unvisited_cities, pheromones, distances, alpha, beta):
    pheromone = pheromones[current_city][unvisited_cities]
    heuristic = 1 / distances[current_city][unvisited_cities]
    attractiveness = (pheromone ** alpha) * (heuristic ** beta)
    probabilities = attractiveness / attractiveness.sum()
    return probabilities

# Helper function to check vehicle capacity
def check_capacity(route, demand, capacity):
    load = 0
    for city in route:
        load += demand[city - 1] if city != 0 else 0  # demand[city-1] because city 0 is the depot
    return load <= capacity

# Main ACO loop
best_distance = float('inf')
best_solution = None

for iteration in range(num_iterations):
    all_routes = []
    all_distances = []

    for ant in range(num_ants):
        # Each ant starts from depot (city 0)
        current_city = 0
        unvisited_cities = set(range(1, num_customers + 1))
        route = [[current_city]]  # Vehicle routes
        total_distance = 0
        vehicle_idx = 0  # First vehicle

        while unvisited_cities:
            unvisited_list = list(unvisited_cities)
            probabilities = calculate_probabilities(current_city, unvisited_list, pheromones, distance_matrix, alpha, beta)
            next_city = np.random.choice(unvisited_list, p=probabilities)
            unvisited_cities.remove(next_city)
            
            # Add city to the current vehicle route
            route[vehicle_idx].append(next_city)
            total_distance += distance_matrix[current_city][next_city]

            # If the vehicle's capacity is exceeded, switch to the next vehicle
            if not check_capacity(route[vehicle_idx], demand, vehicle_capacity):
                vehicle_idx += 1
                route.append([0])  # Starting with depot for the new vehicle

            current_city = next_city

        # Return to depot
        total_distance += distance_matrix[route[vehicle_idx][-1]][0]

        # Store route and total distance for the ant
        all_routes.append(route)
        all_distances.append(total_distance)

        # Update best solution
        if total_distance < best_distance:
            best_distance = total_distance
            best_solution = route

    # Update pheromones based on solutions found by ants
    pheromones *= (1 - rho)  # Evaporation
    for route, dist in zip(all_routes, all_distances):
        pheromone_contribution = 1 / dist
        for vehicle_route in route:
            for i in range(len(vehicle_route) - 1):
                pheromones[vehicle_route[i]][vehicle_route[i + 1]] += pheromone_contribution
                pheromones[vehicle_route[i + 1]][vehicle_route[i]] += pheromone_contribution  # Symmetric update

    print(f"Iteration {iteration + 1}: Best Distance = {best_distance:.2f}")

# Output the best solution
print("\nOptimal Route:")
for vehicle_route in best_solution:
    print(vehicle_route)
print("Optimal Distance:", best_distance)

# Plot the best solution
def plot_solution(solution, cities):
    for route in solution:
        route_coords = np.array([cities[city] for city in route])
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Optimal Vehicle Routing')
    plt.show()

plot_solution(best_solution, cities)
