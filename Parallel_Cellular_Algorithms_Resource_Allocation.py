import numpy as np

# Problem Parameters
tasks = 5                  # Number of tasks
resources = 3              # Number of resources
max_resources = [10, 8, 6]  # Maximum available units of each resource
task_demands = [3, 4, 2, 5, 6]  # Resource demands for each task
print("Dinesh kumar G-1BM22CS091")
# Cellular Automata Parameters
iterations = 10         # Maximum number of iterations
grid = np.zeros(tasks, dtype=int)  # Grid representing resource allocation for tasks
local_interaction_radius = 1       # Range of interaction with neighboring cells

# Initialize the grid with random allocations
for i in range(tasks):
    grid[i] = np.random.randint(0, resources)

# Function to calculate fitness of a cell's state
def calculate_fitness(grid, task_demands, max_resources):
    resource_usage = [0] * resources
    for task, resource in enumerate(grid):
        resource_usage[resource] += task_demands[task]
    
    # Penalize over-allocation beyond max_resources
    penalty = sum(max(0, resource_usage[r] - max_resources[r]) for r in range(resources))
    return -penalty  # Fitness improves as penalty reduces

# Cellular Update Rules
def update_cell(task, grid, task_demands, max_resources, local_interaction_radius):
    current_state = grid[task]
    best_state = current_state
    best_fitness = calculate_fitness(grid, task_demands, max_resources)

    # Explore neighboring states
    for new_state in range(resources):
        temp_grid = grid.copy()
        temp_grid[task] = new_state
        new_fitness = calculate_fitness(temp_grid, task_demands, max_resources)
        
        # Update best state if fitness improves
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_state = new_state

    return best_state

# Main Cellular Evolution Process
for _ in range(iterations):
    new_grid = grid.copy()
    for task in range(tasks):
        # Update each task (cell) based on local rules
        new_grid[task] = update_cell(task, grid, task_demands, max_resources, local_interaction_radius)
    grid = new_grid

    # Check if the solution is feasible (no over-allocation)
    fitness = calculate_fitness(grid, task_demands, max_resources)
    if fitness == 0:  # Perfect allocation found
        break

# Final Results
print("Final Allocation of Resources to Tasks:", grid)
print("Fitness of Final Allocation:", calculate_fitness(grid, task_demands, max_resources))
