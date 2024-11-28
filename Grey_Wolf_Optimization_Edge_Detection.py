pip install numpy opencv-python matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grey Wolf Optimizer (GWO) Implementation
def gwo_edge_detection(img, wolf_count=10, iterations=50):
    rows, cols = img.shape

    # Objective Function: Maximizes edge quality based on threshold selection
    def fitness(thresholds):
        lower, upper = thresholds
        _, edge_img = cv2.threshold(img, lower, upper, cv2.THRESH_BINARY)
        edge_strength = np.sum(edge_img) / (rows * cols)
        return edge_strength

    # Initialize wolves' positions (thresholds)
    wolves = np.random.randint(0, 256, (wolf_count, 2))
    wolves = np.clip(wolves, 0, 255)  # Ensure thresholds are valid

    # Sort wolves based on fitness
    fitness_vals = np.array([fitness(wolf) for wolf in wolves])
    sorted_indices = np.argsort(-fitness_vals)
    wolves = wolves[sorted_indices]

    alpha, beta, delta = wolves[:3]  # Best wolves

    # GWO Iterations
    for t in range(iterations):
        a = 2 - 2 * (t / iterations)  # Decreasing factor
        for i in range(wolf_count):
            r1, r2 = np.random.random(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(C1 * alpha - wolves[i])
            X1 = alpha - A1 * D_alpha

            r1, r2 = np.random.random(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(C2 * beta - wolves[i])
            X2 = beta - A2 * D_beta

            r1, r2 = np.random.random(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(C3 * delta - wolves[i])
            X3 = delta - A3 * D_delta

            # Update wolf's position
            wolves[i] = np.clip((X1 + X2 + X3) / 3, 0, 255)

        # Evaluate fitness again and update alpha, beta, delta
        fitness_vals = np.array([fitness(wolf) for wolf in wolves])
        sorted_indices = np.argsort(-fitness_vals)
        wolves = wolves[sorted_indices]
        alpha, beta, delta = wolves[:3]

    # Return the best solution (alpha wolf)
    return alpha

# Load image
image = cv2.imread('/content/grayscale.jpg', cv2.IMREAD_GRAYSCALE)

# Apply GWO for edge detection
best_thresholds = gwo_edge_detection(image)
lower_threshold, upper_threshold = best_thresholds

# Apply Canny Edge Detection using GWO thresholds
edges = cv2.Canny(image, int(lower_threshold), int(upper_threshold))
print("Dinesh Kumar G-1BM22CS091")
# Display Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Edge Detection (GWO)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
