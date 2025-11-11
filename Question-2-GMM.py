import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# GMM parameters
pi = [0.35, 0.25, 0.25, 0.15]  # Mixing coefficients
means = np.array([[0, 0], [1.8, 1.5], [5, 0], [0, 5]])

# Fixed covariance matrices - must be numpy arrays
covs = [
    np.array([[1.0, 0.3], [0.3, 1.0]]),
    np.array([[1.0, 0.2], [0.2, 1.0]]),
    np.array([[0.8, 0.1], [0.1, 0.8]]),
    np.array([[1.2, 0.1], [0.1, 1.0]])
]

# Generate sample points for visualization
x = np.linspace(-3, 8, 200)
y = np.linspace(-3, 8, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Calculate the mixture PDF
pdf = np.zeros(X.shape)
for k in range(4):
    rv = multivariate_normal(mean=means[k], cov=covs[k])
    pdf += pi[k] * rv.pdf(pos)

# Create the plot
plt.figure(figsize=(8, 7))

# Plot contour lines
contour = plt.contour(X, Y, pdf, levels=20, cmap='viridis', linewidths=1.5)
plt.colorbar(contour, label='Probability Density')

# Mark the component means
plt.scatter(means[:, 0], means[:, 1],
           color='red', marker='x', s=200, linewidths=3,
           label='Component Means', zorder=5)

# Add labels for each mean
for i, (mx, my) in enumerate(means):
    plt.annotate(f'μ{i+1}', xy=(mx, my), xytext=(5, 5),
                textcoords='offset points', fontsize=10, fontweight='bold')

# Formatting
plt.title("True 4-Component GMM with Overlapping Components", fontsize=14)
plt.xlabel("x₁", fontsize=12)
plt.ylabel("x₂", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# Display the plot
plt.show()

# Print GMM parameters for reference
print("GMM Parameters:")
print("-" * 40)
print(f"Mixing coefficients (π): {pi}")
print(f"\nComponent means:")
for i, mean in enumerate(means):
    print(f"  Component {i+1}: {mean}")
print(f"\nOverlap analysis:")
print(f"  Distance μ1-μ2: {np.linalg.norm(means[0] - means[1]):.2f} (high overlap)")
print(f"  Distance μ1-μ3: {np.linalg.norm(means[0] - means[2]):.2f} (well separated)")
print(f"  Distance μ1-μ4: {np.linalg.norm(means[0] - means[3]):.2f} (well separated)")