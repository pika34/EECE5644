import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data
hidden_sizes = [2, 4, 6, 8, 10, 20]
val_errors = [0.3900, 0.1740, 0.1820, 0.1580, 0.1660, 0.1610]

# Convert to 2D matrix (6 rows, 1 column)
errors_matrix = np.array(val_errors).reshape(len(hidden_sizes), 1)

# Plot heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(errors_matrix, annot=True, fmt=".4f",
            xticklabels=["Validation Error"],
            yticklabels=hidden_sizes,
            cmap="viridis")

plt.xlabel("Metric")
plt.ylabel("Hidden Layer Size")
plt.title("MLP Cross-Validation Error Heatmap")
plt.show()
