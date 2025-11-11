import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# GMM PARAMETERS (from your visualization)
# ============================================================================

# Mixing coefficients
pi = np.array([0.35, 0.25, 0.25, 0.15])

# Component means
means = np.array([[0, 0], [1.8, 1.5], [5, 0], [0, 5]])

# Covariance matrices
covs = [
    np.array([[1.0, 0.3], [0.3, 1.0]]),
    np.array([[1.0, 0.2], [0.2, 1.0]]),
    np.array([[0.8, 0.1], [0.1, 0.8]]),
    np.array([[1.2, 0.1], [0.1, 1.0]])
]


# ============================================================================
# DATASET GENERATION FUNCTION
# ============================================================================

def generate_gmm_dataset(n_samples, pi, means, covs, random_seed=None):
    """
    Generate samples from a Gaussian Mixture Model

    Args:
        n_samples: Total number of samples to generate
        pi: Mixing coefficients (must sum to 1)
        means: List of mean vectors for each component
        covs: List of covariance matrices for each component
        random_seed: Random seed for reproducibility

    Returns:
        X: Generated samples (n_samples x 2)
        y: Component labels (1, 2, 3, or 4)
        component_counts: Number of samples from each component
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_components = len(pi)

    # Sample component assignments based on mixing coefficients
    component_assignments = np.random.choice(n_components, size=n_samples, p=pi)

    # Count samples per component
    component_counts = np.bincount(component_assignments, minlength=n_components)

    # Generate samples
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    sample_idx = 0
    for k in range(n_components):
        n_k = component_counts[k]
        if n_k > 0:
            # Generate samples from component k
            samples_k = np.random.multivariate_normal(means[k], covs[k], size=n_k)
            X[sample_idx:sample_idx + n_k] = samples_k
            y[sample_idx:sample_idx + n_k] = k + 1  # Labels are 1, 2, 3, 4
            sample_idx += n_k

    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y, component_counts


# ============================================================================
# GENERATE MULTIPLE DATASETS
# ============================================================================

def generate_all_datasets():
    """
    Generate datasets with 10, 100, and 1000 samples
    """
    dataset_sizes = [10, 100, 1000]
    datasets = {}

    print("=" * 60)
    print("GENERATING GMM DATASETS")
    print("=" * 60)
    print("\nGMM Parameters:")
    print(f"  Mixing coefficients π: {pi}")
    print(f"  Component means: {means.tolist()}")

    for n_samples in dataset_sizes:
        print(f"\n{'=' * 50}")
        print(f"Generating dataset with {n_samples} samples")
        print(f"{'=' * 50}")

        # Use different seed for each dataset size
        seed = 42 + n_samples

        # Generate dataset
        X, y, counts = generate_gmm_dataset(n_samples, pi, means, covs, random_seed=seed)

        # Store dataset
        datasets[n_samples] = {
            'X': X,
            'y': y,
            'counts': counts
        }

        # Print statistics
        print(f"✓ Generated {n_samples} samples")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Component distribution:")
        for k in range(4):
            expected = pi[k] * n_samples
            actual = counts[k]
            print(f"    Component {k + 1}: {actual:3d} samples (expected: {expected:.1f})")

        print(f"  Feature statistics:")
        print(
            f"    X₁: mean={X[:, 0].mean():6.3f}, std={X[:, 0].std():6.3f}, range=[{X[:, 0].min():6.3f}, {X[:, 0].max():6.3f}]")
        print(
            f"    X₂: mean={X[:, 1].mean():6.3f}, std={X[:, 1].std():6.3f}, range=[{X[:, 1].min():6.3f}, {X[:, 1].max():6.3f}]")

    return datasets


# ============================================================================
# VISUALIZE DATASETS
# ============================================================================

def visualize_datasets(datasets):
    """
    Create scatter plots for all three datasets
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['red', 'blue', 'green', 'purple']

    for idx, n_samples in enumerate([10, 100, 1000]):
        ax = axes[idx]
        X = datasets[n_samples]['X']
        y = datasets[n_samples]['y']

        # Plot samples colored by component
        for k in range(4):
            mask = (y == k + 1)
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=colors[k], label=f'Component {k + 1}',
                       alpha=0.7, s=50 if n_samples == 10 else 20)

        # Mark true means
        ax.scatter(means[:, 0], means[:, 1],
                   color='black', marker='x', s=50, linewidths=3,
                   label='True means', zorder=5)

        # Formatting
        ax.set_xlabel('X₁', fontsize=11)
        ax.set_ylabel('X₂', fontsize=11)
        ax.set_title(f'N = {n_samples} samples', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 8)
        ax.set_ylim(-3, 8)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('GMM Samples: Increasing Dataset Sizes', fontsize=14)
    plt.tight_layout()
    plt.savefig('gmm_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# SAVE DATASETS TO CSV
# ============================================================================

def save_datasets_to_csv(datasets):
    """
    Save datasets to CSV files for later use
    """
    print("\n" + "=" * 60)
    print("SAVING DATASETS TO CSV FILES")
    print("=" * 60)

    for n_samples in [10, 100, 1000]:
        filename = f'gmm_dataset_{n_samples}.csv'

        # Create DataFrame
        df = pd.DataFrame({
            'X1': datasets[n_samples]['X'][:, 0],
            'X2': datasets[n_samples]['X'][:, 1],
            'Component': datasets[n_samples]['y']
        })

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"✓ Saved {filename} ({len(df)} samples)")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_datasets(datasets):
    """
    Perform statistical analysis on generated datasets
    """
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # True statistics
    true_mean = np.sum([pi[k] * means[k] for k in range(4)], axis=0)
    print(f"\nTrue GMM Statistics:")
    print(f"  Population mean: [{true_mean[0]:.3f}, {true_mean[1]:.3f}]")

    print("\nSample Statistics vs True Values:")
    print("-" * 50)
    print(f"{'N':>6} | {'Sample Mean X₁':>14} | {'Sample Mean X₂':>14} | {'Error Norm':>11}")
    print("-" * 50)

    for n_samples in [10, 100, 1000]:
        X = datasets[n_samples]['X']
        sample_mean = X.mean(axis=0)
        error = np.linalg.norm(sample_mean - true_mean)

        print(f"{n_samples:6d} | {sample_mean[0]:14.3f} | {sample_mean[1]:14.3f} | {error:11.3f}")

    print("\nObservation: Sample mean converges to true mean as N increases")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "=" * 70)
    print(" GMM DATASET GENERATION")
    print("=" * 70)

    # Generate datasets
    datasets = generate_all_datasets()

    # Visualize
    visualize_datasets(datasets)

    # Analyze
    analyze_datasets(datasets)

    # Save to CSV
    save_datasets_to_csv(datasets)

    print("\n" + "=" * 70)
    print(" DATASET GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - gmm_dataset_10.csv")
    print("  - gmm_dataset_100.csv")
    print("  - gmm_dataset_1000.csv")
    print("  - gmm_datasets.png (visualization)")

    return datasets


if __name__ == "__main__":
    datasets = main()

    # Example: Access a specific dataset
    print("\nExample usage:")
    print("  X_100 = datasets[100]['X']  # 100x2 feature matrix")
    print("  y_100 = datasets[100]['y']  # 100x1 component labels")