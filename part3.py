import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# ============================================================================
# CONFIGURATION: Medium Overlap Gaussian Parameters
# ============================================================================

# Mean vectors for 4 classes
means = [
    [0.0, 0.0, 0.0],  # Class 1
    [2.8, 0.3, 0.0],  # Class 2
    [1.4, 2.4, 0.2],  # Class 3
    [1.4, 0.8, 2.5]  # Class 4
]

# Covariance matrices for 4 classes
covariances = [
    [[1.2, 0.3, 0.2],  # Class 1
     [0.3, 1.0, 0.15],
     [0.2, 0.15, 1.1]],

    [[1.5, 0.2, 0.25],  # Class 2
     [0.2, 1.1, 0.2],
     [0.25, 0.2, 1.2]],

    [[1.1, 0.35, 0.2],  # Class 3
     [0.35, 1.4, 0.3],
     [0.2, 0.3, 1.0]],

    [[1.0, 0.25, 0.3],  # Class 4
     [0.25, 1.2, 0.35],
     [0.3, 0.35, 1.3]]
]

# Uniform priors
priors = [0.25, 0.25, 0.25, 0.25]


# ============================================================================
# DATASET GENERATION FUNCTION
# ============================================================================

def generate_gaussian_dataset(n_samples, means, covariances, seed=None):
    """
    Generate dataset from 4-class Gaussian distributions

    Args:
        n_samples: Total number of samples
        means: List of 4 mean vectors
        covariances: List of 4 covariance matrices
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: X1, X2, X3, Class
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate samples per class (balanced)
    samples_per_class = n_samples // 4
    remainder = n_samples % 4

    X_all = []
    y_all = []

    # Generate samples for each class
    for class_idx in range(4):
        # Add extra sample to first classes if remainder exists
        n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)

        # Create distribution and sample
        dist = multivariate_normal(mean=means[class_idx],
                                   cov=covariances[class_idx])
        X_class = dist.rvs(size=n_class_samples)

        # Store samples and labels (classes are 1,2,3,4)
        X_all.append(X_class)
        y_all.extend([class_idx + 1] * n_class_samples)

    # Combine all samples
    X = np.vstack(X_all)
    y = np.array(y_all)

    # Shuffle the data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Create DataFrame
    df = pd.DataFrame({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'X3': X[:, 2],
        'Class': y
    })

    return df


# ============================================================================
# GENERATE ALL DATASETS
# ============================================================================

def generate_all_datasets():
    """
    Generate 5 training datasets and 1 test dataset, save as CSV files
    """

    # Training dataset sizes
    train_sizes = [100, 500, 1000, 5000, 10000]

    print("=" * 60)
    print("GENERATING GAUSSIAN DATASETS FOR CLASSIFICATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - 4 classes with uniform priors (0.25 each)")
    print(f"  - 3D Gaussian distributions")
    print(f"  - Medium overlap (~20% Bayes error)")
    print(f"  - Classes labeled as 1, 2, 3, 4")

    # Generate training datasets
    print("\n" + "-" * 40)
    print("TRAINING DATASETS:")
    print("-" * 40)

    for i, n_samples in enumerate(train_sizes):
        # Use different seed for each dataset
        seed = 42 + i

        # Generate dataset
        df = generate_gaussian_dataset(n_samples, means, covariances, seed=seed)

        # Save to CSV
        filename = f'train_{n_samples}.csv'
        df.to_csv(filename, index=False)

        print(f"\n✓ Generated {filename}")
        print(f"  Samples: {len(df)}")
        print(f"  Class distribution: {df['Class'].value_counts().sort_index().tolist()}")
        print(f"  Features range:")
        print(f"    X1: [{df['X1'].min():.2f}, {df['X1'].max():.2f}]")
        print(f"    X2: [{df['X2'].min():.2f}, {df['X2'].max():.2f}]")
        print(f"    X3: [{df['X3'].min():.2f}, {df['X3'].max():.2f}]")

    # Generate test dataset
    print("\n" + "-" * 40)
    print("TEST DATASET:")
    print("-" * 40)

    test_samples = 100000
    test_seed = 999  # Different seed for test set

    df_test = generate_gaussian_dataset(test_samples, means, covariances, seed=test_seed)

    # Save test set
    filename = 'test_100000.csv'
    df_test.to_csv(filename, index=False)

    print(f"\n✓ Generated {filename}")
    print(f"  Samples: {len(df_test)}")
    print(f"  Class distribution: {df_test['Class'].value_counts().sort_index().tolist()}")
    print(f"  Features statistics:")
    for col in ['X1', 'X2', 'X3']:
        print(f"    {col}: mean={df_test[col].mean():.3f}, std={df_test[col].std():.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  Training: train_100.csv, train_500.csv, train_1000.csv,")
    print("           train_5000.csv, train_10000.csv")
    print("  Testing:  test_100000.csv")
    print("\nEach CSV file contains columns:")
    print("  - X1, X2, X3: Feature values")
    print("  - Class: Class label (1, 2, 3, or 4)")

    return True


# ============================================================================
# VERIFY DATASETS (Optional utility function)
# ============================================================================

def verify_datasets():
    """
    Load and verify the generated CSV files
    """
    print("\n" + "=" * 60)
    print("VERIFYING GENERATED DATASETS")
    print("=" * 60)

    files = ['train_100.csv', 'train_500.csv', 'train_1000.csv',
             'train_5000.csv', 'train_10000.csv', 'test_100000.csv']

    for filename in files:
        try:
            df = pd.read_csv(filename)
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Classes: {sorted(df['Class'].unique())}")
            print(f"  First 3 rows:")
            print(df.head(3).to_string(index=False))
        except FileNotFoundError:
            print(f"\n{filename}: NOT FOUND")

    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate all datasets
    generate_all_datasets()

    # Optionally verify the generated files
    print("\nWould you like to verify the generated datasets?")
    verify_datasets()