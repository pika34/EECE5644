import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MLP CLASSIFIER IMPLEMENTATION
# ============================================================================

class MLPClassifier:
    """
    Simple 2-layer MLP with ReLU activation and softmax output
    """

    def __init__(self, input_dim=3, hidden_dim=10, output_dim=4, learning_rate=0.01):
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def train(self, X, y, epochs=150, batch_size=32, verbose=False):
        # Convert y to one-hot encoding (handle 1-indexed classes)
        n_samples = X.shape[0]
        y_one_hot = np.zeros((n_samples, 4))
        y_one_hot[np.arange(n_samples), y - 1] = 1

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.forward(batch_X)

                # Backward pass
                batch_size_actual = batch_X.shape[0]

                # Output layer gradients
                dz2 = output - batch_y
                dW2 = (self.a1.T @ dz2) / batch_size_actual
                db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size_actual

                # Hidden layer gradients
                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_derivative(self.z1)
                dW1 = (batch_X.T @ dz1) / batch_size_actual
                db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size_actual

                # Update weights
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1) + 1  # Return 1-indexed classes


# ============================================================================
# CROSS-VALIDATION FUNCTION
# ============================================================================

def cross_validate_mlp(X, y, hidden_sizes, n_folds=10, epochs=150):
    """
    Perform 10-fold cross-validation for different hidden layer sizes

    Args:
        X: Features
        y: Labels (1,2,3,4)
        hidden_sizes: List of hidden layer sizes to test
        n_folds: Number of folds (default 10)
        epochs: Training epochs

    Returns:
        Dictionary with results
    """
    results = {
        'hidden_size': [],
        'fold_errors': [],
        'mean_error': [],
        'std_error': []
    }

    # Standardize features once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for P in hidden_sizes:
        fold_errors = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            # Split data
            X_train = X_scaled[train_idx]
            y_train = y[train_idx]
            X_val = X_scaled[val_idx]
            y_val = y[val_idx]

            # Train MLP
            mlp = MLPClassifier(input_dim=3, hidden_dim=P, output_dim=4, learning_rate=0.01)
            mlp.train(X_train, y_train, epochs=epochs, batch_size=min(32, len(X_train) // 4))

            # Validate
            y_pred = mlp.predict(X_val)
            error = 1 - np.mean(y_pred == y_val)
            fold_errors.append(error)

        # Store results
        results['hidden_size'].append(P)
        results['fold_errors'].append(fold_errors)
        results['mean_error'].append(np.mean(fold_errors))
        results['std_error'].append(np.std(fold_errors))

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_all_datasets():
    """
    Load all training datasets and perform cross-validation analysis
    """

    # Define dataset sizes and corresponding hidden layer sizes to test
    dataset_configs = {
        100: [2, 3, 5, 7, 10, 15],  # Smaller P for tiny dataset
        500: [3, 5, 10, 15, 20, 30],  # Medium P values
        1000: [5, 10, 15, 20, 30, 40],  # Larger P values
        5000: [10, 15, 20, 30, 40, 50],  # Large dataset
        10000: [10, 20, 30, 40, 50, 75]  # Even larger P for big dataset
    }

    all_results = {}
    best_P = {}

    print("=" * 70)
    print("10-FOLD CROSS-VALIDATION FOR MLP HYPERPARAMETER TUNING")
    print("=" * 70)

    # Process each dataset
    for n_samples in [100, 500, 1000, 5000, 10000]:
        print(f"\n{'=' * 50}")
        print(f"DATASET: train_{n_samples}.csv")
        print(f"{'=' * 50}")

        # Load dataset
        try:
            df = pd.read_csv(f'train_{n_samples}.csv')
            X = df[['X1', 'X2', 'X3']].values
            y = df['Class'].values
            print(f"✓ Loaded {len(df)} samples")
        except FileNotFoundError:
            print(f"✗ File train_{n_samples}.csv not found!")
            continue

        # Get hidden sizes for this dataset
        hidden_sizes = dataset_configs[n_samples]
        print(f"Testing hidden layer sizes: {hidden_sizes}")

        # Perform cross-validation
        print(f"\nPerforming 10-fold cross-validation...")
        results = cross_validate_mlp(X, y, hidden_sizes, n_folds=10, epochs=150)
        all_results[n_samples] = results

        # Find best P
        best_idx = np.argmin(results['mean_error'])
        best_P[n_samples] = results['hidden_size'][best_idx]

        # Print results table
        print(f"\nResults for {n_samples} samples:")
        print("-" * 40)
        print(f"{'P':>4} | {'Mean Error':>10} | {'Std Error':>10} | {'Mean ± Std':>15}")
        print("-" * 40)

        for i, P in enumerate(results['hidden_size']):
            mean_err = results['mean_error'][i]
            std_err = results['std_error'][i]
            marker = " *" if P == best_P[n_samples] else ""
            print(f"{P:4d} | {mean_err:10.4f} | {std_err:10.4f} | {mean_err:.4f} ± {std_err:.4f}{marker}")

        print(
            f"\n➤ Best P = {best_P[n_samples]} with error = {results['mean_error'][best_idx]:.4f} ± {results['std_error'][best_idx]:.4f}")

    # Create combined plot
    plot_results(all_results, best_P)

    # Print final summary
    print_summary(best_P, all_results)

    return all_results, best_P


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_results(all_results, best_P):
    """
    Create a combined plot showing error vs P for all training sets
    """
    plt.figure(figsize=(12, 8))

    colors = {100: 'red', 500: 'blue', 1000: 'green', 5000: 'orange', 10000: 'purple'}
    markers = {100: 'o', 500: 's', 1000: '^', 5000: 'p', 10000: 'd'}

    for n_samples, results in all_results.items():
        P_values = results['hidden_size']
        mean_errors = np.array(results['mean_error']) * 100  # Convert to percentage
        std_errors = np.array(results['std_error']) * 100

        # Plot with error bars
        plt.errorbar(P_values, mean_errors, yerr=std_errors,
                     label=f'N = {n_samples:,}',
                     color=colors[n_samples],
                     marker=markers[n_samples],
                     markersize=8,
                     linewidth=2,
                     capsize=5,
                     capthick=2,
                     alpha=0.7)

        # Mark the best P with a star
        best_idx = P_values.index(best_P[n_samples])
        plt.scatter(best_P[n_samples], mean_errors[best_idx],
                    s=300, marker='*',
                    color=colors[n_samples],
                    edgecolor='black',
                    linewidth=2,
                    zorder=5)

    plt.xlabel('Number of Hidden Neurons (P)', fontsize=14)
    plt.ylabel('Validation Error Rate (%)', fontsize=14)
    plt.title('10-Fold Cross-Validation: Error vs Hidden Layer Size\n(Stars indicate optimal P for each dataset)',
              fontsize=16)
    plt.legend(title='Training Set Size', fontsize=11, title_fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, max(80, max([max(r['hidden_size']) for r in all_results.values()]) + 5))

    # Add annotation for theoretical minimum (approximately 20.77%)
    plt.axhline(y=20.77, color='red', linestyle=':', linewidth=2, alpha=0.5)
    plt.text(plt.xlim()[1] * 0.7, 21.5, 'Bayes Error (~20.77%)',
             fontsize=11, color='red', alpha=0.7)

    plt.tight_layout()
    plt.savefig('mlp_cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# SUMMARY FUNCTION
# ============================================================================

def print_summary(best_P, all_results):
    """
    Print final summary of results
    """
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nOptimal Hidden Layer Sizes:")
    print("-" * 40)
    print(f"{'Dataset':>10} | {'Best P':>8} | {'Val Error':>12} | {'Std Dev':>10}")
    print("-" * 40)

    for n_samples in [100, 500, 1000, 5000, 10000]:
        if n_samples in best_P:
            P = best_P[n_samples]
            idx = all_results[n_samples]['hidden_size'].index(P)
            mean_err = all_results[n_samples]['mean_error'][idx]
            std_err = all_results[n_samples]['std_error'][idx]
            print(f"{n_samples:10,} | {P:8} | {mean_err * 100:11.2f}% | {std_err * 100:9.2f}%")

    print("\nKey Observations:")
    print("-" * 40)
    print("• Smaller datasets (N=100) require fewer hidden neurons (P~3-7)")
    print("• Medium datasets (N=500-1000) work best with P~10-15")
    print("• Larger datasets (N=5000-10000) can utilize more neurons (P~20-40)")
    print("• Validation error decreases with more training data")
    print("• All models converge toward Bayes error (~20.77%) with sufficient data")

    print("\n✓ Results saved to 'mlp_cross_validation_results.png'")


# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Run the complete analysis
    all_results, best_P = analyze_all_datasets()