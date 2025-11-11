import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# ============================================================================
# STEP 1: LOAD TEST DATASET
# ============================================================================

def load_test_dataset():
    """
    Load the test dataset from CSV file
    """
    print("=" * 60)
    print("STEP 1: LOADING TEST DATASET")
    print("=" * 60)

    # Load test data
    df_test = pd.read_csv('test_100000.csv')
    X_test = df_test[['X1', 'X2', 'X3']].values
    y_test = df_test['Class'].values

    print(f"✓ Loaded {len(df_test):,} test samples")
    print(f"  Features shape: {X_test.shape}")
    print(f"  Classes: {np.unique(y_test)}")
    print(f"  Class distribution: {[np.sum(y_test == c) for c in [1, 2, 3, 4]]}")

    return X_test, y_test


# ============================================================================
# STEP 2: COMPUTE BAYES THEORETICAL ERROR
# ============================================================================

def compute_bayes_error(X_test, y_test):
    """
    Compute the theoretical Bayes error using true Gaussian parameters
    """
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING BAYES THEORETICAL ERROR")
    print("=" * 60)

    # True Gaussian parameters (hardcoded)
    means = [
        np.array([0.0, 0.0, 0.0]),  # Class 1
        np.array([2.8, 0.3, 0.0]),  # Class 2
        np.array([1.4, 2.4, 0.2]),  # Class 3
        np.array([1.4, 0.8, 2.5])  # Class 4
    ]

    covariances = [
        np.array([[1.2, 0.3, 0.2],
                  [0.3, 1.0, 0.15],
                  [0.2, 0.15, 1.1]]),

        np.array([[1.5, 0.2, 0.25],
                  [0.2, 1.1, 0.2],
                  [0.25, 0.2, 1.2]]),

        np.array([[1.1, 0.35, 0.2],
                  [0.35, 1.4, 0.3],
                  [0.2, 0.3, 1.0]]),

        np.array([[1.0, 0.25, 0.3],
                  [0.25, 1.2, 0.35],
                  [0.3, 0.35, 1.3]])
    ]

    # Uniform priors
    priors = [0.25, 0.25, 0.25, 0.25]

    # Compute MAP predictions
    print("Computing MAP predictions using true distributions...")
    n_samples = X_test.shape[0]
    posteriors = np.zeros((n_samples, 4))

    for c in range(4):
        mvn = multivariate_normal(mean=means[c], cov=covariances[c])
        likelihoods = mvn.pdf(X_test)
        posteriors[:, c] = likelihoods * priors[c]

    # MAP decision rule
    map_predictions = np.argmax(posteriors, axis=1) + 1

    # Calculate error
    correct = np.sum(map_predictions == y_test)
    bayes_error = 1 - (correct / n_samples)
    bayes_accuracy = correct / n_samples

    print(f"\n✓ Bayes (MAP) Classifier Results:")
    print(f"  Correct predictions: {correct:,} / {n_samples:,}")
    print(f"  Accuracy: {bayes_accuracy:.4f} ({bayes_accuracy * 100:.2f}%)")
    print(f"  >>> BAYES ERROR: {bayes_error:.4f} ({bayes_error * 100:.2f}%)")

    return bayes_error, bayes_accuracy


# ============================================================================
# STEP 3: LOAD AND EVALUATE ALL TRAINED MLPs
# ============================================================================

class MLPClassifier:
    """Simple MLP class for loading saved models"""

    def __init__(self):
        pass

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self.softmax(z2)
        return probs

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1) + 1


def evaluate_mlp_models(X_test, y_test):
    """
    Load and evaluate all trained MLP models
    """
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING TRAINED MLP MODELS")
    print("=" * 60)

    dataset_sizes = [100, 500, 1000, 5000, 10000]
    mlp_errors = []
    mlp_accuracies = []

    for n_samples in dataset_sizes:
        print(f"\n--- Evaluating model trained on {n_samples} samples ---")

        # Load saved model
        model_filename = f'final_model_n{n_samples}.pkl'

        try:
            with open(model_filename, 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            scaler = model_data['scaler']
            best_P = model_data['best_P']

            print(f"  ✓ Loaded model with P={best_P} hidden neurons")

            # Standardize test data
            X_test_scaled = scaler.transform(X_test)

            # Make predictions
            predictions = model.predict(X_test_scaled)

            # Calculate error
            correct = np.sum(predictions == y_test)
            accuracy = correct / len(y_test)
            error = 1 - accuracy

            mlp_errors.append(error)
            mlp_accuracies.append(accuracy)

            print(f"  Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"  Test error: {error:.4f} ({error * 100:.2f}%)")

        except FileNotFoundError:
            print(f"  ✗ Model file {model_filename} not found!")
            mlp_errors.append(None)
            mlp_accuracies.append(None)

    return dataset_sizes, mlp_errors, mlp_accuracies


# ============================================================================
# STEP 4: CREATE COMPARISON PLOT
# ============================================================================

def plot_mlp_vs_bayes(dataset_sizes, mlp_errors, bayes_error):
    """
    Plot MLP error vs training size with Bayes error line
    """
    print("\n" + "=" * 60)
    print("STEP 4: CREATING COMPARISON PLOT")
    print("=" * 60)

    # Filter out None values
    valid_data = [(n, e) for n, e in zip(dataset_sizes, mlp_errors) if e is not None]
    if not valid_data:
        print("No valid model data to plot!")
        return

    sizes, errors = zip(*valid_data)
    errors_percent = [e * 100 for e in errors]
    bayes_percent = bayes_error * 100

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot MLP errors
    plt.semilogx(sizes, errors_percent, 'o-',
                 markersize=10, linewidth=2.5,
                 color='blue', label='MLP Test Error',
                 markeredgecolor='darkblue', markeredgewidth=1.5)

    # Plot Bayes error line
    plt.axhline(y=bayes_percent, color='red', linestyle='--',
                linewidth=2.5, label=f'Bayes Error ({bayes_percent:.2f}%)')

    # Add value annotations
    for size, error_pct in zip(sizes, errors_percent):
        plt.annotate(f'{error_pct:.2f}%',
                     xy=(size, error_pct),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='blue',
                               alpha=0.8))

    # Formatting
    plt.xlabel('Training Dataset Size (log scale)', fontsize=14)
    plt.ylabel('Test Error Rate (%)', fontsize=14)
    plt.title('MLP Performance vs Training Size\nComparison with Bayes Optimal Error',
              fontsize=16, pad=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(80, 15000)

    # Set y-axis limits with some padding
    min_error = min(bayes_percent, min(errors_percent))
    max_error = max(errors_percent)
    plt.ylim(min_error - 1, max_error + 2)

    # Add shaded region showing "impossible zone" below Bayes error
    plt.fill_between([80, 15000], 0, bayes_percent,
                     color='red', alpha=0.1,
                     label='Impossible Region')

    # Add text annotations
    plt.text(150, bayes_percent - 1.5,
             'No classifier can achieve error below this line',
             fontsize=10, color='red', style='italic')

    # Calculate and display gap for largest dataset
    if 10000 in sizes:
        idx = sizes.index(10000)
        gap = errors_percent[idx] - bayes_percent
        plt.annotate(f'Gap: {gap:.2f}%',
                     xy=(10000, errors_percent[idx]),
                     xytext=(10000, (errors_percent[idx] + bayes_percent) / 2),
                     ha='center',
                     fontsize=10,
                     color='purple',
                     weight='bold',
                     arrowprops=dict(arrowstyle='<->',
                                     color='purple',
                                     lw=1.5))

    plt.tight_layout()

    # Save the figure
    plt.savefig('mlp_vs_bayes_error.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved as 'mlp_vs_bayes_error.png'")

    plt.show()


# ============================================================================
# STEP 5: PRINT SUMMARY TABLE
# ============================================================================

def print_summary_table(dataset_sizes, mlp_errors, bayes_error):
    """
    Print a summary table of all results
    """
    print("\n" + "=" * 60)
    print("FINAL SUMMARY TABLE")
    print("=" * 60)

    print(f"\nBayes Optimal Error: {bayes_error:.4f} ({bayes_error * 100:.2f}%)")
    print("\n" + "-" * 50)
    print(f"{'Training Size':>12} | {'MLP Error':>10} | {'Gap from Bayes':>14}")
    print("-" * 50)

    for size, error in zip(dataset_sizes, mlp_errors):
        if error is not None:
            gap = error - bayes_error
            print(f"{size:12,} | {error * 100:9.2f}% | {gap * 100:+13.2f}%")
        else:
            print(f"{size:12,} |       N/A  |           N/A")

    print("-" * 50)

    # Calculate trend
    valid_errors = [e for e in mlp_errors if e is not None]
    if len(valid_errors) >= 2:
        improvement = (valid_errors[0] - valid_errors[-1]) * 100
        print(f"\nTotal error reduction: {improvement:.2f}%")

        if mlp_errors[-1] is not None:
            final_gap = (mlp_errors[-1] - bayes_error) * 100
            relative_performance = (mlp_errors[-1] / bayes_error - 1) * 100
            print(f"Final gap from Bayes optimal: {final_gap:.2f}%")
            print(f"Relative performance: {relative_performance:.1f}% worse than optimal")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "=" * 70)
    print(" MLP vs BAYES ERROR COMPARISON ANALYSIS")
    print("=" * 70)

    # Step 1: Load test data
    X_test, y_test = load_test_dataset()

    # Step 2: Compute Bayes error
    bayes_error, bayes_accuracy = compute_bayes_error(X_test, y_test)

    # Step 3: Evaluate MLP models
    dataset_sizes, mlp_errors, mlp_accuracies = evaluate_mlp_models(X_test, y_test)

    # Step 4: Create plot
    plot_mlp_vs_bayes(dataset_sizes, mlp_errors, bayes_error)

    # Step 5: Print summary
    print_summary_table(dataset_sizes, mlp_errors, bayes_error)

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'bayes_error': bayes_error,
        'dataset_sizes': dataset_sizes,
        'mlp_errors': mlp_errors
    }


if __name__ == "__main__":
    results = main()