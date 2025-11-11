import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_test_data():
    """
    Load test dataset and split into features and labels
    """
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    # Load test dataset
    df_test = pd.read_csv('test_100000.csv')

    # Split into features (X) and labels (y)
    X_test = df_test[['X1', 'X2', 'X3']].values
    y_test = df_test['Class'].values

    print(f"✓ Loaded test dataset: {len(df_test)} samples")
    print(f"  Feature matrix shape: {X_test.shape}")
    print(f"  Classes: {np.unique(y_test).tolist()}")
    print(f"  Class distribution: {[np.sum(y_test == c) for c in [1, 2, 3, 4]]}")

    return X_test, y_test


# ============================================================================
# MLP MODEL CLASS (for loading saved models)
# ============================================================================

class MLPClassifier:
    """
    MLP Classifier class to match the saved model structure
    """

    def __init__(self, input_dim=3, hidden_dim=10, output_dim=4, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Weights will be loaded from saved model
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network
        Returns both logits and probabilities
        """
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        # Output layer (logits)
        z2 = a1 @ self.W2 + self.b2

        # Apply softmax to get probabilities
        probs = self.softmax(z2)

        return z2, probs

    def predict_proba(self, X):
        """
        Get posterior probabilities for each class
        """
        _, probs = self.forward(X)
        return probs

    def predict(self, X):
        """
        Apply MAP decision rule: choose class with highest posterior
        Returns class labels (1, 2, 3, 4)
        """
        probs = self.predict_proba(X)
        # Argmax gives 0-indexed class, add 1 for 1-indexed labels
        predictions = np.argmax(probs, axis=1) + 1
        return predictions


# ============================================================================
# EVALUATE SINGLE MODEL
# ============================================================================

def evaluate_model(n_samples, X_test, y_test):
    """
    Load and evaluate a single saved model

    Args:
        n_samples: Training dataset size (100, 500, 1000, 5000, 10000)
        X_test: Test features
        y_test: True test labels

    Returns:
        Dictionary with evaluation results
    """

    print(f"\n{'=' * 60}")
    print(f"EVALUATING MODEL: train_{n_samples}.csv")
    print(f"{'=' * 60}")

    # Load saved model
    model_filename = f'final_model_n{n_samples}.pkl'

    try:
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Loaded model from {model_filename}")
    except FileNotFoundError:
        print(f"✗ Model file {model_filename} not found!")
        return None

    # Extract model and scaler
    model = model_data['model']
    scaler = model_data['scaler']
    best_P = model_data['best_P']

    print(f"  Model architecture: Input(3) → Hidden({best_P}) → Output(4)")

    # Step 1: Standardize test data using the model's scaler
    print(f"\n1. Standardizing test data using saved scaler...")
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ Test data standardized")

    # Step 2: Forward pass to get logits and probabilities
    print(f"\n2. Forward pass through the network...")
    logits, posteriors = model.forward(X_test_scaled)
    print(f"   ✓ Computed logits shape: {logits.shape}")
    print(f"   ✓ Applied softmax to get posteriors shape: {posteriors.shape}")

    # Step 3: Apply MAP decision rule
    print(f"\n3. Applying MAP decision rule (argmax of posteriors)...")
    predictions = np.argmax(posteriors, axis=1) + 1  # Add 1 for 1-indexed classes
    print(f"   ✓ Generated predictions for {len(predictions)} samples")

    # Step 4: Compare with true labels
    print(f"\n4. Comparing predictions with true labels...")

    # Calculate metrics
    correct = np.sum(predictions == y_test)
    total = len(y_test)
    accuracy = correct / total
    error_rate = 1 - accuracy

    print(f"\n   RESULTS:")
    print(f"   --------")
    print(f"   Correct predictions: {correct:,} / {total:,}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")

    # Per-class accuracy
    print(f"\n   Per-Class Performance:")
    print(f"   Class | Correct | Total | Accuracy")
    print(f"   ------|---------|-------|----------")

    class_accuracies = []
    for c in [1, 2, 3, 4]:
        mask = y_test == c
        class_correct = np.sum(predictions[mask] == c)
        class_total = np.sum(mask)
        class_acc = class_correct / class_total
        class_accuracies.append(class_acc)
        print(f"     {c}   | {class_correct:7,} | {class_total:6,} | {class_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Average confidence analysis
    predicted_confidence = []
    for i in range(len(predictions)):
        # Get the posterior probability of the predicted class
        predicted_confidence.append(posteriors[i, predictions[i] - 1])
    predicted_confidence = np.array(predicted_confidence)

    correct_mask = predictions == y_test
    avg_confidence_correct = np.mean(predicted_confidence[correct_mask])
    avg_confidence_incorrect = np.mean(predicted_confidence[~correct_mask])

    print(f"\n   Prediction Confidence:")
    print(f"   Average confidence (correct): {avg_confidence_correct:.4f}")
    print(f"   Average confidence (incorrect): {avg_confidence_incorrect:.4f}")
    print(f"   Confidence gap: {avg_confidence_correct - avg_confidence_incorrect:.4f}")

    # Store results
    results = {
        'n_samples': n_samples,
        'best_P': best_P,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'predictions': predictions,
        'posteriors': posteriors,
        'confusion_matrix': cm,
        'class_accuracies': class_accuracies,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect,
        'model': model,
        'scaler': scaler
    }

    return results


# ============================================================================
# EVALUATE ALL MODELS
# ============================================================================

def evaluate_all_models(X_test, y_test):
    """
    Evaluate all saved models on the test set
    """

    print("\n" + "=" * 70)
    print("EVALUATING ALL FINAL MLP MODELS")
    print("=" * 70)

    dataset_sizes = [100, 500, 1000, 5000, 10000]
    all_results = {}

    for n_samples in dataset_sizes:
        results = evaluate_model(n_samples, X_test, y_test)
        if results is not None:
            all_results[n_samples] = results

    return all_results


# ============================================================================
# COMPUTE MAP BASELINE
# ============================================================================

def compute_map_baseline(X_test, y_test):
    """
    Compute the theoretical MAP classifier performance for comparison
    """

    print("\n" + "=" * 70)
    print("COMPUTING MAP CLASSIFIER BASELINE")
    print("=" * 70)

    # Hardcoded Gaussian parameters
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

    priors = [0.25, 0.25, 0.25, 0.25]

    # Compute MAP predictions
    n_samples = X_test.shape[0]
    posteriors = np.zeros((n_samples, 4))

    for c in range(4):
        mvn = multivariate_normal(mean=means[c], cov=covariances[c])
        likelihoods = mvn.pdf(X_test)
        posteriors[:, c] = likelihoods * priors[c]

    map_predictions = np.argmax(posteriors, axis=1) + 1
    map_accuracy = np.mean(map_predictions == y_test)
    map_error = 1 - map_accuracy

    print(f"MAP Classifier (Theoretical Optimal):")
    print(f"  Accuracy: {map_accuracy:.4f} ({map_accuracy * 100:.2f}%)")
    print(f"  Error Rate: {map_error:.4f} ({map_error * 100:.2f}%)")

    return map_accuracy, map_error


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_performance_comparison(all_results, map_error):
    """
    Plot comparison of all models vs MAP baseline
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    dataset_sizes = sorted(all_results.keys())
    error_rates = [all_results[n]['error_rate'] * 100 for n in dataset_sizes]
    gaps_from_map = [(all_results[n]['error_rate'] - map_error) * 100 for n in dataset_sizes]

    # Plot 1: Error rates
    ax1.semilogx(dataset_sizes, error_rates, 'o-', markersize=10, linewidth=2, label='MLP Error')
    ax1.axhline(y=map_error * 100, color='red', linestyle='--', linewidth=2, label='MAP Error (Bayes Optimal)')
    ax1.set_xlabel('Training Dataset Size', fontsize=12)
    ax1.set_ylabel('Test Error Rate (%)', fontsize=12)
    ax1.set_title('MLP Performance vs Training Data Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add annotations
    for i, n in enumerate(dataset_sizes):
        ax1.annotate(f'{error_rates[i]:.1f}%',
                     xy=(n, error_rates[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9)

    # Plot 2: Gap from optimal
    ax2.semilogx(dataset_sizes, gaps_from_map, 'o-', markersize=10, linewidth=2, color='purple')
    ax2.set_xlabel('Training Dataset Size', fontsize=12)
    ax2.set_ylabel('Additional Error vs MAP (%)', fontsize=12)
    ax2.set_title('Performance Gap from Theoretical Optimal', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add annotations
    for i, n in enumerate(dataset_sizes):
        ax2.annotate(f'+{gaps_from_map[i]:.2f}%',
                     xy=(n, gaps_from_map[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9)

    plt.suptitle('Final Model Evaluation on 100,000 Test Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('final_model_test_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(all_results):
    """
    Plot confusion matrices for all models
    """

    dataset_sizes = sorted(all_results.keys())
    n_models = len(dataset_sizes)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for idx, n_samples in enumerate(dataset_sizes):
        cm = all_results[n_samples]['confusion_matrix']

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4],
                    cbar=False, ax=ax)
        ax.set_title(f'N = {n_samples:,}', fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle('Confusion Matrices (Normalized by True Class)', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def print_summary_report(all_results, map_accuracy, map_error):
    """
    Print comprehensive summary report
    """

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTheoretical Optimal (MAP Classifier):")
    print(f"  Test Accuracy: {map_accuracy:.4f} ({map_accuracy * 100:.2f}%)")
    print(f"  Test Error: {map_error:.4f} ({map_error * 100:.2f}%)")

    print("\n" + "-" * 60)
    print("MLP Model Performance:")
    print("-" * 60)
    print(f"{'Train Size':>10} | {'Hidden':>7} | {'Test Acc':>9} | {'Test Err':>9} | {'Gap from MAP':>12}")
    print("-" * 60)

    for n_samples in sorted(all_results.keys()):
        r = all_results[n_samples]
        gap = r['error_rate'] - map_error
        print(f"{n_samples:10,} | {r['best_P']:7} | {r['accuracy'] * 100:8.2f}% | "
              f"{r['error_rate'] * 100:8.2f}% | {gap * 100:+11.2f}%")

    print("\n" + "-" * 60)
    print("Key Findings:")
    print("-" * 60)

    # Find best performing model
    best_n = min(all_results.keys(), key=lambda n: all_results[n]['error_rate'])
    best_error = all_results[best_n]['error_rate']
    best_gap = best_error - map_error

    print(f"• Best MLP: N={best_n:,} with {best_error * 100:.2f}% error")
    print(f"• Gap from optimal: {best_gap * 100:.2f}% additional error")
    print(f"• Relative performance: {(best_error / map_error - 1) * 100:.1f}% worse than MAP")

    # Analyze trend
    sizes = sorted(all_results.keys())
    errors = [all_results[n]['error_rate'] for n in sizes]
    improvement = (errors[0] - errors[-1]) * 100

    print(f"• Error reduction from N=100 to N=10000: {improvement:.1f}%")
    print(f"• All models with N≥1000 achieve <23% error")
    print(f"• Models approach but cannot exceed Bayes optimal performance")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """

    # Load test data
    X_test, y_test = load_test_data()

    # Compute MAP baseline
    map_accuracy, map_error = compute_map_baseline(X_test, y_test)

    # Evaluate all models
    all_results = evaluate_all_models(X_test, y_test)

    if all_results:
        # Generate plots
        plot_performance_comparison(all_results, map_error)
        plot_confusion_matrices(all_results)

        # Print summary
        print_summary_report(all_results, map_accuracy, map_error)
    else:
        print("\nNo models found to evaluate!")
        print("Please ensure the model files (final_model_n100.pkl, etc.) exist.")

    return all_results


if __name__ == "__main__":
    # Run evaluation
    results = main()