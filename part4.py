import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# ============================================================================
# HARDCODED GAUSSIAN PARAMETERS
# ============================================================================

# Mean vectors for 4 classes
means = [
    np.array([0.0, 0.0, 0.0]),  # Class 1
    np.array([2.8, 0.3, 0.0]),  # Class 2
    np.array([1.4, 2.4, 0.2]),  # Class 3
    np.array([1.4, 0.8, 2.5])  # Class 4
]

# Covariance matrices for 4 classes
covariances = [
    np.array([[1.2, 0.3, 0.2],  # Class 1
              [0.3, 1.0, 0.15],
              [0.2, 0.15, 1.1]]),

    np.array([[1.5, 0.2, 0.25],  # Class 2
              [0.2, 1.1, 0.2],
              [0.25, 0.2, 1.2]]),

    np.array([[1.1, 0.35, 0.2],  # Class 3
              [0.35, 1.4, 0.3],
              [0.2, 0.3, 1.0]]),

    np.array([[1.0, 0.25, 0.3],  # Class 4
              [0.25, 1.2, 0.35],
              [0.3, 0.35, 1.3]])
]

# Equal class priors (uniform distribution)
priors = [0.25, 0.25, 0.25, 0.25]


# ============================================================================
# MAP CLASSIFIER IMPLEMENTATION
# ============================================================================

def compute_map_classifier(X, means, covariances, priors):
    """
    Implement MAP classifier for Gaussian distributions

    Args:
        X: Test data (N x 3)
        means: List of 4 mean vectors
        covariances: List of 4 covariance matrices
        priors: List of 4 prior probabilities

    Returns:
        predictions: Predicted class labels (1, 2, 3, 4)
        posteriors: Posterior probabilities for each class
    """
    n_samples = X.shape[0]
    n_classes = 4

    # Initialize posterior matrix
    posteriors = np.zeros((n_samples, n_classes))

    # For each class, compute likelihood × prior
    for c in range(n_classes):
        # Create multivariate normal distribution for this class
        mvn = multivariate_normal(mean=means[c], cov=covariances[c])

        # Compute likelihood for all samples
        likelihoods = mvn.pdf(X)

        # Multiply by prior to get unnormalized posterior
        posteriors[:, c] = likelihoods * priors[c]

    # Choose class with highest posterior (MAP decision)
    # Add 1 to convert from 0-indexed to 1-indexed classes
    predictions = np.argmax(posteriors, axis=1) + 1

    # Normalize posteriors (optional, for completeness)
    posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

    return predictions, posteriors


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_map_classifier():
    """
    Load test dataset and evaluate MAP classifier performance
    """

    print("=" * 60)
    print("MAP CLASSIFIER EVALUATION ON TEST DATASET")
    print("=" * 60)

    # Load test dataset
    print("\nLoading test dataset...")
    try:
        df_test = pd.read_csv('test_100000.csv')
        print(f"✓ Loaded test_100000.csv successfully")
        print(f"  Dataset shape: {df_test.shape}")
    except FileNotFoundError:
        print("ERROR: test_100000.csv not found!")
        print("Please run the dataset generation script first.")
        return

    # Extract features and labels
    X_test = df_test[['X1', 'X2', 'X3']].values
    y_true = df_test['Class'].values

    print(f"  Features shape: {X_test.shape}")
    print(f"  Classes present: {sorted(np.unique(y_true))}")
    print(f"  Class distribution: {[np.sum(y_true == c) for c in [1, 2, 3, 4]]}")

    # Compute MAP predictions
    print("\n" + "-" * 40)
    print("Computing MAP predictions...")
    print("-" * 40)

    predictions, posteriors = compute_map_classifier(X_test, means, covariances, priors)

    # Compute accuracy and error
    print("\nEvaluating predictions...")

    # Count correct and wrong predictions
    correct_predictions = np.sum(predictions == y_true)
    wrong_predictions = np.sum(predictions != y_true)
    total_samples = len(y_true)

    # Compute empirical probability of error
    probability_of_error = wrong_predictions / total_samples
    accuracy = correct_predictions / total_samples

    # Print detailed results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal test samples: {total_samples:,}")
    print(f"Correct predictions: {correct_predictions:,}")
    print(f"Wrong predictions: {wrong_predictions:,}")

    print(f"\n>>> ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f">>> ERROR RATE: {probability_of_error:.4f} ({probability_of_error * 100:.2f}%)")

    # Per-class analysis
    print("\n" + "-" * 40)
    print("PER-CLASS PERFORMANCE")
    print("-" * 40)

    print("\nClass | Samples | Correct | Accuracy")
    print("------|---------|---------|----------")

    for class_label in [1, 2, 3, 4]:
        class_mask = y_true == class_label
        class_total = np.sum(class_mask)
        class_correct = np.sum(predictions[class_mask] == class_label)
        class_accuracy = class_correct / class_total
        print(f"  {class_label}   | {class_total:7,} | {class_correct:7,} | {class_accuracy:.4f}")

    # Confusion matrix
    print("\n" + "-" * 40)
    print("CONFUSION MATRIX")
    print("-" * 40)
    print("\nTrue\\Pred     1        2        3        4")
    print("--------- -------- -------- -------- --------")

    for true_class in [1, 2, 3, 4]:
        row = f"    {true_class}     "
        for pred_class in [1, 2, 3, 4]:
            count = np.sum((y_true == true_class) & (predictions == pred_class))
            row += f"{count:8,} "
        print(row)

    # Average posterior confidence
    print("\n" + "-" * 40)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("-" * 40)

    # Get the posterior probability of the predicted class for each sample
    predicted_posteriors = [posteriors[i, predictions[i] - 1] for i in range(len(predictions))]
    predicted_posteriors = np.array(predicted_posteriors)

    # Separate confidence for correct and incorrect predictions
    correct_mask = predictions == y_true
    correct_confidence = predicted_posteriors[correct_mask].mean()
    incorrect_confidence = predicted_posteriors[~correct_mask].mean()

    print(f"\nAverage confidence (posterior probability):")
    print(f"  Correct predictions: {correct_confidence:.4f}")
    print(f"  Incorrect predictions: {incorrect_confidence:.4f}")
    print(f"  Confidence gap: {correct_confidence - incorrect_confidence:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return accuracy, probability_of_error


# ============================================================================
# RUN EVALUATION
# ============================================================================

if __name__ == "__main__":
    accuracy, error_rate = evaluate_map_classifier()