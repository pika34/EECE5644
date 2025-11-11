import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations


class GaussianClassDesigner:
    """Design and analyze Gaussian class-conditional PDFs for 4-class problem"""

    def __init__(self, means, covariances, priors=None):
        """
        Initialize with specified parameters

        Args:
            means: List of 4 mean vectors (each 3D)
            covariances: List of 4 covariance matrices (each 3x3)
            priors: Class priors (default: uniform [0.25, 0.25, 0.25, 0.25])
        """
        self.n_classes = 4
        self.n_features = 3
        self.means = np.array(means)
        self.covariances = np.array(covariances)
        self.priors = np.array(priors) if priors else np.ones(4) / 4

        # Create multivariate normal distributions
        self.distributions = [
            multivariate_normal(mean=self.means[i], cov=self.covariances[i])
            for i in range(self.n_classes)
        ]

        # Classes are labeled 1, 2, 3, 4 (not 0-indexed)
        self.class_labels = [1, 2, 3, 4]

    def compute_theoretical_error(self, n_samples=100000):
        """
        Estimate the Bayes error rate using Monte Carlo simulation
        """
        errors = 0
        samples_per_class = n_samples // self.n_classes

        for class_idx in range(self.n_classes):
            true_class = class_idx + 1  # Classes are 1, 2, 3, 4

            # Generate samples from this class
            X = self.distributions[class_idx].rvs(size=samples_per_class)

            # Compute posteriors for all classes
            posteriors = np.zeros((samples_per_class, self.n_classes))
            for c in range(self.n_classes):
                likelihoods = self.distributions[c].pdf(X)
                posteriors[:, c] = likelihoods * self.priors[c]

            # MAP decision (returns indices 0-3, so add 1 for class labels 1-4)
            predicted_classes = np.argmax(posteriors, axis=1) + 1
            errors += np.sum(predicted_classes != true_class)

        error_rate = errors / n_samples
        return error_rate

    def visualize_distributions(self, n_samples=500):
        """
        Visualize the class distributions in 3D
        """
        fig = plt.figure(figsize=(10, 8))

        # 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green', 'purple']

        for i in range(self.n_classes):
            samples = self.distributions[i].rvs(size=n_samples)
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                       c=colors[i], alpha=0.3, label=f'Class {i + 1}', s=20)

        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_zlabel('X3', fontsize=12)
        ax.set_title('3D Gaussian Class Distributions', fontsize=14)
        ax.legend(loc='best', fontsize=10)

        # Add grid for better depth perception
        ax.grid(True, alpha=0.3)

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.show()

    def compute_class_separability(self):
        """
        Compute pairwise Bhattacharyya distances to measure class separability
        """
        n_pairs = 6  # C(4,2) = 6 pairs
        bhatt_distances = {}

        for i, j in combinations(range(self.n_classes), 2):
            # Bhattacharyya distance for Gaussians
            mu_diff = self.means[i] - self.means[j]
            cov_avg = (self.covariances[i] + self.covariances[j]) / 2

            term1 = 0.125 * mu_diff.T @ np.linalg.inv(cov_avg) @ mu_diff
            term2 = 0.5 * np.log(np.linalg.det(cov_avg) /
                                 np.sqrt(np.linalg.det(self.covariances[i]) *
                                         np.linalg.det(self.covariances[j])))

            bhatt_dist = term1 + term2
            bhatt_distances[f'Class {i + 1}-{j + 1}'] = bhatt_dist

        return bhatt_distances

    def generate_dataset(self, n_samples_per_class=2500):
        """
        Generate training and test datasets
        """
        X_train, y_train = [], []
        X_test, y_test = [], []

        # 80-20 train-test split
        n_train = int(0.8 * n_samples_per_class)
        n_test = n_samples_per_class - n_train

        for class_idx in range(self.n_classes):
            class_label = class_idx + 1  # Classes are 1, 2, 3, 4

            # Training data
            X_class_train = self.distributions[class_idx].rvs(size=n_train)
            X_train.append(X_class_train)
            y_train.extend([class_label] * n_train)

            # Test data
            X_class_test = self.distributions[class_idx].rvs(size=n_test)
            X_test.append(X_class_test)
            y_test.extend([class_label] * n_test)

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Shuffle the data
        train_idx = np.random.permutation(len(y_train))
        test_idx = np.random.permutation(len(y_test))

        return (X_train[train_idx], y_train[train_idx],
                X_test[test_idx], y_test[test_idx])

    def map_classifier(self, X):
        """
        Implement the theoretical MAP classifier

        Returns:
            predictions: Class labels (1, 2, 3, or 4)
            posteriors: Posterior probabilities for each class
        """
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.n_classes))

        for c in range(self.n_classes):
            likelihoods = self.distributions[c].pdf(X)
            posteriors[:, c] = likelihoods * self.priors[c]

        # Normalize to get true posteriors
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

        # MAP decision: argmax returns 0-3, so add 1 for class labels 1-4
        predictions = np.argmax(posteriors, axis=1) + 1

        return predictions, posteriors


# ===========================================================================
# PARAMETER CONFIGURATIONS FOR DIFFERENT ERROR RATES
# ===========================================================================

def get_configuration_low_overlap():
    """
    Configuration with ~10-12% error rate (moderate overlap)
    Classes arranged in tetrahedral pattern with controlled overlap
    """
    # Means in tetrahedral arrangement for Classes 1, 2, 3, 4
    means = [
        [0, 0, 0],  # Class 1: origin
        [3.5, 0, 0],  # Class 2: along x-axis
        [1.75, 3.0, 0],  # Class 3: in x-y plane
        [1.75, 1.0, 3.0]  # Class 4: out of plane
    ]

    # Moderate covariances for controlled overlap
    covariances = [
        [[0.8, 0.1, 0.1],
         [0.1, 0.8, 0.1],
         [0.1, 0.1, 0.8]],  # Class 1

        [[0.9, 0.15, 0.1],
         [0.15, 0.8, 0.1],
         [0.1, 0.1, 0.9]],  # Class 2

        [[0.85, 0.1, 0.15],
         [0.1, 0.85, 0.1],
         [0.15, 0.1, 0.8]],  # Class 3

        [[0.8, 0.1, 0.1],
         [0.1, 0.9, 0.15],
         [0.1, 0.15, 0.85]]  # Class 4
    ]

    return means, covariances


def get_configuration_medium_overlap():
    """
    Configuration with ~15-17% error rate (medium overlap)
    Classes with more overlap, elliptical covariances
    """
    # Means closer together for Classes 1, 2, 3, 4
    means = [
        [0, 0, 0],  # Class 1
        [2.8, 0.3, 0],  # Class 2
        [1.4, 2.4, 0.2],  # Class 3
        [1.4, 0.8, 2.5]  # Class 4
    ]

    # Larger, more elliptical covariances
    covariances = [
        [[1.2, 0.3, 0.2],
         [0.3, 1.0, 0.15],
         [0.2, 0.15, 1.1]],  # Class 1

        [[1.5, 0.2, 0.25],
         [0.2, 1.1, 0.2],
         [0.25, 0.2, 1.2]],  # Class 2

        [[1.1, 0.35, 0.2],
         [0.35, 1.4, 0.3],
         [0.2, 0.3, 1.0]],  # Class 3

        [[1.0, 0.25, 0.3],
         [0.25, 1.2, 0.35],
         [0.3, 0.35, 1.3]]  # Class 4
    ]

    return means, covariances


def get_configuration_high_overlap():
    """
    Configuration with ~18-20% error rate (high overlap)
    Classes with significant overlap
    """
    # Means even closer for Classes 1, 2, 3, 4
    means = [
        [0, 0, 0],  # Class 1
        [2.2, 0.5, 0.3],  # Class 2
        [1.0, 2.0, 0.5],  # Class 3
        [1.1, 0.8, 2.0]  # Class 4
    ]

    # Large covariances with significant correlation
    covariances = [
        [[1.5, 0.4, 0.3],
         [0.4, 1.3, 0.35],
         [0.3, 0.35, 1.4]],  # Class 1

        [[1.8, 0.5, 0.4],
         [0.5, 1.5, 0.3],
         [0.4, 0.3, 1.6]],  # Class 2

        [[1.6, 0.45, 0.35],
         [0.45, 1.7, 0.4],
         [0.35, 0.4, 1.3]],  # Class 3

        [[1.4, 0.35, 0.5],
         [0.35, 1.6, 0.45],
         [0.5, 0.45, 1.7]]  # Class 4
    ]

    return means, covariances


# ===========================================================================
# TESTING AND VALIDATION
# ===========================================================================

def test_configuration(config_name, means, covariances):
    """Test a configuration and report statistics"""
    print(f"\n{'=' * 60}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'=' * 60}")

    designer = GaussianClassDesigner(means, covariances)

    # Compute theoretical error
    error_rate = designer.compute_theoretical_error(n_samples=100000)
    print(f"Theoretical MAP Error Rate: {error_rate * 100:.2f}%")

    # Compute separability metrics
    bhatt_distances = designer.compute_class_separability()
    print("\nBhattacharyya Distances (class separability):")
    for pair, dist in bhatt_distances.items():
        print(f"  {pair}: {dist:.3f}")

    # Visualize if needed
    designer.visualize_distributions(n_samples=300)

    return designer


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    # Test all three configurations
    configs = [
        ("Low Overlap (~10-12% error)", *get_configuration_low_overlap()),
        ("Medium Overlap (~15-17% error)", *get_configuration_medium_overlap()),
        ("High Overlap (~18-20% error)", *get_configuration_high_overlap())
    ]

    designers = []
    for config_name, means, covariances in configs:
        designer = test_configuration(config_name, means, covariances)
        designers.append(designer)

    # Select the best configuration (you can modify this)
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("Choose the 'Medium Overlap' configuration for your MLP training.")
    print("It provides a good balance between learnability and challenge.")

    # Generate dataset for the selected configuration
    selected_designer = designers[1]  # Medium overlap
    X_train, y_train, X_test, y_test = selected_designer.generate_dataset()
    print(f"\nGenerated Dataset:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]} dimensions")
    print(f"Classes: {np.unique(y_train).tolist()} (labels 1, 2, 3, 4)")