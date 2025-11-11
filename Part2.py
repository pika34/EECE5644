import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# MLP IMPLEMENTATION
# ============================================================================

class MLPClassifier:
    """
    2-Layer MLP (1 hidden layer + softmax output layer)
    Uses ReLU activation in hidden layer and softmax in output layer
    Trained with cross-entropy loss
    """

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        """
        Initialize MLP with Xavier/He initialization

        Args:
            input_dim: Number of input features (3 for our problem)
            hidden_dim: Number of hidden neurons (P)
            output_dim: Number of classes (4 for our problem)
            learning_rate: Learning rate for gradient descent
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Initialize weights using He initialization for ReLU
        # W1: (input_dim x hidden_dim), b1: (hidden_dim,)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        # W2: (hidden_dim x output_dim), b2: (output_dim,)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # For storing training history
        self.train_losses = []
        self.val_losses = []

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation function (numerically stable)"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network

        Args:
            X: Input data (N x input_dim)

        Returns:
            Output probabilities (N x output_dim)
        """
        # Hidden layer: ReLU(XW1 + b1)
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Output layer: Softmax(a1W2 + b2)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss

        Args:
            y_pred: Predicted probabilities (N x output_dim)
            y_true: One-hot encoded true labels (N x output_dim)

        Returns:
            Average cross-entropy loss
        """
        # Avoid log(0) by clipping
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Cross-entropy: -sum(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def backward(self, X, y_true, y_pred):
        """
        Backward pass (backpropagation)

        Args:
            X: Input data (N x input_dim)
            y_true: One-hot encoded true labels (N x output_dim)
            y_pred: Predicted probabilities (N x output_dim)
        """
        N = X.shape[0]

        # Output layer gradients
        # For softmax + cross-entropy, the gradient simplifies to (y_pred - y_true)
        dz2 = y_pred - y_true
        dW2 = (self.a1.T @ dz2) / N
        db2 = np.sum(dz2, axis=0, keepdims=True) / N

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / N
        db1 = np.sum(dz1, axis=0, keepdims=True) / N

        # Update weights using gradient descent
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=200, batch_size=32, verbose=True):
        """
        Train the MLP using mini-batch gradient descent

        Args:
            X_train: Training data (N x input_dim)
            y_train: Training labels (N,) - will be one-hot encoded
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print progress
        """
        # Convert labels to one-hot encoding if needed
        if len(y_train.shape) == 1:
            y_train_oh = self.one_hot_encode(y_train)
        else:
            y_train_oh = y_train

        if y_val is not None and len(y_val.shape) == 1:
            y_val_oh = self.one_hot_encode(y_val)
        else:
            y_val_oh = y_val

        N = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            # Mini-batch training
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, y_pred)

            # Calculate losses
            train_pred = self.forward(X_train)
            train_loss = self.cross_entropy_loss(train_pred, y_train_oh)
            self.train_losses.append(train_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(val_pred, y_val_oh)
                self.val_losses.append(val_loss)

                if verbose and epoch % 20 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            elif verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input data (N x input_dim)

        Returns:
            Predicted class labels (N,) - using 1,2,3,4 labeling
        """
        probs = self.forward(X)
        # Add 1 to convert from 0-indexed to 1-indexed classes
        return np.argmax(probs, axis=1) + 1

    def predict_proba(self, X):
        """
        Get prediction probabilities

        Args:
            X: Input data (N x input_dim)

        Returns:
            Class probabilities (N x output_dim)
        """
        return self.forward(X)

    def one_hot_encode(self, y):
        """
        Convert class labels to one-hot encoding
        Handles labels 1,2,3,4 by converting to 0,1,2,3 internally
        """
        y_shifted = y - 1  # Convert 1,2,3,4 to 0,1,2,3
        n_classes = self.output_dim
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y_shifted] = 1
        return one_hot


# ============================================================================
# CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# ============================================================================

def cross_validate_mlp(X, y, hidden_dims, k_folds=5, epochs=200, verbose=True):
    """
    Perform k-fold cross-validation to find best number of hidden neurons

    Args:
        X: Input data
        y: Labels (1,2,3,4)
        hidden_dims: List of hidden dimensions to try
        k_folds: Number of CV folds
        epochs: Training epochs per fold

    Returns:
        Dictionary with CV results
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_results = {
        'hidden_dim': [],
        'mean_train_acc': [],
        'std_train_acc': [],
        'mean_val_acc': [],
        'std_val_acc': [],
        'mean_val_loss': [],
        'std_val_loss': []
    }

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for hidden_dim in tqdm(hidden_dims, desc="Testing hidden dimensions"):
        train_accs = []
        val_accs = []
        val_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            # Split data
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]

            # Train MLP
            mlp = MLPClassifier(
                input_dim=3,
                hidden_dim=hidden_dim,
                output_dim=4,
                learning_rate=0.01
            )

            mlp.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                epochs=epochs,
                batch_size=32,
                verbose=False
            )

            # Evaluate
            train_pred = mlp.predict(X_train_fold)
            val_pred = mlp.predict(X_val_fold)

            train_acc = np.mean(train_pred == y_train_fold)
            val_acc = np.mean(val_pred == y_val_fold)

            # Get validation loss
            val_probs = mlp.predict_proba(X_val_fold)
            y_val_oh = mlp.one_hot_encode(y_val_fold)
            val_loss = mlp.cross_entropy_loss(val_probs, y_val_oh)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

        # Store results
        cv_results['hidden_dim'].append(hidden_dim)
        cv_results['mean_train_acc'].append(np.mean(train_accs))
        cv_results['std_train_acc'].append(np.std(train_accs))
        cv_results['mean_val_acc'].append(np.mean(val_accs))
        cv_results['std_val_acc'].append(np.std(val_accs))
        cv_results['mean_val_loss'].append(np.mean(val_losses))
        cv_results['std_val_loss'].append(np.std(val_losses))

        if verbose:
            print(f"Hidden dim {hidden_dim}: Val Acc = {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")

    return cv_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_cv_results(cv_results):
    """Plot cross-validation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    hidden_dims = cv_results['hidden_dim']

    # Accuracy plot
    ax1.errorbar(hidden_dims, cv_results['mean_train_acc'],
                 yerr=cv_results['std_train_acc'],
                 label='Training', marker='o', capsize=5)
    ax1.errorbar(hidden_dims, cv_results['mean_val_acc'],
                 yerr=cv_results['std_val_acc'],
                 label='Validation', marker='s', capsize=5)
    ax1.set_xlabel('Number of Hidden Neurons')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Cross-Validation: Accuracy vs Hidden Layer Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.errorbar(hidden_dims, cv_results['mean_val_loss'],
                 yerr=cv_results['std_val_loss'],
                 marker='o', color='red', capsize=5)
    ax2.set_xlabel('Number of Hidden Neurons')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Cross-Validation: Loss vs Hidden Layer Size')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_learning_curves(mlp):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.train_losses, label='Training Loss', linewidth=2)
    if mlp.val_losses:
        plt.plot(mlp.val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.show()


def compare_classifiers(X_test, y_test, mlp_model, map_predictions, scaler=None):
    """
    Compare MLP and MAP classifier performance

    Args:
        X_test: Test data
        y_test: True labels
        mlp_model: Trained MLP model
        map_predictions: MAP classifier predictions
        scaler: StandardScaler used for MLP
    """
    # Get MLP predictions
    if scaler:
        X_test_scaled = scaler.transform(X_test)
        mlp_predictions = mlp_model.predict(X_test_scaled)
    else:
        mlp_predictions = mlp_model.predict(X_test)

    # Calculate error rates
    mlp_error = 1 - np.mean(mlp_predictions == y_test)
    map_error = 1 - np.mean(map_predictions == y_test)

    # Create comparison report
    print("\n" + "=" * 60)
    print("CLASSIFIER PERFORMANCE COMPARISON")
    print("=" * 60)

    print(f"\nTheoretical MAP Classifier:")
    print(f"  Error Rate: {map_error * 100:.2f}%")
    print(f"  Accuracy: {(1 - map_error) * 100:.2f}%")

    print(f"\nMLP Classifier:")
    print(f"  Error Rate: {mlp_error * 100:.2f}%")
    print(f"  Accuracy: {(1 - mlp_error) * 100:.2f}%")

    print(f"\nPerformance Gap:")
    print(f"  Additional Error: {(mlp_error - map_error) * 100:.2f}%")
    print(f"  Relative Increase: {(mlp_error / map_error - 1) * 100:.1f}%")

    # Detailed classification reports
    print("\n" + "-" * 40)
    print("MAP Classifier - Per Class Performance:")
    print(classification_report(y_test, map_predictions, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))

    print("-" * 40)
    print("MLP Classifier - Per Class Performance:")
    print(classification_report(y_test, mlp_predictions, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))

    # Plot confusion matrices
    plot_confusion_matrix(y_test, map_predictions, "MAP Classifier - Confusion Matrix")
    plot_confusion_matrix(y_test, mlp_predictions, "MLP Classifier - Confusion Matrix")

    return {
        'mlp_error': mlp_error,
        'map_error': map_error,
        'gap': mlp_error - map_error
    }


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def run_mlp_experiment(X_train, y_train, X_test, y_test, map_predictions):
    """
    Complete MLP experiment pipeline

    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        map_predictions: Predictions from theoretical MAP classifier
    """

    print("=" * 60)
    print("MLP CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    # Step 1: Cross-validation to find best hidden layer size
    print("\n[1] CROSS-VALIDATION FOR HYPERPARAMETER TUNING")
    print("-" * 40)

    # Test different hidden layer sizes
    hidden_dims = [5, 10, 15, 20, 30, 40, 50, 75, 100]

    cv_results = cross_validate_mlp(
        X_train, y_train,
        hidden_dims=hidden_dims,
        k_folds=5,
        epochs=200,
        verbose=True
    )

    # Plot CV results
    plot_cv_results(cv_results)

    # Find best hidden dimension
    best_idx = np.argmax(cv_results['mean_val_acc'])
    best_hidden_dim = cv_results['hidden_dim'][best_idx]
    best_val_acc = cv_results['mean_val_acc'][best_idx]

    print(f"\nBest hidden layer size: {best_hidden_dim} neurons")
    print(f"Cross-validation accuracy: {best_val_acc * 100:.2f}%")

    # Step 2: Train final model with best parameters
    print("\n[2] TRAINING FINAL MODEL")
    print("-" * 40)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split training data for validation
    val_size = int(0.2 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_train_final = X_train_scaled[train_idx]
    y_train_final = y_train[train_idx]
    X_val_final = X_train_scaled[val_idx]
    y_val_final = y_train[val_idx]

    # Train final model
    final_mlp = MLPClassifier(
        input_dim=3,
        hidden_dim=best_hidden_dim,
        output_dim=4,
        learning_rate=0.01
    )

    print(f"Training with {best_hidden_dim} hidden neurons...")
    final_mlp.train(
        X_train_final, y_train_final,
        X_val_final, y_val_final,
        epochs=300,
        batch_size=32,
        verbose=True
    )

    # Plot learning curves
    plot_learning_curves(final_mlp)

    # Step 3: Compare with MAP classifier
    print("\n[3] PERFORMANCE COMPARISON")
    print("-" * 40)

    results = compare_classifiers(
        X_test, y_test,
        final_mlp,
        map_predictions,
        scaler
    )

    # Step 4: Additional Analysis
    print("\n[4] ADDITIONAL ANALYSIS")
    print("-" * 40)

    # Get prediction probabilities for test set
    test_probs = final_mlp.predict_proba(X_test_scaled)

    # Calculate average confidence for correct and incorrect predictions
    mlp_preds = final_mlp.predict(X_test_scaled)
    correct_mask = mlp_preds == y_test

    correct_confidence = np.mean([test_probs[i, mlp_preds[i] - 1]
                                  for i in range(len(mlp_preds)) if correct_mask[i]])
    incorrect_confidence = np.mean([test_probs[i, mlp_preds[i] - 1]
                                    for i in range(len(mlp_preds)) if not correct_mask[i]])

    print(f"Average confidence on correct predictions: {correct_confidence:.3f}")
    print(f"Average confidence on incorrect predictions: {incorrect_confidence:.3f}")
    print(f"Confidence gap: {correct_confidence - incorrect_confidence:.3f}")

    return final_mlp, scaler, results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("MLP Classifier Implementation for Gaussian Mixture Classification")
    print("=" * 60)
    print("\nThis script expects you to have already generated:")
    print("  - X_train, y_train: Training data from Gaussian distributions")
    print("  - X_test, y_test: Test data from Gaussian distributions")
    print("  - map_predictions: Predictions from theoretical MAP classifier")
    print("\nExample usage:")
    print("  final_mlp, scaler, results = run_mlp_experiment(")
    print("      X_train, y_train, X_test, y_test, map_predictions)")

    # Example with dummy data (replace with your actual data)
    # Assuming you have the GaussianClassDesigner from previous code

    # Get medium overlap configuration
    from Question1 import GaussianClassDesigner, get_configuration_medium_overlap

    means, covariances = get_configuration_medium_overlap()
    designer = GaussianClassDesigner(means, covariances)

    # Generate dataset
    X_train, yn_train, X_test, y_test = designer.generate_dataset(n_samples_per_class=2500)

    # Get MAP predictions
    map_predictions, _ = designer.map_classifier(X_test)

    # Run MLP experiment
    final_mlp, scaler, results = run_mlp_experiment(
        X_train, y_train, X_test, y_test, map_predictions
    )
