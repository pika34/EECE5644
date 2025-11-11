import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MLP CLASSIFIER WITH CROSS-ENTROPY LOSS
# ============================================================================

class MLPClassifier:
    """
    2-layer MLP with ReLU activation, softmax output, and cross-entropy loss
    """

    def __init__(self, input_dim=3, hidden_dim=10, output_dim=4, learning_rate=0.01):
        # Initialize weights with He initialization for ReLU
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Random initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Store training history
        self.train_losses = []
        self.val_losses = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss
        y_true: one-hot encoded true labels
        y_pred: predicted probabilities
        """
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        """
        Backpropagation with cross-entropy loss
        """
        batch_size = X.shape[0]

        # Output layer gradients (simplified for softmax + cross-entropy)
        dz2 = y_pred - y_true
        dW2 = (self.a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=200, batch_size=32, verbose=True):
        """
        Train using mini-batch gradient descent with cross-entropy loss
        """
        # Convert y to one-hot encoding
        n_samples = X_train.shape[0]
        y_train_oh = np.zeros((n_samples, self.output_dim))
        y_train_oh[np.arange(n_samples), y_train - 1] = 1

        if X_val is not None and y_val is not None:
            n_val = X_val.shape[0]
            y_val_oh = np.zeros((n_val, self.output_dim))
            y_val_oh[np.arange(n_val), y_val - 1] = 1

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(batch_X)

                # Backward pass
                self.backward(batch_X, batch_y, y_pred)

            # Calculate and store losses
            train_pred = self.forward(X_train)
            train_loss = self.cross_entropy_loss(y_train_oh, train_pred)
            self.train_losses.append(train_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(y_val_oh, val_pred)
                self.val_losses.append(val_loss)

                if verbose and epoch % 50 == 0:
                    print(f"    Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            elif verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch:3d}: Train Loss = {train_loss:.4f}")

    def predict(self, X):
        """Make predictions (returns class labels 1,2,3,4)"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1) + 1

    def predict_proba(self, X):
        """Return prediction probabilities"""
        return self.forward(X)


# ============================================================================
# TRAINING WITH MULTIPLE RANDOM INITIALIZATIONS
# ============================================================================

def train_with_multiple_inits(X_train, y_train, X_val, y_val,
                              hidden_dim, n_inits=10, epochs=200):
    """
    Train multiple models with different random initializations
    Select the best model based on validation performance

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        hidden_dim: Number of hidden neurons (best P from cross-validation)
        n_inits: Number of random initializations to try
        epochs: Training epochs per initialization

    Returns:
        best_model: Model with lowest validation error
        all_results: Results from all initializations
    """

    print(f"  Training with {n_inits} random initializations (P={hidden_dim})...")

    best_model = None
    best_val_error = float('inf')
    all_results = []

    for init_num in range(n_inits):
        # Set random seed for reproducibility
        np.random.seed(42 + init_num * 100)

        # Create and train model
        model = MLPClassifier(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=4,
            learning_rate=0.01
        )

        # Train with cross-entropy loss
        model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=min(32, len(X_train) // 4),
            verbose=False
        )

        # Evaluate on validation set
        y_pred_val = model.predict(X_val)
        val_error = 1 - np.mean(y_pred_val == y_val)

        # Evaluate on training set
        y_pred_train = model.predict(X_train)
        train_error = 1 - np.mean(y_pred_train == y_train)

        # Store results
        result = {
            'init': init_num + 1,
            'model': model,
            'train_error': train_error,
            'val_error': val_error,
            'final_train_loss': model.train_losses[-1],
            'final_val_loss': model.val_losses[-1] if model.val_losses else None
        }
        all_results.append(result)

        # Check if this is the best model
        if val_error < best_val_error:
            best_val_error = val_error
            best_model = model

        print(f"    Init {init_num + 1:2d}: Train Error = {train_error:.4f}, Val Error = {val_error:.4f}")

    return best_model, all_results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_final_models():
    """
    Train final MLP models for each dataset using best P from cross-validation
    """

    # Best P values from cross-validation (you should update these with your actual results)
    # These are example values - replace with your actual cross-validation results
    best_P = {
        100: 10,  # Update with your actual best P
        500: 15,  # Update with your actual best P
        1000: 20,  # Update with your actual best P
        5000: 30,  # Update with your actual best P
        10000: 40  # Update with your actual best P
    }

    # Number of random initializations
    n_inits = 10

    # Store all final models and results
    final_models = {}
    all_training_results = {}

    print("=" * 70)
    print("TRAINING FINAL MLP MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"\nUsing cross-entropy loss for maximum likelihood training")
    print(f"Number of random initializations per model: {n_inits}")
    print(f"Best P values from cross-validation: {best_P}")

    # Load test dataset for final evaluation
    print("\nLoading test dataset for final evaluation...")
    df_test = pd.read_csv('test_100000.csv')
    X_test = df_test[['X1', 'X2', 'X3']].values
    y_test = df_test['Class'].values
    print(f"✓ Loaded test dataset: {len(df_test)} samples")

    # Process each training dataset
    for n_samples in [100, 500, 1000, 5000, 10000]:
        print(f"\n{'=' * 60}")
        print(f"DATASET: train_{n_samples}.csv (P={best_P[n_samples]})")
        print(f"{'=' * 60}")

        # Load training dataset
        try:
            df_train = pd.read_csv(f'train_{n_samples}.csv')
            X = df_train[['X1', 'X2', 'X3']].values
            y = df_train['Class'].values
            print(f"✓ Loaded {len(df_train)} training samples")
        except FileNotFoundError:
            print(f"✗ File train_{n_samples}.csv not found!")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        # Split into train/validation (80/20)
        val_size = int(0.2 * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]

        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")

        # Train with multiple initializations
        best_model, all_results = train_with_multiple_inits(
            X_train, y_train, X_val, y_val,
            hidden_dim=best_P[n_samples],
            n_inits=n_inits,
            epochs=200
        )

        # Find best and worst initializations
        val_errors = [r['val_error'] for r in all_results]
        best_init_idx = np.argmin(val_errors)
        worst_init_idx = np.argmax(val_errors)

        print(f"\n  Summary of {n_inits} initializations:")
        print(
            f"    Best init:  Val Error = {all_results[best_init_idx]['val_error']:.4f} (Init #{all_results[best_init_idx]['init']})")
        print(
            f"    Worst init: Val Error = {all_results[worst_init_idx]['val_error']:.4f} (Init #{all_results[worst_init_idx]['init']})")
        print(f"    Mean Val Error: {np.mean(val_errors):.4f} ± {np.std(val_errors):.4f}")

        # Evaluate best model on full test set
        y_pred_test = best_model.predict(X_test_scaled)
        test_error = 1 - np.mean(y_pred_test == y_test)
        test_accuracy = 1 - test_error

        print(f"\n  >>> Final Test Performance (100K samples):")
        print(f"      Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
        print(f"      Test Error: {test_error:.4f} ({test_error * 100:.2f}%)")

        # Store results
        final_models[n_samples] = {
            'model': best_model,
            'scaler': scaler,
            'best_P': best_P[n_samples],
            'test_error': test_error,
            'test_accuracy': test_accuracy,
            'all_init_results': all_results
        }
        all_training_results[n_samples] = all_results

    # Plot learning curves for best models
    plot_learning_curves(final_models)

    # Plot initialization variability
    plot_initialization_results(all_training_results)

    # Print final summary
    print_final_summary(final_models)

    # Save models
    save_models(final_models)

    return final_models


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_learning_curves(final_models):
    """Plot training and validation loss curves for best models"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    dataset_sizes = [100, 500, 1000, 5000, 10000]

    for idx, n_samples in enumerate(dataset_sizes):
        if n_samples in final_models:
            model = final_models[n_samples]['model']
            ax = axes[idx]

            # Plot losses
            epochs = range(1, len(model.train_losses) + 1)
            ax.plot(epochs, model.train_losses, label='Training Loss', linewidth=2)
            if model.val_losses:
                ax.plot(epochs, model.val_losses, label='Validation Loss', linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cross-Entropy Loss')
            ax.set_title(f'N = {n_samples:,} (P = {final_models[n_samples]["best_P"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide the 6th subplot
    axes[5].axis('off')

    plt.suptitle('Learning Curves for Final Models (Best Initialization)', fontsize=16)
    plt.tight_layout()
    plt.savefig('final_models_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_initialization_results(all_training_results):
    """Plot variation in performance across different initializations"""

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    all_errors = []
    labels = []

    for i, n_samples in enumerate([100, 500, 1000, 5000, 10000]):
        if n_samples in all_training_results:
            val_errors = [r['val_error'] * 100 for r in all_training_results[n_samples]]
            positions.append(i)
            all_errors.append(val_errors)
            labels.append(f'N={n_samples}')

    # Create box plot
    bp = ax.boxplot(all_errors, positions=positions, labels=labels,
                    patch_artist=True, widths=0.6)

    # Color the boxes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Validation Error Rate (%)', fontsize=12)
    ax.set_xlabel('Training Dataset Size', fontsize=12)
    ax.set_title('Performance Variability Across Random Initializations\n(10 initializations per dataset)',
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for Bayes error
    ax.axhline(y=20.77, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.7, 21.5, 'Bayes Error (20.77%)', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('initialization_variability.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# SUMMARY AND SAVE FUNCTIONS
# ============================================================================

def print_final_summary(final_models):
    """Print comprehensive summary of all final models"""

    print("\n" + "=" * 70)
    print("FINAL MODEL SUMMARY")
    print("=" * 70)

    print("\nTest Set Performance (100,000 samples):")
    print("-" * 50)
    print(f"{'Dataset':>10} | {'Best P':>8} | {'Test Acc':>10} | {'Test Error':>12} | {'Gap from Bayes':>15}")
    print("-" * 50)

    bayes_error = 0.2077  # From MAP classifier

    for n_samples in [100, 500, 1000, 5000, 10000]:
        if n_samples in final_models:
            fm = final_models[n_samples]
            gap = fm['test_error'] - bayes_error
            print(f"{n_samples:10,} | {fm['best_P']:8} | {fm['test_accuracy'] * 100:9.2f}% | "
                  f"{fm['test_error'] * 100:11.2f}% | {gap * 100:14.2f}%")

    print("\nKey Insights:")
    print("-" * 50)
    print("• Multiple initializations help avoid poor local minima")
    print("• Larger datasets show more consistent performance across initializations")
    print("• All models converge toward Bayes error as dataset size increases")
    print("• Cross-entropy loss effectively implements maximum likelihood training")


def save_models(final_models):
    """Save trained models to disk"""

    print("\n" + "-" * 50)
    print("Saving models...")

    for n_samples, model_data in final_models.items():
        filename = f'final_model_n{n_samples}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'best_P': model_data['best_P'],
                'test_error': model_data['test_error']
            }, f)
        print(f"  ✓ Saved {filename}")

    print("\nModels saved successfully!")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    # NOTE: Update the best_P values in train_final_models() with your actual
    # cross-validation results before running!

    print("\n" + "=" * 70)
    print("IMPORTANT: Update best_P values with your cross-validation results!")
    print("=" * 70)
    print("\nEdit the best_P dictionary in train_final_models() function")
    print("with the optimal P values from your cross-validation before running.")

    # Train all final models
    final_models = train_final_models()