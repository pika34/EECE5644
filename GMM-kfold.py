import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD DATASETS
# ============================================================================

def load_gmm_datasets():
    """
    Load the previously generated GMM datasets from CSV files
    """
    print("=" * 60)
    print("LOADING GMM DATASETS")
    print("=" * 60)

    datasets = {}
    dataset_sizes = [10, 100, 1000]

    for n_samples in dataset_sizes:
        filename = f'gmm_dataset_{n_samples}.csv'
        try:
            df = pd.read_csv(filename)
            X = df[['X1', 'X2']].values
            y = df['Component'].values

            datasets[n_samples] = {
                'X': X,
                'y': y,
                'n_samples': n_samples
            }

            print(f"✓ Loaded {filename}: {X.shape[0]} samples, {X.shape[1]} features")

        except FileNotFoundError:
            print(f"✗ File {filename} not found! Please run the dataset generation script first.")

    return datasets


# ============================================================================
# CROSS-VALIDATION FOR GMM
# ============================================================================

def cross_validate_gmm(X, K_values, n_folds=10, random_state=42):
    """
    Perform k-fold cross-validation for GMM with different numbers of components

    Args:
        X: Input data (n_samples x n_features)
        K_values: List of component numbers to test
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        results: Dictionary with CV results
    """

    # Initialize results storage
    results = {
        'K': [],
        'fold_loglik': [],  # Store all fold log-likelihoods
        'mean_loglik': [],
        'std_loglik': [],
        'aic': [],
        'bic': []
    }

    # Special handling for small datasets
    actual_folds = min(n_folds, X.shape[0])
    if actual_folds < n_folds:
        print(f"  Note: Using {actual_folds} folds instead of {n_folds} due to small sample size")

    # Create KFold object
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=random_state)

    # Test each value of K
    for K in K_values:
        fold_logliks = []
        aic_scores = []
        bic_scores = []

        # Skip if K > n_samples (can't fit more components than samples)
        if K > X.shape[0]:
            print(f"  Skipping K={K} (exceeds number of samples)")
            continue

        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train = X[train_idx]
            X_val = X[val_idx]

            # Skip if validation set is too small
            if len(X_val) < 2:
                continue

            try:
                # Train GMM on training fold
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type='full',
                    random_state=random_state,
                    max_iter=100,
                    n_init=5
                )
                gmm.fit(X_train)

                # Compute validation log-likelihood
                val_loglik = gmm.score(X_val) * len(X_val)  # Total log-likelihood
                fold_logliks.append(val_loglik)

                # Compute AIC and BIC on training data
                aic_scores.append(gmm.aic(X_train))
                bic_scores.append(gmm.bic(X_train))

            except:
                # Handle convergence issues
                continue

        # Store results if we got valid folds
        if fold_logliks:
            results['K'].append(K)
            results['fold_loglik'].append(fold_logliks)
            results['mean_loglik'].append(np.mean(fold_logliks))
            results['std_loglik'].append(np.std(fold_logliks))
            results['aic'].append(np.mean(aic_scores) if aic_scores else np.nan)
            results['bic'].append(np.mean(bic_scores) if bic_scores else np.nan)

    return results


# ============================================================================
# RUN CROSS-VALIDATION FOR ALL DATASETS
# ============================================================================

def run_all_cross_validations(datasets):
    """
    Run cross-validation for all dataset sizes
    """
    print("\n" + "=" * 60)
    print("RUNNING CROSS-VALIDATION")
    print("=" * 60)

    # Define K values to test
    K_ranges = {
        10: [1, 2, 3, 4, 5],  # For N=10: test K=1 to 5
        100: [1, 2, 3, 4, 5, 6, 7, 8,9,10],  # For N=100: test K=1 to 8
        1000: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # For N=1000: test K=1 to 10
    }

    all_results = {}

    for n_samples in [10, 100, 1000]:
        if n_samples not in datasets:
            continue

        print(f"\n--- Dataset with N={n_samples} samples ---")
        X = datasets[n_samples]['X']
        K_values = K_ranges[n_samples]

        print(f"  Testing K values: {K_values}")
        print(f"  Running 10-fold cross-validation...")

        # Run cross-validation
        results = cross_validate_gmm(X, K_values, n_folds=10)
        all_results[n_samples] = results

        # Find best K
        if results['mean_loglik']:
            best_idx = np.argmax(results['mean_loglik'])
            best_K = results['K'][best_idx]
            best_loglik = results['mean_loglik'][best_idx]

            print(f"  ✓ Best K = {best_K} with mean log-likelihood = {best_loglik:.2f}")

    return all_results


# ============================================================================
# CREATE RESULTS TABLE
# ============================================================================

def create_results_table(all_results):
    """
    Create and display results tables for each dataset
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS TABLES")
    print("=" * 60)

    tables = {}

    for n_samples, results in all_results.items():
        print(f"\n--- Results for N={n_samples} ---")

        # Check if results are empty
        if not results['K']:
            print(f"  No valid results for N={n_samples} (dataset too small)")
            continue

        # Create DataFrame
        df = pd.DataFrame({
            'K': results['K'],
            'Mean_LogLik': results['mean_loglik'],
            'Std_LogLik': results['std_loglik'],
            'AIC': results['aic'],
            'BIC': results['bic']
        })

        # Check if DataFrame is empty
        if df.empty or len(df) == 0:
            print(f"  No valid cross-validation results for N={n_samples}")
            continue

        # Round for display
        df = df.round(2)

        # Mark best K (only if we have data)
        if not df['Mean_LogLik'].isna().all():
            best_idx = df['Mean_LogLik'].idxmax()
            df['Best'] = ''
            if best_idx is not None:
                df.loc[best_idx, 'Best'] = '*'
        else:
            df['Best'] = ''

        tables[n_samples] = df

        print(df.to_string(index=False))

        # Save to CSV
        filename = f'gmm_cv_results_n{n_samples}.csv'
        df.to_csv(filename, index=False)
        print(f"  Saved to {filename}")

    return tables


# ============================================================================
# PLOT LOG-LIKELIHOOD CURVES
# ============================================================================

def plot_loglikelihood_curves(all_results):
    """
    Plot log-likelihood vs K for all datasets
    """
    print("\n" + "=" * 60)
    print("PLOTTING LOG-LIKELIHOOD CURVES")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'green', 'red']

    for idx, n_samples in enumerate([10, 100, 1000]):
        if n_samples not in all_results:
            continue

        ax = axes[idx]
        results = all_results[n_samples]

        # Check if we have valid results
        if not results['K'] or not results['mean_loglik']:
            ax.text(0.5, 0.5, f'No valid results\nfor N={n_samples}',
                    ha='center', va='center', fontsize=12,
                    transform=ax.transAxes)
            ax.set_title(f'N = {n_samples} samples', fontsize=12)
            ax.set_xlabel('Number of Components (K)', fontsize=11)
            ax.set_ylabel('Log-Likelihood', fontsize=11)
            continue

        # Plot with error bars
        ax.errorbar(results['K'], results['mean_loglik'],
                    yerr=results['std_loglik'],
                    marker='o', markersize=8,
                    linewidth=2, capsize=5,
                    color=colors[idx],
                    label='Mean ± Std')

        # Mark the best K
        if results['mean_loglik']:
            best_idx = np.argmax(results['mean_loglik'])
            best_K = results['K'][best_idx]
            best_loglik = results['mean_loglik'][best_idx]

            ax.scatter(best_K, best_loglik,
                       s=200, marker='*',
                       color='red', edgecolor='black',
                       linewidth=2, zorder=5,
                       label=f'Best K={best_K}')

        # Add vertical line at true K=4
        ax.axvline(x=4, color='gray', linestyle='--',
                   alpha=0.5, label='True K=4')

        # Formatting
        ax.set_xlabel('Number of Components (K)', fontsize=11)
        ax.set_ylabel('Log-Likelihood', fontsize=11)
        ax.set_title(f'N = {n_samples} samples', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Set integer x-ticks
        if results['K']:
            ax.set_xticks(results['K'])

    plt.suptitle('GMM Model Selection: Log-Likelihood vs Number of Components',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('gmm_model_selection.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Plot saved as 'gmm_model_selection.png'")


# ============================================================================
# TRAIN FINAL BEST MODELS
# ============================================================================

def train_best_models(datasets, all_results):
    """
    Train the best GMM for each dataset size
    """
    print("\n" + "=" * 60)
    print("TRAINING BEST MODELS")
    print("=" * 60)

    best_models = {}

    for n_samples in [10, 100, 1000]:
        if n_samples not in datasets or n_samples not in all_results:
            continue

        # Check if we have valid results
        results = all_results[n_samples]
        if not results['K'] or not results['mean_loglik']:
            print(f"\n--- N={n_samples}: No valid CV results to select best K ---")
            continue

        # Find best K
        best_idx = np.argmax(results['mean_loglik'])
        best_K = results['K'][best_idx]

        print(f"\n--- N={n_samples}: Training GMM with K={best_K} ---")

        # Train on full dataset
        X = datasets[n_samples]['X']
        best_gmm = GaussianMixture(
            n_components=best_K,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            n_init=10
        )
        best_gmm.fit(X)

        # Compute metrics
        loglik = best_gmm.score(X) * len(X)
        aic = best_gmm.aic(X)
        bic = best_gmm.bic(X)

        print(f"  Log-likelihood: {loglik:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  BIC: {bic:.2f}")

        # Store model
        best_models[n_samples] = {
            'model': best_gmm,
            'K': best_K,
            'loglik': loglik,
            'aic': aic,
            'bic': bic
        }

        # Print learned parameters
        print(f"  Learned means shape: {best_gmm.means_.shape}")
        print(f"  Learned weights: {best_gmm.weights_.round(3)}")

    return best_models


# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================

def print_final_summary(best_models):
    """
    Print final summary comparing all best models
    """
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\nTrue GMM has K=4 components")
    print("\nBest K selected by cross-validation:")
    print("-" * 40)
    print(f"{'Dataset':>10} | {'Best K':>8} | {'Correct?':>10}")
    print("-" * 40)

    for n_samples, model_info in best_models.items():
        correct = "✓" if model_info['K'] == 4 else "✗"
        print(f"{n_samples:10} | {model_info['K']:8} | {correct:>10}")

    print("\nKey Observations:")
    print("-" * 40)
    print("• N=10: Too few samples to identify true K=4")
    print("• N=100: May identify K=4 or nearby value")
    print("• N=1000: Should reliably identify K=4")
    print("• Model selection improves with more data")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "=" * 70)
    print(" GMM MODEL SELECTION VIA CROSS-VALIDATION")
    print("=" * 70)

    # Load datasets
    datasets = load_gmm_datasets()

    if not datasets:
        print("\nNo datasets found! Please run the dataset generation script first.")
        return

    # Run cross-validation
    all_results = run_all_cross_validations(datasets)

    # Create results tables
    tables = create_results_table(all_results)

    # Plot log-likelihood curves
    plot_loglikelihood_curves(all_results)

    # Train best models
    best_models = train_best_models(datasets, all_results)

    # Print summary
    print_final_summary(best_models)

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)

    return all_results, best_models


if __name__ == "__main__":
    results, models = main()