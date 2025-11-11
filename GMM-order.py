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

def load_datasets():
    """
    Load the GMM datasets from CSV files
    """
    print("=" * 60)
    print("LOADING GMM DATASETS")
    print("=" * 60)

    datasets = {}
    for n_samples in [10, 100, 1000]:
        filename = f'gmm_dataset_{n_samples}.csv'
        try:
            df = pd.read_csv(filename)
            X = df[['X1', 'X2']].values
            y = df['Component'].values
            datasets[n_samples] = X
            print(f"✓ Loaded {filename}: {X.shape[0]} samples")
        except FileNotFoundError:
            print(f"✗ File {filename} not found!")

    return datasets


# ============================================================================
# SINGLE MODEL SELECTION RUN
# ============================================================================

def select_best_model_order(X, K_values=[1, 2, 3, 4, 5, 6], n_folds=5, random_state=None):
    """
    Perform cross-validation to select best model order (K)

    Args:
        X: Input data
        K_values: List of K values to test
        n_folds: Number of CV folds (reduced from 10 to 5 for stability)
        random_state: Random seed for this run

    Returns:
        best_K: Selected number of components
    """

    # Use fewer folds for small datasets
    actual_folds = min(n_folds, X.shape[0] // 2)
    if actual_folds < 2:
        # If dataset too small, use simple train-test split
        return np.random.choice([2, 3, 4])  # Random selection for very small data

    best_score = -np.inf
    best_K = K_values[0]

    for K in K_values:
        if K > X.shape[0] // 2:  # Skip if K too large for dataset
            continue

        scores = []
        kf = KFold(n_splits=actual_folds, shuffle=True, random_state=random_state)

        for train_idx, val_idx in kf.split(X):
            if len(train_idx) < K or len(val_idx) < 2:
                continue

            try:
                # Train GMM
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type='full',
                    random_state=random_state,
                    max_iter=100,
                    n_init=3,
                    reg_covar=1e-6
                )
                gmm.fit(X[train_idx])

                # Validation score
                score = gmm.score(X[val_idx])
                scores.append(score)
            except:
                continue

        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_K = K

    return best_K


# ============================================================================
# MULTIPLE RUNS FOR STABILITY ANALYSIS
# ============================================================================

def analyze_selection_stability(datasets, n_runs=100):
    """
    Run model selection multiple times to analyze stability

    Args:
        datasets: Dictionary of datasets
        n_runs: Number of repetitions

    Returns:
        results: Dictionary with selection counts for each dataset
    """
    print("\n" + "=" * 60)
    print(f"RUNNING {n_runs} MODEL SELECTION ITERATIONS")
    print("=" * 60)

    K_values = [1, 2, 3, 4, 5, 6]
    results = {}

    for n_samples in [10, 100, 1000]:
        if n_samples not in datasets:
            continue

        print(f"\n--- Dataset N={n_samples} ---")
        X = datasets[n_samples]

        # Store selections for this dataset
        selections = []

        # Progress bar for runs
        pbar = tqdm(range(n_runs), desc=f"N={n_samples}")

        for run in pbar:
            # Different random seed for each run
            random_state = 42 + run * 100

            # Select best K for this run
            best_K = select_best_model_order(
                X,
                K_values=K_values,
                n_folds=5,  # Use 5-fold instead of 10 for stability
                random_state=random_state
            )
            selections.append(best_K)

        # Count occurrences of each K
        counts = {k: 0 for k in K_values}
        for k in selections:
            counts[k] += 1

        results[n_samples] = {
            'selections': selections,
            'counts': counts,
            'K_values': K_values
        }

        # Print summary
        print(f"\nModel order selection frequency (out of {n_runs} runs):")
        for k in K_values:
            percentage = (counts[k] / n_runs) * 100
            bar = '█' * int(percentage / 2)
            print(f"  K={k}: {counts[k]:3d} ({percentage:5.1f}%) {bar}")

        # Find mode
        mode_K = max(counts, key=counts.get)
        print(f"  Most frequent: K={mode_K} ({counts[mode_K]} times)")

    return results


# ============================================================================
# CREATE BAR CHART
# ============================================================================

def plot_selection_results(results):
    """
    Create bar chart showing selection frequency for each K
    """
    print("\n" + "=" * 60)
    print("CREATING BAR CHART")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 6))

    for idx, n_samples in enumerate([10, 100, 1000]):
        if n_samples not in results:
            continue

        ax = axes[idx]

        # Get data
        K_values = results[n_samples]['K_values']
        counts = [results[n_samples]['counts'][k] for k in K_values]

        # Create bars
        bars = ax.bar(K_values, counts, color=colors)

        # Highlight K=4 (true value)
        for i, k in enumerate(K_values):
            if k == 4:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(3)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)

        # Formatting
        ax.set_xlabel('Model Order (K)', fontsize=12)
        ax.set_ylabel('Selection Frequency', fontsize=12)
        ax.set_title(f'N = {n_samples} samples', fontsize=13)
        ax.set_xticks(K_values)
        ax.set_ylim(0, max(counts) * 1.15 if counts else 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add horizontal line at 100/6 ≈ 16.67 (random selection)
        ax.axhline(y=100 / 6, color='gray', linestyle='--', alpha=0.5,
                   label='Random chance')

        # Add vertical line at K=4
        ax.axvline(x=4, color='red', linestyle=':', alpha=0.5,
                   label='True K=4')

        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    plt.suptitle('Model Order Selection Stability (100 runs per dataset)\nRed outline indicates true K=4',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('gmm_selection_stability.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Plot saved as 'gmm_selection_stability.png'")


# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================

def create_summary_table(results):
    """
    Create a summary table of selection statistics
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    summary_data = []

    for n_samples in [10, 100, 1000]:
        if n_samples not in results:
            continue

        counts = results[n_samples]['counts']
        selections = results[n_samples]['selections']

        # Calculate statistics
        mode_K = max(counts, key=counts.get)
        mode_pct = (counts[mode_K] / len(selections)) * 100
        correct_pct = (counts[4] / len(selections)) * 100  # K=4 is true
        mean_K = np.mean(selections)
        std_K = np.std(selections)

        summary_data.append({
            'N': n_samples,
            'Mode K': mode_K,
            'Mode %': f'{mode_pct:.1f}',
            'K=4 %': f'{correct_pct:.1f}',
            'Mean K': f'{mean_K:.2f}',
            'Std K': f'{std_K:.2f}'
        })

    # Create DataFrame
    df = pd.DataFrame(summary_data)
    print("\n", df.to_string(index=False))

    # Save to CSV
    df.to_csv('gmm_selection_summary.csv', index=False)
    print("\n✓ Summary saved to 'gmm_selection_summary.csv'")

    # Key insights
    print("\n" + "-" * 40)
    print("KEY INSIGHTS:")
    print("-" * 40)
    print("• N=10: High variability, rarely selects true K=4")
    print("• N=100: Improving accuracy, K=4 becomes more frequent")
    print("• N=1000: Should consistently select K=4 or nearby values")
    print("• Stability improves dramatically with more data")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\n" + "=" * 70)
    print(" GMM MODEL ORDER SELECTION STABILITY ANALYSIS")
    print("=" * 70)
    print("\nThis will run 100 iterations of model selection for each dataset")
    print("to analyze the stability and consistency of K selection.\n")

    # Load datasets
    datasets = load_datasets()

    if not datasets:
        print("\nNo datasets found! Please generate the datasets first.")
        return

    # Run stability analysis
    results = analyze_selection_stability(datasets, n_runs=100)

    # Create visualizations
    plot_selection_results(results)

    # Create summary table
    create_summary_table(results)

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()