#!/usr/bin/env python3
"""
Section 1 — Problem setup & data generation for concentric noisy rings.

Generates:
 - 1000 training samples (labels in {-1, +1})
 - 10000 test samples

Data model:
 x = r_l * [cos(theta), sin(theta)]^T + n,
 where theta ~ Uniform[-pi, pi], n ~ N(0, sigma^2 I),
 r_{-1} = 2, r_{+1} = 4, sigma = 1.

Saves results to 'rings_data.npz' with arrays:
 - X_train, y_train, X_test, y_test

Also:
 - plots scatter of training samples
 - plots histogram of radial distances (to show overlap)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os

R_NEG = 2.0   # radius for class -1
R_POS = 4.0   # radius for class +1
SIGMA = 1.0   # Gaussian noise std dev
SEED = 0

def generate_ring_samples(n_samples, r_neg=R_NEG, r_pos=R_POS, sigma=SIGMA, random_state=None):
    """
    Generate n_samples (half per class) according to the spec.

    Returns:
        X: (n_samples, 2) float
        y: (n_samples,) int in {-1, +1}
    """
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2

    # angles
    theta_neg = rng.uniform(-np.pi, np.pi, size=n_half)
    theta_pos = rng.uniform(-np.pi, np.pi, size=n_half)

    # ideal ring points (no noise)
    x_neg = r_neg * np.column_stack((np.cos(theta_neg), np.sin(theta_neg)))
    x_pos = r_pos * np.column_stack((np.cos(theta_pos), np.sin(theta_pos)))

    # vertical stack and add Gaussian noise
    X = np.vstack((x_neg, x_pos))
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    X = X + noise

    # labels: -1 for the inner ring, +1 for the outer ring
    y = np.hstack((np.full(n_half, -1, dtype=int), np.full(n_half, +1, dtype=int)))

    # shuffle for random ordering
    X, y = shuffle(X, y, random_state=rng)

    return X, y

def plot_training_scatter(X, y, figsize=(6,6), savepath=None):
    plt.figure(figsize=figsize)
    plt.scatter(X[y==-1,0], X[y==-1,1], s=12, alpha=0.7, label='class -1 (r=2)')
    plt.scatter(X[y==+1,0], X[y==+1,1], s=12, alpha=0.7, label='class +1 (r=4)')
    plt.legend()
    plt.gca().set_aspect('equal', 'box')
    plt.title('Training samples (noisy concentric rings)')
    plt.xlabel('x1'); plt.ylabel('x2')
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

def plot_radial_histograms(X, y, bins=50, savepath=None):
    """Plot histogram of radial distances for each class to show overlap."""
    r = np.linalg.norm(X, axis=1)
    r_neg = r[y==-1]
    r_pos = r[y==+1]

    plt.figure(figsize=(6,4))
    plt.hist(r_neg, bins=bins, alpha=0.6, label='class -1 (r=2)', density=True)
    plt.hist(r_pos, bins=bins, alpha=0.6, label='class +1 (r=4)', density=True)
    plt.axvline(R_NEG, color='k', linestyle='--', label='r=-1 (ideal)')
    plt.axvline(R_POS, color='k', linestyle=':', label='r=+1 (ideal)')
    plt.xlabel('radial distance ||x||')
    plt.ylabel('density')
    plt.title('Radial distance distributions (training set)')
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    out_file = 'rings_data.npz'
    print("Generating data with seed =", SEED)
    X_train, y_train = generate_ring_samples(1000, random_state=SEED)
    X_test, y_test = generate_ring_samples(10000, random_state=SEED + 1)

    # basic sanity prints
    print("Shapes:")
    print(" X_train:", X_train.shape, " y_train:", y_train.shape)
    print(" X_test :", X_test.shape, " y_test :", y_test.shape)
    print("Label counts (train):", np.unique(y_train, return_counts=True))
    print("Label counts (test) :", np.unique(y_test, return_counts=True))

    # create output dir if needed
    out_dir = 'output_figs'
    os.makedirs(out_dir, exist_ok=True)

    # plots
    plot_training_scatter(X_train, y_train, savepath=os.path.join(out_dir, 'train_scatter.png'))
    plot_radial_histograms(X_train, y_train, savepath=os.path.join(out_dir, 'radial_hist.png'))

    # save datasets
    np.savez_compressed(out_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved datasets to '{out_file}'. Plots saved in '{out_dir}/'.")

if __name__ == "__main__":
    main()
