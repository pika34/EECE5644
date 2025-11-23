#!/usr/bin/env python3
"""
GMM-based image segmentation for Training Image #246053 [color].

Steps:
 - Locate and load the image (looks for file containing '246053' in /mnt/data)
 - Build 5D features for each pixel: (row, col, R, G, B)
 - Normalize each feature column to [0,1]
 - Use K-fold CV (avg validation log-likelihood) to choose K (n_components)
 - Fit final GMM on all pixels using best K
 - Assign each pixel to most likely component and visualize segmentation
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from skimage import io, img_as_float
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Parameters (change if needed)
# -------------------------
SEARCH_DIR = "C://Users//mmali//Downloads//246053.jpg"
  # looks for files containing this string in name
IMAGE_PATH = None           # optionally set explicit path here, else script searches SEARCH_DIR
CV_FOLDS = 5
Ks = [2, 4, 8, 16, 32]      # candidate mixture sizes
GMM_COVTYPE = 'full'
GMM_N_INIT = 3
GMM_REG_COVAR = 1e-6
CV_SUBSAMPLE_LIMIT = 50000  # if number of pixels > this, subsample for CV speed
RANDOM_STATE = 0



# -------------------------
# Load image
# -------------------------
img_path = SEARCH_DIR
print("Using image:", img_path)
img = img_as_float(io.imread(img_path))  # float in [0,1] if possible
# If grayscale convert to RGB
if img.ndim == 2:
    img = np.stack([img, img, img], axis=-1)
H, W, C = img.shape
print(f"Image shape: H={H}, W={W}, C={C}")

# -------------------------
# Build 5D features: (row, col, R, G, B)
# -------------------------
rows = np.arange(H)
cols = np.arange(W)
rr, cc = np.meshgrid(rows, cols, indexing='ij')  # shapes (H,W)
R = img[:, :, 0].ravel()
G = img[:, :, 1].ravel()
B = img[:, :, 2].ravel()
coord_r = rr.ravel().astype(float)
coord_c = cc.ravel().astype(float)

X_raw = np.column_stack([coord_r, coord_c, R, G, B])  # shape (N,5)
N = X_raw.shape[0]
print("Number of pixels (N):", N)

# -------------------------
# Normalize each feature column to [0,1] (min-max)
# -------------------------
mins = X_raw.min(axis=0)
maxs = X_raw.max(axis=0)
ranges = maxs - mins
# avoid divide by zero
ranges[ranges == 0] = 1.0
X = (X_raw - mins) / ranges
# Keep copy of mins/ranges if needed later
print("Feature mins:", mins)
print("Feature maxs:", maxs)

# -------------------------
# CV for model order selection (average validation log-likelihood)
# -------------------------
def compute_cv_scores(X_full, Ks, cv_folds=5, subsample_limit=50000, random_state=0):
    """
    Returns dict: {K: avg_validation_log_likelihood}
    Uses subsampling if X_full is large (to speed up CV).
    """
    rng = np.random.RandomState(random_state)
    N = X_full.shape[0]
    if N > subsample_limit:
        idx = rng.choice(np.arange(N), size=subsample_limit, replace=False)
        X_for_cv = X_full[idx]
        print(f"Subsampled {subsample_limit} / {N} pixels for CV.")
    else:
        X_for_cv = X_full
        print(f"Using all {N} pixels for CV.")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = {}
    for K in Ks:
        fold_scores = []
        print(f"Evaluating K={K} ...")
        for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_for_cv)):
            Xtr = X_for_cv[train_idx]
            Xval = X_for_cv[val_idx]
            gmm = GaussianMixture(n_components=K, covariance_type=GMM_COVTYPE,
                                  reg_covar=GMM_REG_COVAR, n_init=GMM_N_INIT,
                                  random_state=random_state)
            gmm.fit(Xtr)
            # score returns mean log-likelihood per sample on Xval
            val_score = gmm.score(Xval)
            fold_scores.append(val_score)
            print(f"  fold {fold_i+1}/{cv_folds}: val_mean_loglik = {val_score:.4f}")
        avg_score = float(np.mean(fold_scores))
        cv_scores[K] = avg_score
        print(f" --> K={K}, average val log-likelihood = {avg_score:.4f}")
    return cv_scores

cv_scores = compute_cv_scores(X, Ks, cv_folds=CV_FOLDS, subsample_limit=CV_SUBSAMPLE_LIMIT, random_state=RANDOM_STATE)

# -------------------------
# Choose best K and plot model-order curve
# -------------------------
best_K = max(cv_scores.items(), key=lambda kv: kv[1])[0]
print("Best K (by CV avg log-likelihood):", best_K)

plt.figure(figsize=(6,4))
Ks_list = list(cv_scores.keys())
scores_list = [cv_scores[k] for k in Ks_list]
plt.plot(Ks_list, scores_list, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of GMM components K')
plt.ylabel('Average validation log-likelihood (higher is better)')
plt.title('CV Model-order selection (GMM)')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Train final GMM on full data with best_K
# -------------------------
print(f"Training final GMM with K={best_K} on all pixels...")
final_gmm = GaussianMixture(n_components=best_K, covariance_type=GMM_COVTYPE,
                            reg_covar=GMM_REG_COVAR, n_init=10, random_state=RANDOM_STATE,
                            max_iter=500)
final_gmm.fit(X)
print("Final GMM trained.")

# -------------------------
# Assign each pixel to most likely component (MAP)
# -------------------------
probs = final_gmm.predict_proba(X)  # shape (N, K)
labels = probs.argmax(axis=1)       # shape (N,)
labels_img = labels.reshape(H, W)

# -------------------------
# Create grayscale visualization for label image
# -------------------------
K = best_K
grays = np.linspace(0, 1, K)  # values in [0,1] for imshow with cmap='gray'
seg_img = grays[labels].reshape(H, W)

# Option: colorize segments by component means in RGB for nicer visualization
component_colors = np.clip(final_gmm.means_[:, 2:5] * ranges[2:5] + mins[2:5], 0, 1)  # convert mean back to original scale approx
seg_color = np.zeros((H, W, 3), dtype=float)
for k in range(K):
    seg_color[labels_img == k] = component_colors[k]

# -------------------------
# Plot original and segmentation side by side
# -------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Original image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(seg_img, cmap='gray', vmin=0, vmax=1)
plt.title(f"GMM segmentation (labels grayscale) K={K}")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(seg_color)
plt.title("GMM segmentation (colored by component mean)")
plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------
# Optional: save segmentation images
# -------------------------
out_dir = "gmm_seg_output"
os.makedirs(out_dir, exist_ok=True)
plt.imsave(os.path.join(out_dir, "original.png"), img)
plt.imsave(os.path.join(out_dir, f"seg_gray_K{K}.png"), seg_img, cmap='gray')
plt.imsave(os.path.join(out_dir, f"seg_color_K{K}.png"), seg_color)
print("Saved outputs to folder:", out_dir)

# -------------------------
# Print quick diagnostics
# -------------------------
print("CV scores (K -> avg val log-likelihood):")
for k,v in cv_scores.items():
    print(f"  K={k}: {v:.4f}")

# Show component sizes
unique, counts = np.unique(labels, return_counts=True)
print("Final component counts (label -> pixels):")
for u,cnt in zip(unique, counts):
    print(f"  {u}: {cnt} pixels ({cnt/N:.3%})")
