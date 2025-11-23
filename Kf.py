# -------------------------------------------------------------
# K-FOLD CROSS VALIDATION SETUP
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, zero_one_loss
import main
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# -------------------------------------------------------------
# K-FOLD CROSS VALIDATION SETUP (use the dataset from above)
# -------------------------------------------------------------

X_train, y_train = main.generate_ring_samples(1000, random_state=main.SEED)
X_test, y_test = main.generate_ring_samples(10000, random_state=main.SEED + 1)

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=0)

C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [0.01, 0.1, 1, 10]

val_errors = np.zeros((len(C_values), len(gamma_values)))

# -------------------------------------------------------------
# GRID SEARCH OVER (C, gamma)
# -------------------------------------------------------------
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):

        fold_errors = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_val)
            error = 1 - accuracy_score(y_val, y_pred)
            fold_errors.append(error)

        val_errors[i, j] = np.mean(fold_errors)
        print(f"C={C}, gamma={gamma}: val_error={val_errors[i, j]:.4f}")

# -------------------------------------------------------------
# HEATMAP OF VALIDATION ERROR
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(val_errors, cmap='viridis', origin='lower')
plt.colorbar(label="Validation Error")

plt.xticks(np.arange(len(gamma_values)), gamma_values)
plt.yticks(np.arange(len(C_values)), C_values)

plt.xlabel("gamma")
plt.ylabel("C")
plt.title("Validation Error Heatmap for SVM (RBF)")
plt.show()
results_svm = []  # store (C, gamma, val_error)

for C in C_values:
    for gamma in gamma_values:

        fold_errors = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = SVC(C=C, kernel="rbf", gamma=gamma)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_errors.append(zero_one_loss(y_val, y_pred))

        avg_error = np.mean(fold_errors)
        results_svm.append((C, gamma, avg_error))
        print(f"C={C}, gamma={gamma}: val_error={avg_error:.4f}")

# Find best
best_svm = min(results_svm, key=lambda x: x[2])
best_C, best_gamma, best_error = best_svm

print("\nBEST SVM HYPERPARAMETERS:")
print(f"C = {best_C}, gamma = {best_gamma}, val_error = {best_error:.4f}")
best_svm_model = SVC(C=best_C, gamma=best_gamma, kernel="rbf")
best_svm_model.fit(X_train, y_train)

test_pred = best_svm_model.predict(X_test)
test_error_svm = zero_one_loss(y_test, test_pred)

print(f"Final SVM Test Error: {test_error_svm:.4f}")

hidden_sizes = [2, 4, 6, 8, 10, 20]

results_mlp = []

for h in hidden_sizes:
    fold_errors = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = MLPClassifier(
            hidden_layer_sizes=(h,),
            activation="tanh",  # good for circular boundaries
            solver="adam",
            max_iter=3000,
            early_stopping=True,
            random_state=0
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        fold_errors.append(zero_one_loss(y_val, y_pred))

    avg_error = np.mean(fold_errors)
    results_mlp.append((h, avg_error))
    print(f"hidden={h}: val_error={avg_error:.4f}")

best_mlp = min(results_mlp, key=lambda x: x[1])
best_hidden, best_val_err = best_mlp

print("\nBEST MLP HYPERPARAMETER:")
print(f"Hidden layer size = {best_hidden}, val_error = {best_val_err:.4f}")

best_mlp_model = MLPClassifier(
    hidden_layer_sizes=(best_hidden,),
    activation="tanh",
    solver="adam",
    max_iter=3000
)

best_mlp_model.fit(X_train, y_train)
test_pred_mlp = best_mlp_model.predict(X_test)
test_error_mlp = zero_one_loss(y_test, test_pred_mlp)

print(f"Final MLP Test Error: {test_error_mlp:.4f}")
