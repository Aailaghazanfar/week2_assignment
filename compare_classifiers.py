
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/content/week2_assignment"

CLASS_NAMES = ["birds", "cats", "dogs"]

RANDOM_STATE = 42


# ============================================================
# Load Setting A HOG Features
# ============================================================

X = np.load(
    os.path.join(DATA_DIR, "X_setting_A.npy")
)

y = np.load(
    os.path.join(DATA_DIR, "y_setting_A.npy")
)

print("Setting A features loaded")
print("Feature matrix shape:", X.shape)
print("Number of labels:", len(y))


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain/Test Split")
print("-" * 40)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# Classifiers
# ============================================================

models = {
    "Linear SVM": LinearSVC(
        max_iter=10000,
        random_state=RANDOM_STATE
    ),

    "k-NN (k=5)": KNeighborsClassifier(
        n_neighbors=5
    )
}


# ============================================================
# Train and Evaluate
# ============================================================

results = {}


for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_NAMES
    )

    print("Confusion Matrix:")
    print(cm)


# ============================================================
# Summary Table
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFIER COMPARISON")
print("=" * 60)

print(
    f"{'Classifier':<20} {'Accuracy':>10}"
)

print("-" * 32)

for name, accuracy in results.items():

    print(
        f"{name:<20} {accuracy:>10.4f}"
    )


# ============================================================
# Best Classifier
# ============================================================

best_model = max(
    results,
    key=results.get
)

print("\nBest classifier:")
print(best_model)

print(
    f"Best accuracy: {results[best_model]:.4f}"
)
