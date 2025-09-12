import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(data_csv="data/processed/posture_data.csv", model_out_dir="data/models"):
    """
    Trains and compares SVC and RandomForest models, saving the best one.
    """
    df = pd.read_csv(data_csv)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 1. Hyperparameter Tuning for SVC ---
    svc_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }
    svc_grid = GridSearchCV(SVC(probability=True), svc_params, refit=True, verbose=0, cv=5)
    svc_grid.fit(X_train, y_train)
    svc_best = svc_grid.best_estimator_
    svc_accuracy = accuracy_score(y_test, svc_best.predict(X_test))
    print(f"Tuned SVC Accuracy: {svc_accuracy:.4f}")
    print(f"Best SVC params: {svc_grid.best_params_}")

    # --- 2. Hyperparameter Tuning for RandomForest ---
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, refit=True, verbose=0, cv=5)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_accuracy = accuracy_score(y_test, rf_best.predict(X_test))
    print(f"Tuned RandomForest Accuracy: {rf_accuracy:.4f}")
    print(f"Best RandomForest params: {rf_grid.best_params_}")

    # --- 3. Compare and Select Best Model ---
    if rf_accuracy > svc_accuracy:
        best_model = rf_best
        model_name = "posture_model_rf.pkl"
        print("RandomForest is the winner!")
    else:
        best_model = svc_best
        model_name = "posture_model_svc.pkl"
        print("SVC is the winner!")

    # --- Save the Best Model ---
    os.makedirs(model_out_dir, exist_ok=True)
    model_path = os.path.join(model_out_dir, model_name)
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    train_model()