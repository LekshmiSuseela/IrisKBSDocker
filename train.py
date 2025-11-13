import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load Iris dataset
iris = load_iris()
X, y = iris.data.copy(), iris.target.copy()

def poison_data(X, poison_level=0.05, seed=42):
    np.random.seed(seed)
    X_poisoned = X.copy()
    n_samples = int(len(X) * poison_level)
    idx = np.random.choice(len(X), n_samples, replace=False)
    # Replace selected rows with random numbers within feature ranges
    for i in idx:
        X_poisoned[i] = np.random.uniform(X.min(axis=0), X.max(axis=0))
    return X_poisoned

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poison_levels = [0.05, 0.1, 0.5]

for level in poison_levels:
    X_train_poisoned = poison_data(X_train, poison_level=level)
    
    with mlflow.start_run(run_name=f"poison_{int(level*100)}%"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_poisoned, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Poison level {int(level*100)}% -> Accuracy: {acc:.3f}")
        
        # Log model and metrics in MLFlow
        mlflow.log_param("poison_level", level)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")
