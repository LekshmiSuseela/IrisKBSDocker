import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
import shap

# Load iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = pd.Series(iris.target, name="target")

# Add random sensitive attribute 'location'
np.random.seed(42)
X["location"] = np.random.randint(0, 2, X.shape[0])

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

# Fairness Evaluation using location as sensitive attribute
sensitive_feature = X_test["location"]

metric_frame = MetricFrame(
    metrics=accuracy_score,
    y_true=y_test,
    y_pred=preds,
    sensitive_features=sensitive_feature
)

print("\nFairness Metrics by Group:\n", metric_frame.by_group)

# ----- SHAP EXPLAINER -----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot for CLASS Virginica (class index 2)
print("\nGenerating SHAP summary plot for class Virginica...")
shap.summary_plot(shap_values[2], X_test, show=True)
