from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("feature_selection", SelectPercentile(f_classif, percentile=50)),
    ("model", LogisticRegression(max_iter=200))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "iris_pipeline.pkl")

print("Model saved successfully")
