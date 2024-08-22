import mlflow
import mlflow.data.pandas_dataset
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import dagshub
# dagshub.init(repo_owner='George1044', repo_name='mlflow_demo', mlflow=True)


# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Set up the experiment
mlflow.set_experiment("my_ml_experiment")

# Start a new MLflow run
with mlflow.start_run():
    # Define model parameters
    n_estimators = 100
    max_depth = 5
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_input(dataset=mlflow.data.from_pandas(pd.DataFrame(iris.data)), context="training")

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log the accuracy as a metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model trained with accuracy: {accuracy}")

    # Optionally, log any other artifacts (e.g., model plot, confusion matrix)
    # mlflow.log_artifact("path/to/artifact.png")

