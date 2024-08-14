# model.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load Iris dataset and train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction endpoint
@app.post("/predict")
async def predict(features: IrisFeatures):
    try:
        data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
        prediction = model.predict(data)
        return {"prediction": iris.target_names[prediction][0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    
# setosa
#{
#   "sepal_length": 5.1,
#   "sepal_width": 3.5,
#   "petal_length": 1.4,
#   "petal_width": 0.2
# }

# virginica
# {
#   "sepal_length": 7,
#   "sepal_width": 0.5,
#   "petal_length": 9,
#   "petal_width": 0.2
# }

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris classifier API!"}