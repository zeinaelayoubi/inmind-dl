# model1.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pika
import os
import json

# Load Iris dataset and train Model 1 (Setosa vs Non-Setosa)
iris = load_iris()
X = iris.data
y = (iris.target != 0).astype(int)  # Binary classification: Setosa (0) vs Non-Setosa (1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)

# Initialize FastAPI app for Model 1
app = FastAPI()

# RabbitMQ connection for sending predictions to Model 2
def get_rabbitmq_connection():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost")))
    return connection

# Define request schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction endpoint for Model 1
@app.post("/predict")
async def predict(features: IrisFeatures):
    try:
        data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
        prediction = model1.predict(data)
        if prediction[0] == 0:  # Setosa
            return {"prediction": "setosa"}
        else:
            # Non-setosa, send data to Model 2
            connection = get_rabbitmq_connection()
            channel = connection.channel()
            channel.queue_declare(queue='non_setosa_predictions')
            message = json.dumps(features.dict())
            channel.basic_publish(exchange='', routing_key='non_setosa_predictions', body=message)
            return {"prediction": "non-setosa, forwarding to Model 2"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Model 1 healthy"}
