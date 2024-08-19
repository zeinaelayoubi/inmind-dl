# model2.py
from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import psycopg2
import pika
import os
import json 
import time

# Load Iris dataset and train Model 2 (Versicolor vs Virginica)
iris = load_iris()
X = iris.data[iris.target != 0]  # Only non-setosa samples
y = iris.target[iris.target != 0] - 1  # Versicolor (0) vs Virginica (1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)

# Initialize FastAPI app for Model 2
app = FastAPI()

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        database=os.getenv("POSTGRES_DB", "iris_db"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password")
    )
    return conn

# RabbitMQ connection for receiving predictions from Model 1
def get_rabbitmq_connection():
    connection = None
    for i in range(10):  # Try to connect up to 10 times
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOST", "localhost"), heartbeat=60)
            )
            print("Connected to RabbitMQ")
            break
        except pika.exceptions.AMQPConnectionError as e:
            print(f"RabbitMQ connection failed on attempt {i + 1}, retrying...")
            time.sleep(5)  # Wait for 5 seconds before retrying
    if connection is None:
        raise Exception("Failed to connect to RabbitMQ after multiple attempts")
    return connection

# Background task to process messages from RabbitMQ
def process_message(ch, method, properties, body):
    features = json.loads(body)
    data = np.array([[features['sepal_length'], features['sepal_width'], features['petal_length'], features['petal_width']]])
    prediction = model2.predict(data)
    features['prediction'] = iris.target_names[prediction + 1][0]
    log_prediction_to_db(features)
    print(f"Model 2 prediction: {features['prediction']}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Function to log predictions to the PostgreSQL database
def log_prediction_to_db(data: dict):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO public.predictions (sepal_length, sepal_width, petal_length, petal_width, prediction) VALUES (%s, %s, %s, %s, %s)",
            (data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width'], data['prediction'])
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Prediction logged successfully.")
        return True
    except Exception as e:
        print(f"Failed to log prediction: {e}")
        return False


# Start consuming messages from RabbitMQ
@app.on_event("startup")
def startup_event():
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    channel.queue_declare(queue='non_setosa_predictions')
    channel.basic_consume(queue='non_setosa_predictions', on_message_callback=process_message)
    print("Started consuming messages from RabbitMQ...")
    channel.start_consuming()

        
# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Model 2 healthy"}
