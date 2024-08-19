# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the necessary dependencies
RUN pip install --no-cache-dir scikit-learn prometheus_client

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the app
CMD ["python", "model.py"]
