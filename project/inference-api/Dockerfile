# Use a valid CUDA base image with Python support
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get clean

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
