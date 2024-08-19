FROM python:3.11-slim

RUN mkdir app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src .

EXPOSE 8000
CMD ["uvicorn","src/academy:app","--host","0.0.0.0","-p","8000"]

#in cmd , to build the image
#docker build -t academy-demo -f Dockerfile .

#docker run academy-demo