FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install onnxruntime

COPY main.py .

COPY model.onnx .