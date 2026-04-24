FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# models folder starts empty — main.py downloads them on first start
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app_api", "--host", "0.0.0.0", "--port", "8000"]
