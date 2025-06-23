# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entry point to run training
CMD ["python", "src/train.py"]
