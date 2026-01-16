FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch to save space
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces serves on port 7860 by default
ENV PORT=7860

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
