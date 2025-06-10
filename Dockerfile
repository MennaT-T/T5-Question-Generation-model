FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout 100 --retries 3 \
    torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout 100 --retries 3 \
    transformers==4.36.2 && \
    pip install --no-cache-dir --timeout 100 --retries 3 \
    sentence-transformers==2.2.2 && \
    pip install --no-cache-dir --timeout 100 --retries 3 \
    langchain==0.1.0

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["python", "api.py"] 