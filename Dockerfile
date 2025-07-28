# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Set PROJECT_ROOT environment variable
ENV PROJECT_ROOT=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory and download base models if needed
RUN mkdir -p models/bert models/t5

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]