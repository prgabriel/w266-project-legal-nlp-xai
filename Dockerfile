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

# Copy application code (including models via LFS)
COPY . .

# Verify models are present
RUN ls -la models/bert/ && echo "BERT model files found" || echo "BERT model files missing"
RUN ls -la models/t5/ && echo "T5 model files found" || echo "T5 model files missing"

# Create additional directories if needed
RUN mkdir -p logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]