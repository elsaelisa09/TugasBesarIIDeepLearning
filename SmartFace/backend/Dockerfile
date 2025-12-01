# Use Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY attendance.json .
COPY .env.example .

# Create model cache directory
RUN mkdir -p model_cache

# Expose port
EXPOSE 7860

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=7860
ENV HOST=0.0.0.0

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "--preload"]
