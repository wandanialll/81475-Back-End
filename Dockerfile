# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Set environment variables (optional, load .env instead)
ENV PYTHONUNBUFFERED=1

# Expose port 80 for Gunicorn
EXPOSE 80

# Command to run your app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "run:app"]
