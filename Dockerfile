# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libomp-dev \
 && rm -rf /var/lib/apt/lists/*


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port if running API (optional)
EXPOSE 8080

# Run the script
CMD ["python", "accident_risk_pred.py"]
