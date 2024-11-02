# Use Miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and Python source file to the container
COPY requirements.txt .
COPY langchainChatbotWithMemory.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80
EXPOSE 80

# Run Streamlit app
CMD ["streamlit", "run", "langchainChatbotWithMemory.py", "--server.port=80", "--server.address=0.0.0.0"]