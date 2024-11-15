# Use Python 3.10 slim version as base image
FROM python:3.10-slim-buster

# Update package manager and install AWS CLI
RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy all project files from the current directory to the container's /app directory
COPY . /app

# Install Python dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Run the application
CMD ["python3", "app.py"]
