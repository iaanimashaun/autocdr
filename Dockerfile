# Use an appropriate base image
# FROM python:3.8-slim
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


RUN apt update

RUN apt-get install -y python3 python3-pip


# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the rest of the application files
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Start the Flask app
CMD ["python3", "app.py"]


