# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Expose the port gunicorn will run on
EXPOSE 8080

# Define the command to run your app using Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:server"]
