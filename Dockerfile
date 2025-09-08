# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy the startup script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Expose the port that the Flask application will run on
EXPOSE 5000

# Set the entrypoint to the startup script
ENTRYPOINT ["./start.sh"]
