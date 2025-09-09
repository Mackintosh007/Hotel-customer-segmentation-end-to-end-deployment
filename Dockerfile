# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
# Using `python -m pip` ensures packages are installed for this Python interpreter
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Flask application will run on
EXPOSE 5000

# Define the command to run the Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
