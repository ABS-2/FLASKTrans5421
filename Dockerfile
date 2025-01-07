# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY app.py .
COPY templates ./templates
COPY models ./models
COPY requirements.txt .

# Upgrade pip and install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 9930

# Command to run the Flask app
CMD ["python", "app.py"]