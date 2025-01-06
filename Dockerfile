# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application file
COPY app.py .

# Copy specific files from the templates folder
COPY templates/index.html ./templates/

# Copy specific model files
COPY models/gru_seq2seq_model.h5 ./models/
COPY models/eng_tokenizer.pkl ./models/
COPY models/fr_tokenizer.pkl ./models/

# Copy the requirements.txt file
COPY requirements.txt .

# Upgrade pip and install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app will run on
EXPOSE 9930

# Command to run the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:9930", "app:app"]