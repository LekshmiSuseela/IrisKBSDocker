FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model into container
COPY app.py .
COPY model.joblib .

# Expose port 5000 for Flask
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
