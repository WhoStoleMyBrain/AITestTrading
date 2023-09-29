# Use a lightweight Python base image
FROM python:3.9-slim

# Set environment variables for Django
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE tradingbot.settings

# Set the working directory
WORKDIR /app

# Install required packages and libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to start the Django app
CMD ["gunicorn", "tradingbot.wsgi:application", "--bind", "0.0.0.0:8000"]
