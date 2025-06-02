# Use official Python image
FROM python:3.12

# Set working directory inside the container
WORKDIR /app

# Copy the contents of the app directory into the container's /app directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]