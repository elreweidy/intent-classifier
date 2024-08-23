# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages and download NLTK data
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt', quiet=True, download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', quiet=True, download_dir='/usr/local/share/nltk_data')"

# Copy the current directory contents into the container
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the command to start the Flask app
CMD ["python", "app.py"]