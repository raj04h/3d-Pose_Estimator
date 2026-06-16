# Base Image
FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Copy Requirements
COPY requirements.txt .

# Install Dependencies
RUN pip install --upgrade pip

RUN pip install \
    --default-timeout=1000 \
    --no-cache-dir \
    -r requirements.txt

# Copy Project
COPY . .

# Streamlit Port
EXPOSE 8501

# Start App
CMD ["python", "-m", "streamlit", "run", "Web_view/app_web.py", "--server.address=0.0.0.0"]