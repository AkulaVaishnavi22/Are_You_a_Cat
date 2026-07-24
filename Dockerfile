FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries=5 --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Run your Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]