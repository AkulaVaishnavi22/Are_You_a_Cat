# 1. Base lightweight Python image
FROM python:3.10-slim

# 2. Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set workspace directory inside container
WORKDIR /app

# 4. Install updated system dependencies (using libgl1 instead of libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all project files into the container
COPY . /app

# 7. Expose Streamlit port
EXPOSE 8501

# 8. Command to start Streamlit bound to port 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]