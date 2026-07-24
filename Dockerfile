# Step 1: Choose your base Python image
FROM python:3.10-slim

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Copy requirements file into the container
COPY requirements.txt .

# Step 4: Upgrade pip and install packages with increased timeout & retries
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --retries=5 --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your app's code
COPY . .

# Step 6: Command to run your application (adjust filename/command if needed)
CMD ["python", "main.py"]