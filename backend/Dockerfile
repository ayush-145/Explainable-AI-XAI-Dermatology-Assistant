# Stage 1: Use the official Python image
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy all your application code and the model
COPY . .

# Stage 5: Expose the port the app runs on
EXPOSE 8000

# Stage 6: Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]