FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY router/ ./router/
COPY main.py .

# The harness mounts /input and /output; ensure they exist so a missing
# mount doesn't crash us before we even read the tasks file.
RUN mkdir -p /input /output

ENTRYPOINT ["python", "main.py"]
