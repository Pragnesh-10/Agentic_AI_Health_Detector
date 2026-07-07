FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Code Engine expects the app to listen on port 8080 by default
ENV PORT=8080

# For free tier, 1 worker is recommended. Scale workers up for paid tiers.
CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT}
