FROM python:3.10-slim

# libgomp1 is needed by tensorflow-cpu's OpenMP runtime; slim images omit it
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker can cache this layer across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (Fundus_diseases dataset is excluded via .dockerignore)
COPY . .

# Hugging Face Spaces runs containers as a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Single worker: tensorflow loads a full model copy per worker process,
# and the free HF tier's RAM budget doesn't need more than one for this load.
CMD ["gunicorn", "app1:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
