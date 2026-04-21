# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencies (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Railway injects PORT env var ─────────────────────────────────────────────
ENV PORT=8000

EXPOSE $PORT

# ── Start ─────────────────────────────────────────────────────────────────────
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
