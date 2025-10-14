# ===========================
# Stage 1: Build dependencies
# ===========================
FROM python:3.11-slim-bookworm AS builder

# Set up working directory
WORKDIR /app

# Install build tools (only for this stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list early for caching
COPY requirements.txt .

# Install dependencies with prebuilt PyTorch wheel (CPU-only)
RUN pip install --no-cache-dir -r requirements.txt


# ===========================
# Stage 2: Runtime image
# ===========================
FROM python:3.11-slim-bookworm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PATH="/home/ctfuser/.local/bin:$PATH"

# Working directory
WORKDIR /app

# Copy installed packages from builder (much smaller than reinstalling)
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY . .

# Make entrypoint executable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Create a non-root user for security
RUN useradd -m -u 1000 ctfuser && chown -R ctfuser:ctfuser /app
USER ctfuser

# Expose Flask port
EXPOSE 5000

# Health check (using curl installed in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Entry point
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]

