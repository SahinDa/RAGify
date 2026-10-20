FROM python:3.12-slim

# 1. Optimize Python runtime for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# 2. Security: Create a dedicated non-root system user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

# 3. Layer Caching: Install dependencies first before copying full code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy source code and transfer ownership to the non-root user
COPY --chown=appuser:appgroup . .

# 5. Drop root privileges
USER appuser

EXPOSE 8000

# 6. Start server: dynamically binds to $PORT (assigned by Render/cloud) with 8000 fallback
# Use 1 worker for free tiers (512MB RAM) to prevent OOM termination; increase for paid multi-core tiers
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]