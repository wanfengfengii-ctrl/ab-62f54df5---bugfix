# Pure-backend exact CAM cutter-compensation audit service.
# Stdlib-only geometry kernel; the image only carries the ASGI runtime.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the application.
COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default bind (can be overridden at run time / via compose).
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    APP_WORKERS=2 \
    APP_LOG_LEVEL=info

EXPOSE 8080

# Container-level health check hitting the exact-information health route.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import json,os,urllib.request,sys; \
p=int(os.environ.get('APP_PORT','8080')); \
r=urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3); \
sys.exit(0 if json.load(r).get('status')=='ok' else 1)"

# Shell form expands APP_* environment variables at container start.
CMD uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT} \
    --workers ${APP_WORKERS} --log-level ${APP_LOG_LEVEL}
