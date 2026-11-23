FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# The model cache is a Docker volume; create it here so it belongs to the app user.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /home/app

COPY *.py ./
USER app

EXPOSE 8001
CMD ["python", "forecaster.py"]
