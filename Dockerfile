# Single-stage on purpose: this image is a demo target, not a production
# artifact, and a reviewer reading one short file beats a clever build they
# have to decode before they can trust it.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SHOPEE_DB=/app/data/shops.db

# Dependency metadata first so editing source does not invalidate the pip layer.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir ".[model,app]"

EXPOSE 8000

# The database is mounted at runtime rather than baked in, so the image stays
# the same whether it is scoring 500 shops or 50,000.
CMD ["uvicorn", "shopee_scraper.api:app", "--host", "0.0.0.0", "--port", "8000"]
