FROM python:3.12-slim-bookworm AS runtime
ARG SOURCE_REPOSITORY
ARG SOURCE_REVISION
ARG SOURCE_TREE_SHA256
LABEL org.opencontainers.image.source=$SOURCE_REPOSITORY \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      tech.cybe.digital-school.source-tree-sha256=$SOURCE_TREE_SHA256

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVER_PORT=8082

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER app

EXPOSE 8082

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${SERVER_PORT} --no-access-log"]
