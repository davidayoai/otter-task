
# CPU-only default base image
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-downgrades \
    ca-certificates=20230311+deb12u1 \
    procps=2:4.0.2-3 \
    ripgrep=13.0.0-4+b2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock

RUN pip install --require-hashes --no-deps -r /tmp/requirements.lock

# Copy the actual dataset
COPY task_inputs/data/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && test -f /app/data/data.csv

ENV OTTER_DATA_PATH=/app/data/data.csv
