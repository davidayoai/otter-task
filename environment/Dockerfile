# CPU-only default base image. For GPU tasks, switch to a CUDA base image and
# pin the CUDA build of the ML framework in the lock file.
FROM python:3.12-slim@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates=20230311+deb12u1 \
    procps=2:4.0.2-3 \
    ripgrep=3.8-1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

COPY task_inputs/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && (test -f /app/data/data.csv || test -f /app/data/input_data.csv || test -f /app/data/input_data.npz)

ENV OTTER_DATA_PATH=/app/data/data.csv
