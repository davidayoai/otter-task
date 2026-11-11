# CPU-only default base image. For GPU tasks, switch to a CUDA base image and
# pin the CUDA build of the ML framework in the lock file.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df


ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-downgrades \
    ca-certificates=20230311+deb12u1 \
    procps=2:4.0.2-3 \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.lock /tmp/requirements.lock
RUN pip install --require-hashes --no-deps -r /tmp/requirements.lock

COPY task_inputs/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && (test -f /app/data/data.csv || test -f /app/data/input_data.csv || test -f /app/data/input_data.npz)

ENV OTTER_DATA_PATH=/app/data/data.csv
