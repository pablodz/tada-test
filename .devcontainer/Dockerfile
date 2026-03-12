FROM python:3.12

RUN rm -f /etc/apt/sources.list.d/yarn.list* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*