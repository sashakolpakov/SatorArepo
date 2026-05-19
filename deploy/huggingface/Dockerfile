FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[web]"

EXPOSE 7860

CMD ["arepo-web", "--host", "0.0.0.0", "--port", "7860"]
