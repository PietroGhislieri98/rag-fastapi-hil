FROM python:3.11-bookworm

# Ambiente
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Toolchain minima per eventuali wheel native
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Reqs
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# App
COPY server.py ./

EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
