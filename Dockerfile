FROM python:3.11-slim

# build tools for lightgbm/catboost wheels; git for the champion store
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sinus.py sinus_search.py sinus_daemon.py sinus_gitstore.py railway_daemon.py ./

ENV PYTHONUNBUFFERED=1 SINUS_VOLUME=/data SINUS_NODE=railway

CMD ["python", "railway_daemon.py"]
