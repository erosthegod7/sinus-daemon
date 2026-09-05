
FROM python:3.11-slim
 
# build tools for lightgbm/catboost wheels; git for the champion store
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# sinus_train.py is not just the laptop trainer: it owns the feature installs (horizons,
# candle + ATM blocks) and the chain builder the champion was fitted on. Serving without it
# produces a feature matrix the scaler has to pad with NaN — the model runs, and is wrong.
COPY sinus.py sinus_search.py sinus_daemon.py sinus_gitstore.py railway_daemon.py sinus_inbox.py \
     sinus_train.py sinus_chain_loader.py polygon_chain_history.py sinus_magnitude.py ./
 
ENV PYTHONUNBUFFERED=1 SINUS_VOLUME=/data SINUS_NODE=railway
 
# default entrypoint is the search daemon; the inbox service overrides this with
# `python sinus_inbox.py` in its Railway start command
CMD ["python", "railway_daemon.py"]
 
