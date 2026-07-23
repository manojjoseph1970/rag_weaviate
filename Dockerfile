# FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1 \
#     HF_HOME=/models/huggingface
#     PORT=8080
#     HOME=/home/app

# WORKDIR /app

# #RUN addgroup --system app && adduser --system --ingroup app app
# RUN addgroup --system --gid 10001 app && \
#     adduser --system \
#         --uid 10001 \
#         --ingroup app \
#         --home /home/app \
#         app && \
#     mkdir -p /home/app /models/huggingface && \
#     chown -R 10001:10001 /home/app /models

# COPY requirements.txt .
# RUN python -m pip install --upgrade pip && \
#     python -m pip install --no-cache-dir -r requirements.txt
# RUN mkdir -p /models/huggingface
# # checking the packages and if missing it will fail
# RUN python -c "import fastapi, uvicorn, sentence_transformers"

# # Download the embedding model at build time. This prevents every pod from
# # downloading it during startup and makes deployments reproducible.
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# # consider offline mode in production so pods never unexpectedly access the internet:
# ENV HF_HUB_OFFLINE=1 \
#     TRANSFORMERS_OFFLINE=1

# COPY app ./app

# # RUN groupmod -g 10001 app && \
# #     usermod -u 10001 -g 10001 app && \
# #     chown -R 10001:10001 /app /models


# USER 10001:10001
# #USER app

# EXPOSE 8080

########### modified docker file ##############################

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HOME=/home/app \
    PORT=8080

WORKDIR /app

# Create the runtime user with the final UID and GID.
RUN addgroup --system --gid 10001 app && \
    adduser --system \
        --uid 10001 \
        --ingroup app \
        --home /home/app \
        app && \
    mkdir -p /models/huggingface /home/app && \
    chown -R 10001:10001 /models /home/app /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -c "import fastapi, uvicorn, sentence_transformers"

# Download the embedding model into the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Prevent runtime attempts to download the model.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=10001:10001 app ./app

RUN chown -R 10001:10001 /models

USER 10001:10001

EXPOSE ${PORT}

#CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]


CMD ["sh", "-c", "python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]



