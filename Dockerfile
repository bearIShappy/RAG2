# ==========================
# STAGE 1: The "Builder"
# ==========================
FROM python:3.12.6-slim-bookworm AS builder

# 1. Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. CRITICAL: Create the virtual environment
# Without this line, /opt/venv/bin/pip will NOT exist 
RUN python -m venv /opt/venv

# 3. Add venv to PATH immediately
# This allows you to just type 'pip' instead of '/opt/venv/bin/pip' every time
ENV PATH="/opt/venv/bin:$PATH"

# 4. Copy and Install Dependencies
COPY requirements.txt .
# If you have a local packages folder, uncomment the next line
# COPY stpackages/ ./stpackages/ 

# Because we added venv to PATH above, we can just use 'pip' here
# It will automatically install into /opt/venv
RUN pip install --no-cache-dir -r requirements.txt

# Install the additional package
RUN pip install --no-cache-dir rapidocr_onnxruntime

# ==========================
# STAGE 2: The "Final" Image
# ==========================
FROM python:3.12.6-slim-bookworm

# 1. Install runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy the venv from builder [cite: 4]
COPY --from=builder /opt/venv /opt/venv

# 3. Add venv to PATH in the final image too
ENV PATH="/opt/venv/bin:$PATH"

# 4. Download models
RUN python -m nltk.downloader -q punkt stopwords averaged_perceptron_tagger && \
    python -m spacy download en_core_web_sm && \
    python -m textblob.download_corpora

# 5. Copy app code
COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "app:app"]