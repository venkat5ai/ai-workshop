# Stage 1: Builder
# This stage installs all system dependencies and Python packages required for building and downloading.
FROM python:3.10-slim-buster AS builder

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        curl \
        poppler-utils \
        libmagic1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        tcl tk \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean # Clean apt cache immediately after install for minimal layer size

# Set the working directory for the builder stage
WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt to leverage Docker cache for dependency installation
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by unstructured
# Downloads to /root/nltk_data by default in this environment.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('words')"


# Stage 2: Runtime
# This stage creates the final, lean image with only necessary runtime components.
FROM python:3.10-slim-buster

# Install only the essential runtime system dependencies.
# Development and build tools (e.g., build-essential, python3-dev, libsqlite3-dev) are excluded here.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libmagic1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        tcl tk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean # Clean apt cache immediately after install

# Set the working directory for the runtime stage
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy NLTK data from the builder stage to a more standard NLTK data path
COPY --from=builder /root/nltk_data /usr/local/share/nltk_data/

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy your application code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 3010

# Set the ENTRYPOINT to our script. This makes the container act like an executable.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default CMD (arguments for the ENTRYPOINT).
CMD ["cloud"]