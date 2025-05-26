# Stage 1: Builder
# This stage installs all system dependencies and Python packages
FROM python:3.10-slim-buster AS builder

# Install system dependencies required for building Python packages and runtime
# Keep --no-install-recommends for smaller image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Core build tools
        build-essential \
        python3-dev \
        curl \
        # Poppler and related for PDF processing
        poppler-utils \
        # libmagic for file type detection
        libmagic1 \
        # Tesseract OCR and English language data
        tesseract-ocr \
        tesseract-ocr-eng \
        # OpenCV/graphics libraries for unstructured's image processing
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
        # Modern SQLite3 for ChromaDB
        libsqlite3-dev \
        # Clean up apt caches to minimize layer size
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the builder stage
WORKDIR /app

# Upgrade pip first (good practice)
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt to leverage Docker cache for dependency installation
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# This step might take a long time due to complex dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by unstructured
# These will be downloaded into the builder stage's filesystem.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('words')"


# Stage 2: Runtime
# This stage creates the final, lean image
FROM python:3.10-slim-buster

# Install only the *runtime* system dependencies.
# Many of the `*-dev` packages and `build-essential` are not needed here.
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
        # NOTE: libsqlite3-dev is NOT needed in runtime, as pysqlite3-binary bundles its own,
        # and the built Python packages from builder stage contain necessary links.
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the runtime stage
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy NLTK data from the builder stage
COPY --from=builder /root/nltk_data /root/nltk_data

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