# Start with a Debian-based slim image
FROM python:3.10-slim-buster

# Install Poppler and other system dependencies required by unstructured and other packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        poppler-utils \
        libmagic1 \
        libgomp1 \
        curl \
        libjpeg-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        tcl tk \
        libsqlite3-dev \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libxext6 \
        # ADDED THESE LINES for Tesseract OCR
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip first (good practice)
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt before installing dependencies to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by unstructured (punkt_tab, punkt, averaged_perceptron_tagger_eng, maxent_ne_chunker, words)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy the rest of your application code (agent.py, config.py, templates, etc.)
COPY . .

# Expose the port your Flask app runs on
EXPOSE 3010

# Set the ENTRYPOINT to our script. This makes the container act like an executable.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default CMD (arguments for the ENTRYPOINT).
CMD ["cloud"]