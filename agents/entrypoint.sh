#!/bin/sh

# This script is the ENTRYPOINT for the Docker container. 
# It executes the main Python application with arguments passed from CMD or docker run.

# Execute the main Python application with arguments passed to this script.
# The "$@" expands to all arguments passed to this script (e.g., "cloud" or "local").
# The -u flag ensures unbuffered output for Python, which is good for Docker logs.
exec python -u agent.py "$@"