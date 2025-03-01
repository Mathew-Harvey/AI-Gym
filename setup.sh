#!/bin/bash

# Ensure the script stops on errors
set -e

# Upgrade pip (good practice)
pip install --upgrade pip

# Install required Python packages
pip install -r requirements.txt

# (Optional) If OpenCV needs extra system dependencies:
# sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx

# Set environment variables if needed
export STREAMLIT_SERVER_PORT=10000
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
