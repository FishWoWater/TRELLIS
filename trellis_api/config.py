"""
Configuration constants for the TRELLIS API
"""
import os

# Server configuration
BASE_IP = "http://21.6.198.96"
BASE_PORT = 6006
BASE_URL = f"{BASE_IP}:{BASE_PORT}" if BASE_PORT else BASE_IP
# Directory to place the `service` will be the parent of THIS file
SERVER_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "service")
INPUT_DIR = f"{SERVER_ROOT}/input"
OUTPUT_DIR = f"{SERVER_ROOT}/output"

# Create necessary directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request handling
IP_HISTORY_LIMIT = -1

# Worker types
AI_WORKER = "ai_worker"
MAIN_WORKER = "main_worker"

# GPU Configuration
LOW_VRAM = True
USE_GPU = True
