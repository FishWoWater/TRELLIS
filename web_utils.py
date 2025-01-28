import os
import redis
import json
from datetime import datetime

# Constants
BASE_IP = "http://21.6.198.96"
BASE_PORT = 6006
BASE_URL = f"{BASE_IP}:{BASE_PORT}" if BASE_PORT else BASE_IP
SERVER_ROOT = "./service"
INPUT_DIR = f"{SERVER_ROOT}/input"
OUTPUT_DIR = f"{SERVER_ROOT}/output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redis keys
QUEUE_KEY = 'trellis:request_queue'
PROCESSING_KEY = 'trellis:processing'
IP_HISTORY_KEY = 'trellis:ip_history'
WORKER_TYPE_KEY = 'trellis:worker_type'
WORKER_COUNT_KEY = 'trellis:worker_count'

IP_HISTORY_LIMIT = 10

# Worker types
AI_WORKER = 'ai_worker'
MAIN_WORKER = 'main_worker'

# Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_ip_output_dir(ip_address):
    """Get the output directory for a specific IP address"""
    ip_dir = os.path.join(OUTPUT_DIR, ip_address.replace(':', '_'))
    os.makedirs(ip_dir, exist_ok=True)
    return ip_dir
