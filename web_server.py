import os
import uuid
import logging
import json
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gunicorn.app.base import BaseApplication
from gunicorn.workers.gthread import ThreadWorker
from web_utils import *

# Worker configurations
NUM_MAIN_WORKERS = 2  # Use fixed number of main workers

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

def get_client_ip():
    """Get the client's real IP address, considering proxy headers"""
    if 'X-Forwarded-For' in request.headers:
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    return request.remote_addr

@app.route('/my_requests', methods=['GET'])
def get_my_requests():
    """Get recent requests and their status for the current IP address"""
    client_ip = get_client_ip()

    # Get list of request IDs for this IP
    request_ids = redis_client.lrange(f"{IP_HISTORY_KEY}:{client_ip}", 0, -1)
    requests = []

    for request_id in request_ids:
        request_id = request_id.decode('utf-8')
        # Get request status
        status = redis_client.hget(PROCESSING_KEY, request_id)
        if status:
            status = json.loads(status)
        else:
            # Check if it's still in queue
            if request_id.encode() in redis_client.lrange(QUEUE_KEY, 0, -1):
                status = {'status': 'queued'}
            else:
                continue  # Skip if no status found

        # Get output files if request is complete
        output_files = []
        if status.get('status') == 'complete':
            ip_dir = get_ip_output_dir(client_ip)
            request_dir = os.path.join(ip_dir, request_id)
            if os.path.exists(request_dir):
                output_files = [f"{BASE_URL}/output/{client_ip}/{request_id}/{f}" for f in os.listdir(request_dir)]

        requests.append({
            'request_id': request_id,
            'status': status.get('status'),
            'start_time': status.get('start_time'),
            'finish_time': status.get('finish_time'),
            'output_files': output_files if output_files else None,
            'error': status.get('error'),
            'image_name': status.get('image_name', '')
        })

    return jsonify({'ip_address': client_ip, 'requests': requests})

@app.route('/output/<ip_address>/<request_id>/<filename>')
def serve_file(ip_address, request_id, filename):
    """Serve files from the IP-specific output directory"""
    ip_dir = get_ip_output_dir(ip_address)
    return send_from_directory(os.path.join(ip_dir, request_id), filename)

@app.route('/image_to_3d', methods=['POST'])
def image_to_3d():
    """Handle new image to 3D requests"""
    try:
        # Get client IP
        client_ip = get_client_ip()
        ip_dir = get_ip_output_dir(client_ip)

        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create request directory
        request_dir = os.path.join(ip_dir, request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Get image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image file selected'}), 400

        # Save image file
        input_path = os.path.join(INPUT_DIR, f"{request_id}_{image_file.filename}")
        image_file.save(input_path)

        # Handle detail variation mode
        is_dv_mode = request.form.get('is_dv_mode', 'false').lower() == 'true'
        mesh_input_path = ''
        if is_dv_mode:
            if 'mesh' not in request.files:
                return jsonify({'error': 'No mesh file provided for detail variation mode'}), 400
            
            mesh_file = request.files['mesh']
            if not mesh_file.filename:
                return jsonify({'error': 'No mesh file selected for detail variation mode'}), 400

            mesh_input_path = os.path.join(INPUT_DIR, f"{request_id}_{mesh_file.filename}")
            mesh_file.save(mesh_input_path)

        # Prepare request data
        request_data = {
            'request_id': request_id,
            'input_path': input_path,
            'mesh_input_path': mesh_input_path,
            'request_output_dir': request_dir,
            'is_dv_mode': is_dv_mode,
            'image_name': image_file.filename,
            # Optional parameters
            'ss_sample_steps': int(request.form.get('ss_sample_steps', 25)),
            'ss_cfg_strength': float(request.form.get('ss_cfg_strength', 8.5)),
            'slat_sample_steps': int(request.form.get('slat_sample_steps', 25)),
            'slat_cfg_strength': float(request.form.get('slat_cfg_strength', 4.0)),
        }

        # Add to processing queue
        redis_client.lpush(QUEUE_KEY, json.dumps(request_data))
        
        # Add to IP history
        redis_client.lpush(f"{IP_HISTORY_KEY}:{client_ip}", request_id)
        redis_client.ltrim(f"{IP_HISTORY_KEY}:{client_ip}", 0, IP_HISTORY_LIMIT - 1)

        return jsonify({
            'request_id': request_id,
            'status': 'queued',
            'message': 'Request queued successfully'
        })

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/queue_status', methods=['GET'])
def queue_status():
    """Get the current status of the request queue"""
    try:
        # Get queue length
        queue_length = redis_client.llen(QUEUE_KEY)
        
        # Get all processing requests
        processing = redis_client.hgetall(PROCESSING_KEY)
        processing_requests = []
        
        for request_id, status in processing.items():
            status = json.loads(status)
            if status.get('status') == 'processing':
                processing_requests.append({
                    'request_id': request_id.decode('utf-8'),
                    'start_time': status.get('start_time'),
                    'image_name': status.get('image_name', ''),
                    'worker_pid': status.get('worker_pid')
                })

        return jsonify({
            'queue_length': queue_length,
            'processing_requests': processing_requests
        })
    except Exception as e:
        logging.error(f"Error getting queue status: {str(e)}")
        return jsonify({'error': str(e)}), 500

class WebWorker(ThreadWorker):
    def __init__(self, *args, **kwargs):
        print('WebWorker __init__ called')
        super().__init__(*args, **kwargs)
        self.worker_type = MAIN_WORKER

class WebApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        # Initialize Redis counter for main workers
        redis_client.hset(WORKER_COUNT_KEY, MAIN_WORKER, 0)
        logging.info(f"Initializing WebApplication with {NUM_MAIN_WORKERS} main workers")
        super().__init__()

    def load_config(self):
        config = {
            key: value for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

    def init_worker_with_type(self, worker):
        try:
            redis_client.hset(WORKER_TYPE_KEY, str(os.getpid()), MAIN_WORKER)
            count = redis_client.hincrby(WORKER_COUNT_KEY, MAIN_WORKER, 1)
            logging.info(f"Initializing main worker {os.getpid()}")
            logging.info(f"Current main workers: {count}/{NUM_MAIN_WORKERS}")
        except Exception as e:
            logging.error(f"Error initializing worker: {str(e)}")
            raise

    def worker_init(self, worker):
        try:
            self.init_worker_with_type(worker)
        except Exception as e:
            logging.error(f"Error in worker_init: {str(e)}")
            raise

if __name__ == '__main__':
    # Clear existing main worker counts from Redis
    redis_client.hdel(WORKER_COUNT_KEY, MAIN_WORKER)
    
    options = {
        'bind': '0.0.0.0:6006',
        'workers': NUM_MAIN_WORKERS,
        'worker_class': 'web_server.WebWorker',
        'threads': 1,
        'worker_tmp_dir': '/dev/shm',
        'timeout': 300,
    }
    
    logging.info(f"Starting web server with {NUM_MAIN_WORKERS} main workers")
    WebApplication(app, options).run()
