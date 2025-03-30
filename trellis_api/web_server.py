import os
import uuid
import sys
import signal
import logging
import logging.handlers
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gunicorn.app.base import BaseApplication
from gunicorn.workers.gthread import ThreadWorker
from trellis_api.web_utils import (
    create_task,
    get_tasks_for_ip,
    get_ip_output_dir,
    get_processing_tasks,
    get_task_by_id,
)
from trellis_api.models import Task, TaskStatus, TaskType, Session
from trellis_api.config import BASE_URL, INPUT_DIR

# Worker configurations
NUM_MAIN_WORKERS = 2  # Use fixed number of main workers

# Set up enhanced logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

# Configure the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicate logs
for handler in root_logger.handlers[:]: 
    root_logger.removeHandler(handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# Add file handler for web server logs
log_file = os.path.join(os.path.dirname(__file__), 'trellis_web.log')
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Log startup message
logging.info("TRELLIS web server logging initialized")

app = Flask(__name__)
CORS(app)


def get_client_ip():
    """Get the client's real IP address, considering proxy headers"""
    if "X-Forwarded-For" in request.headers:
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    return request.remote_addr


@app.route("/my_requests", methods=["GET"])
def get_my_requests():
    """Get recent requests and their status for the current IP address"""
    client_ip = get_client_ip()

    # Get tasks for this IP
    tasks = get_tasks_for_ip(client_ip)
    requests = []

    for task in tasks:
        # Get output files if request is complete
        output_files = []
        if task["status"] == TaskStatus.COMPLETE.value:
            ip_dir = get_ip_output_dir(client_ip)
            request_dir = os.path.join(ip_dir, task["request_id"])
            if os.path.exists(request_dir):
                output_files = [
                    f"{BASE_URL}/output/{client_ip}/{task['request_id']}/{f}"
                    for f in os.listdir(request_dir)
                ]

        requests.append(
            {
                "request_id": task["request_id"],
                "task_type": task["task_type"], 
                "status": task["status"],
                "start_time": task["start_time"],
                "finish_time": task["finish_time"],
                "output_files": output_files if output_files else None,
                "error": task["error"],
                "image_name": task["image_name"],
                "text": task["text"]
            }
        )

    return jsonify({"ip_address": client_ip, "requests": requests})


@app.route("/output/<ip_address>/<request_id>/<filename>")
def serve_file(ip_address, request_id, filename):
    """Serve files from the IP-specific output directory"""
    ip_dir = get_ip_output_dir(ip_address)
    full_path = os.path.join(ip_dir, request_id, filename)
    app.logger.info(f"Attempting to serve file from: {full_path}")
    return send_from_directory(os.path.join(ip_dir, request_id), filename)


@app.route("/image_to_3d", methods=["POST"])
def image_to_3d():
    """Handle new image to 3D requests with base64 encoded files"""
    try:
        # Get client IP
        client_ip = get_client_ip()
        ip_dir = get_ip_output_dir(client_ip)

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Create request directory
        request_dir = os.path.join(ip_dir, request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Get JSON data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        # Check for required fields
        if "image_data" not in request.json:
            return jsonify({"error": "No image data provided"}), 400

        # Get image name or generate one
        image_name = request.json.get("image_name", f"image_{request_id}.png")
        
        # Decode and save the image file
        try:
            image_data = base64.b64decode(request.json["image_data"])
            input_path = os.path.join(INPUT_DIR, f"{request_id}_{image_name}")
            with open(input_path, 'wb') as f:
                f.write(image_data)
        except Exception as e:
            return jsonify({"error": f"Error decoding image data: {str(e)}"}), 400

        # Handle detail variation mode
        is_dv_mode = request.json.get("is_dv_mode", False)
        mesh_input_path = ""
        
        if is_dv_mode:
            if "mesh_data" not in request.json:
                return jsonify({"error": "No mesh data provided for detail variation mode"}), 400

            try:
                mesh_data = base64.b64decode(request.json["mesh_data"])
                mesh_input_path = os.path.join(INPUT_DIR, f"{request_id}_mesh.glb")
                with open(mesh_input_path, 'wb') as f:
                    f.write(mesh_data)
            except Exception as e:
                return jsonify({"error": f"Error decoding mesh data: {str(e)}"}), 400

        # Create task in database
        request_data = {
            "request_id": request_id,
            "task_type": TaskType.IMAGE_TO_3D.value,
            "input_path": input_path,
            "mesh_input_path": mesh_input_path,
            "request_output_dir": request_dir,
            "is_dv_mode": is_dv_mode,
            "image_name": image_name,
            "ss_sample_steps": int(request.json.get("ss_sample_steps", 12)),
            "ss_cfg_strength": float(request.json.get("ss_cfg_strength", 7.5)),
            "slat_sample_steps": int(request.json.get("slat_sample_steps", 12)),
            "slat_cfg_strength": float(request.json.get("slat_cfg_strength", 3.5)),
        }
        create_task(request_data, client_ip)

        return jsonify(
            {
                "request_id": request_id,
                "status": TaskStatus.QUEUED.value,
                "message": "Request queued successfully",
            }
        )

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/text_to_3d", methods=["POST"])
def text_to_3d():
    """Handle new text to 3D requests with base64 encoded files"""
    try:
        # Get client IP
        client_ip = get_client_ip()
        ip_dir = get_ip_output_dir(client_ip)

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Create request directory
        request_dir = os.path.join(ip_dir, request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Get JSON data
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400

        # Get text input
        if "text" not in request.json:
            return jsonify({"error": "No text input provided"}), 400

        input_text = request.json["text"]
        negative_text = request.json.get("negative_text", "")

        # Handle detail variation mode
        is_dv_mode = request.json.get("is_dv_mode", False)
        mesh_input_path = ""
        
        if is_dv_mode:
            if "mesh_data" not in request.json:
                return jsonify({"error": "No mesh data provided for detail variation mode"}), 400

            try:
                mesh_data = base64.b64decode(request.json["mesh_data"])
                mesh_input_path = os.path.join(INPUT_DIR, f"{request_id}_mesh.glb")
                with open(mesh_input_path, 'wb') as f:
                    f.write(mesh_data)
            except Exception as e:
                return jsonify({"error": f"Error decoding mesh data: {str(e)}"}), 400

        # Create task in database
        request_data = {
            "request_id": request_id,
            "task_type": TaskType.TEXT_TO_3D.value,
            "input_text": input_text,
            "negative_text": negative_text,
            "request_output_dir": request_dir,
            "is_dv_mode": is_dv_mode,
            "mesh_input_path": mesh_input_path,
            "input_path": "",  # Empty for text-to-3D tasks
            "ss_sample_steps": request.json.get("ss_sample_steps", 12),
            "ss_cfg_strength": request.json.get("ss_cfg_strength", 7.5),
            "slat_sample_steps": request.json.get("slat_sample_steps", 12),
            "slat_cfg_strength": request.json.get("slat_cfg_strength", 3.5),
        }
        create_task(request_data, client_ip)

        return jsonify(
            {
                "request_id": request_id,
                "status": TaskStatus.QUEUED.value,
                "message": "Request queued successfully",
            }
        )

    except Exception as e:
        logging.error(f"Error handling request: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "TRELLIS API server is running"})


@app.route("/task/<request_id>", methods=["GET"])
def get_task(request_id):
    """Get task status by request ID"""
    task = get_task_by_id(request_id)

    if task:
        return jsonify(task)

    return jsonify({"error": "Task not found"}), 404


@app.route("/queue_status", methods=["GET"])
def queue_status():
    """Get the current status of the request queue"""
    try:
        # Get queue length (tasks in QUEUED state)
        session = Session()
        queue_length = (
            session.query(Task).filter(Task.status == TaskStatus.QUEUED.value).count()
        )
        session.close()

        # Get all processing tasks
        processing_tasks = get_processing_tasks()
        processing_requests = [
            {
                "request_id": task["request_id"],
                "start_time": task["start_time"],
                "image_name": task["image_name"],
                "worker_pid": task["worker_pid"],
            }
            for task in processing_tasks
        ]

        return jsonify(
            {"queue_length": queue_length, "processing_requests": processing_requests}
        )
    except Exception as e:
        logging.error(f"Error getting queue status: {str(e)}")
        return jsonify({"error": str(e)}), 500


class WebWorker(ThreadWorker):
    """Web server worker thread"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_id = os.getpid()
        logging.info(f"Initializing web worker {self.worker_id}")


class WebApplication(BaseApplication):
    """Simplified web server application"""

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        logging.info(f"Initializing web server with {options.get('workers', 1)} workers")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        super().__init__()
        
    def handle_shutdown(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received shutdown signal {sig}, shutting down gracefully...")
        # Exit with success code after logging
        logging.info("Waiting for workers to complete current requests...")
        sys.exit(0)

    def load_config(self):
        # Directly apply options to configuration
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    options = {
        "bind": "0.0.0.0:6006",
        "workers": NUM_MAIN_WORKERS,
        "worker_class": "trellis_api.web_server.WebWorker",  # Updated to use package path
        "threads": 4,
        "worker_tmp_dir": "/dev/shm",
        "timeout": 300,
        "graceful_timeout": 10,  # Give workers 10 seconds to finish
        "capture_output": True,  # Capture stdout/stderr from workers
        "loglevel": "info",
    }

    try:
        logging.info(f"Starting web server with {NUM_MAIN_WORKERS} workers")
        logging.info(f"Press Ctrl+C to stop the server gracefully")
        WebApplication(app, options).run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logging.error(f"Error running web server: {str(e)}")
        logging.exception("Detailed error:")
    finally:
        logging.info("Web server shutdown complete")
