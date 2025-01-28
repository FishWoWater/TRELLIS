import os
import logging
import time 
import threading
import torch
import json
from datetime import datetime
from PIL import Image
from web_utils import *
from gunicorn.app.base import BaseApplication
from gunicorn.workers.gthread import ThreadWorker

# TODO: remove dependency of AI_WORKER on gunicorn 

# Worker configurations
NUM_GPUS = 1    # Edit manually 
NUM_AI_WORKERS = NUM_GPUS * 1   # Adjust the multiplier to have multi-workers per-gpu

# Set up logging
logging.basicConfig(level=logging.INFO)

LOW_VRAM = True 
USE_GPU = True
pipeline = None

def init_pipeline(gpu_id=None):
    global pipeline
    try:
        if pipeline is None:
            logging.info("Ready to initialize the pipeline.")
            try:
                logging.info("Loading pretrained model...")
                from trellis.pipelines import TrellisImageTo3DPipeline
                pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading pretrained model: {str(e)}")
                raise

            logging.info(f"Setting pipeline parameters: low_vram={LOW_VRAM}, use_gpu={USE_GPU}, gpu_id={gpu_id}")
            pipeline.low_vram = LOW_VRAM
            
            if USE_GPU and not LOW_VRAM and gpu_id is not None:
                try:
                    logging.info(f"Moving pipeline to GPU {gpu_id}")
                    pipeline.cuda()
                    logging.info("Pipeline successfully moved to GPU")
                except Exception as e:
                    logging.error(f"Error moving pipeline to GPU {gpu_id}: {str(e)}")
                    raise
            else:
                logging.info("Pipeline will run on CPU")
    except Exception as e:
        logging.error(f"Pipeline initialization failed: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise

def voxelize(mesh_path: str, resolution: int = 64):
    # don't place open3d elsewhere, it's torch initialize will break the multiprocess convention 
    import open3d as o3d 
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh,
                                                                            voxel_size=1 / resolution,
                                                                            min_bound=(-0.5, -0.5, -0.5),
                                                                            max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel

def process_queue():
    """Background worker to process queued requests"""
    while True:
        try:
            # Get next request from queue
            request_data = redis_client.rpop(QUEUE_KEY)
            if not request_data:
                time.sleep(0.3)  # Wait if queue is empty
                continue

            request_data = json.loads(request_data)
            request_id = request_data['request_id']

            # Mark request as processing
            redis_client.hset(
                PROCESSING_KEY, request_id,
                json.dumps({
                    'start_time': datetime.now().isoformat(),
                    'image_name': request_data.get('image_name', ''),
                    'status': 'processing',
                    'worker_pid': os.getpid()
                }))

            try:
                process_single_request(request_data)
                # Update status to complete
                redis_client.hset(
                    PROCESSING_KEY, request_id,
                    json.dumps({
                        'status': 'complete',
                        'image_name': request_data.get('image_name', ''),
                        'finish_time': datetime.now().isoformat()
                    }))
            except Exception as e:
                logging.error(f"Error processing request {request_id}: {str(e)}")
                redis_client.hset(
                    PROCESSING_KEY, request_id,
                    json.dumps({
                        'status': 'error',
                        'error': str(e),
                        'image_name': '',
                        'finish_time': datetime.now().isoformat()
                    }))
        except Exception as e:
            logging.error(f"Error in queue processing: {str(e)}")

def process_single_request(request_data):
    import torch 
    from trellis.utils import render_utils, postprocessing_utils
    """Process a single request from the queue"""
    request_id = request_data['request_id']
    input_path = request_data['input_path']
    mesh_input_path = request_data.get('mesh_input_path', '')
    request_output_dir = request_data['request_output_dir']
    is_dv_mode = request_data['is_dv_mode']

    logging.info(f"Processing request {request_id}")
    image = Image.open(input_path)

    try:
        if is_dv_mode:
            binary_voxel = voxelize(mesh_input_path, resolution=64)
            outputs = pipeline.run_detail_variation(
                binary_voxel,
                image,
                seed=1,
                sparse_structure_sampler_params={
                    "steps": request_data.get('sparse_structure_sample_steps', 12),
                    "cfg_strength": request_data.get('ss_cfg_strength', 7.5),
                },
                slat_sampler_params={
                    "steps": request_data.get('slat_sample_steps', 12),
                    "cfg_strength": request_data.get('slat_cfg_strength', 3.5),
                },
            )
        else:
            outputs = pipeline.run(
                image,
                seed=1,
                sparse_structure_sampler_params={
                    "steps": request_data.get('sparse_structure_sample_steps', 12),
                    "cfg_strength": request_data.get('ss_cfg_strength', 7.5),
                },
                slat_sampler_params={
                    "steps": request_data.get('slat_sample_steps', 12),
                    "cfg_strength": request_data.get('slat_cfg_strength', 3.5),
                },
            )

        torch.cuda.empty_cache()
        # Save outputs
        os.makedirs(request_output_dir, exist_ok=True)

        # By default don't render the video to save memory and time
        if request_data.get('debug', False):
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(request_output_dir, "gs.mp4"), video, fps=30)
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(os.path.join(request_output_dir, "mesh.mp4"), video, fps=30)

        # we should return in the y-up mode
        trimesh_yup = postprocessing_utils.to_trimesh(outputs['gaussian'][0],
                                                      outputs['mesh'][0],
                                                      simplify=request_data.get('simplify_ratio', 0.95),
                                                      texture_size=request_data.get('texture_size', 1024),
                                                      get_srgb_texture=False,
                                                      fill_holes=True,
                                                      texture_bake_mode=request_data.get('texture_bake_mode', 'fast'),
                                                      render_resolution=512,
                                                      debug=False,
                                                      verbose=True)

        # Save mesh
        mesh_path = os.path.join(request_output_dir, "output.glb")
        trimesh_yup.export(mesh_path)

    except Exception as e:
        logging.error(f"Error processing request {request_id}: {str(e)}")
        raise

class AIWorker(ThreadWorker):
    def __init__(self, *args, **kwargs):
        print('AIWorker __init__ called')
        super().__init__(*args, **kwargs)
        self.worker_type = None
        self.gpu_id = None
        self._pipeline_initialized = False

    def init_process(self):
        print('Entering init_process')
        try:
            # Initialize worker type before parent initialization
            if hasattr(self.app, 'worker_init'):
                print('Calling worker_init')
                self.app.worker_init(self)
                logging.info(f"Worker {os.getpid()} initialized as {self.worker_type}")
            else:
                logging.error("Application missing worker_init method")
                return
            super().init_process()
        except Exception as e:
            logging.error(f"Error in worker init_process: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise

class AIApplication(BaseApplication):
    def __init__(self, options=None):
        self.options = options or {}
        # Initialize Redis counters
        redis_client.hset(WORKER_COUNT_KEY, AI_WORKER, 0)
        logging.info(f"Initializing AIApplication with {NUM_AI_WORKERS} AI workers")
        logging.info(f"Detected {NUM_GPUS} GPUs")
        super().__init__()

    def load_config(self):
        config = {
            key: value for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return lambda: None  # Dummy WSGI application since we don't handle web requests

    def init_worker_with_type(self, worker, gpu_id=None):
        try:
            worker.worker_type = AI_WORKER
            worker.gpu_id = gpu_id
            redis_client.hset(WORKER_TYPE_KEY, str(os.getpid()), AI_WORKER)
            count = redis_client.hincrby(WORKER_COUNT_KEY, AI_WORKER, 1)
            logging.info(f"Initializing AI worker {os.getpid()} with GPU {gpu_id}")
            logging.info(f"Current AI workers: {count}/{NUM_AI_WORKERS}")
        except Exception as e:
            logging.error(f"Error initializing worker: {str(e)}")
            raise

    def worker_init(self, worker):
        try:
            gpu_id = (int(redis_client.hget(WORKER_COUNT_KEY, AI_WORKER) or 0)) % NUM_GPUS
            self.init_worker_with_type(worker, gpu_id)
        except Exception as e:
            logging.error(f"Error in worker_init: {str(e)}")
            raise

def post_worker_init(worker):
    try:
        # Initialize CUDA context
        torch.cuda.init()
        torch.cuda.set_device(worker.gpu_id)
        logging.info(f"CUDA initialization successful. Device count: {torch.cuda.device_count()}")
        
        try:
            # Initialize pipeline
            init_pipeline(gpu_id=worker.gpu_id)
            logging.info("Pipeline initialized successfully")
            
            # Start the queue processing thread
            logging.info(f"Starting queue processing thread for AI worker {os.getpid()}")
            processing_thread = threading.Thread(target=process_queue, daemon=True)
            processing_thread.start()
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise
        
    except Exception as e:
        logging.error(f"Critical error in post_fork: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise

if __name__ == '__main__':
    # Clear any existing worker types and counts from Redis
    redis_client.delete(WORKER_TYPE_KEY)
    redis_client.delete(WORKER_COUNT_KEY)
    
    options = {
        'bind': '127.0.0.1:6007',  # Different port for AI workers
        'workers': NUM_AI_WORKERS,
        'worker_class': 'ai_worker.AIWorker',
        'threads': 1,
        'worker_tmp_dir': '/dev/shm',
        'timeout': 300,
        'post_worker_init': post_worker_init
    }
    
    logging.info(f"Starting AI server with {NUM_AI_WORKERS} workers")
    logging.info(f"GPU Configuration: {NUM_GPUS} GPUs detected")
    AIApplication(options).run()
