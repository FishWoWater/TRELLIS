import os
import sys
import logging
import time
import threading
import enum
import torch
import numpy as np
import imageio
import open3d as o3d
from PIL import Image
from trellis_api.web_utils import get_next_task, update_task_status
from trellis_api.models import TaskStatus, TaskType
from trellis_api.config import LOW_VRAM, USE_GPU

# Worker configurations
NUM_GPUS = 1  # Edit manually

# Worker allocation per GPU
IMAGE_WORKERS_PER_GPU = 1  # Number of image-to-3D workers per GPU
TEXT_WORKERS_PER_GPU = 1  # Number of text-to-3D workers per GPU

# Total worker counts
NUM_IMAGE_WORKERS = NUM_GPUS * IMAGE_WORKERS_PER_GPU
NUM_TEXT_WORKERS = NUM_GPUS * TEXT_WORKERS_PER_GPU

# Set up logging with more detailed configuration
import logging.handlers

# Create a formatter that includes thread information
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

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

# Optionally add a file handler to keep logs persistent
log_file = os.path.join(os.path.dirname(__file__), 'trellis_worker.log')
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Log startup message
logging.info("TRELLIS AI worker logging initialized")


class PipelineType(enum.Enum):
    IMAGE = "image"
    TEXT = "text"


class Pipeline:
    def __init__(self, pipeline_type: PipelineType, gpu_id=None):
        self.type = pipeline_type
        self.pipeline = None
        self.gpu_id = gpu_id
        self.init_pipeline()

    def init_pipeline(self):
        try:
            if self.type == PipelineType.IMAGE:
                logging.info("Loading image-to-3D model...")
                from trellis.pipelines import TrellisImageTo3DPipeline

                self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                    "JeffreyXiang/TRELLIS-image-large"
                )
                logging.info("Image-to-3D model loaded successfully.")
            else:
                logging.info("Loading text-to-3D model...")
                from trellis.pipelines import TrellisTextTo3DPipeline

                self.pipeline = TrellisTextTo3DPipeline.from_pretrained(
                    "./pretrained/TRELLIS-text-xlarge"
                )
                logging.info("Text-to-3D model loaded successfully.")

            self.pipeline.low_vram = LOW_VRAM

            if USE_GPU and not LOW_VRAM and self.gpu_id is not None:
                logging.info(f"Moving {self.type.value} pipeline to GPU {self.gpu_id}")
                self.pipeline.cuda()
                logging.info("Pipeline successfully moved to GPU")
            else:
                logging.info(f"{self.type.value} pipeline will run on CPU")

        except Exception as e:
            logging.error(f"Error initializing {self.type.value} pipeline: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise

    def run(self, *args, **kwargs):
        return self.pipeline.run(*args, **kwargs)

    def run_detail_variation(self, *args, **kwargs):
        if self.type == PipelineType.IMAGE:
            return self.pipeline.run_detail_variation(*args, **kwargs)
        elif self.type == PipelineType.TEXT:
            return self.pipeline.run_variant(*args, **kwargs)
        raise NotImplementedError("Detail variation only supported for image pipeline")


_pipeline = None


def init_pipeline(gpu_id=None, pipeline_type=None):
    """Initialize a pipeline of the specified type"""
    global _pipeline
    if pipeline_type is None:
        raise ValueError("Pipeline type must be specified")
    logging.info(
        f"Initializing pipeline for type {pipeline_type.value} on GPU {gpu_id}"
    )
    _pipeline = Pipeline(pipeline_type, gpu_id)


def voxelize(mesh_path: str, resolution: int = 64):
    # don't place open3d elsewhere, it's torch initialize will break the multiprocess convention
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel


def process_queue(pipeline_type: PipelineType, worker_id=None, gpu_id=None):
    """Background worker to process queued requests"""
    # Import sys if not already imported
    import sys
    
    task_type = (
        TaskType.IMAGE_TO_3D
        if pipeline_type == PipelineType.IMAGE
        else TaskType.TEXT_TO_3D
    )
    
    # Get current thread name and process ID
    thread_name = threading.current_thread().name
    process_id = os.getpid()
    
    worker_info = f"{pipeline_type.value} worker {worker_id} on GPU {gpu_id}"
    logging.info(f"[Thread: {thread_name}] [PID: {process_id}] {worker_info} started processing queue")
    
    # Flush stdout to ensure logs are visible immediately
    sys.stdout.flush()

    while True:
        try:
            # Get next task from queue matching our pipeline type
            task_dict = get_next_task(task_type=task_type)
            if not task_dict:
                time.sleep(0.3)  # Wait if queue is empty
                continue

            request_id = task_dict["request_id"]
            # Update task status to processing (not needed since get_next_task already does this)
            # But we do want to record the worker PID
            update_task_status(
                request_id, TaskStatus.PROCESSING.value, worker_pid=os.getpid()
            )

            logging.info(f"[Thread: {thread_name}] [PID: {process_id}] {worker_info} processing task {request_id}")
            sys.stdout.flush()

            try:
                process_single_request(task_dict)
                # Update status to complete
                update_task_status(request_id, TaskStatus.COMPLETE.value)
                logging.info(f"[Thread: {thread_name}] [PID: {process_id}] {worker_info} completed task {request_id}")
                sys.stdout.flush()
            except Exception as e:
                logging.error(
                    f"[Thread: {thread_name}] [PID: {process_id}] {worker_info} error processing request {request_id}: {str(e)}"
                )
                sys.stdout.flush()
                update_task_status(request_id, TaskStatus.ERROR.value, error=str(e))
        except Exception as e:
            logging.error(f"[Thread: {thread_name}] [PID: {process_id}] Error in {worker_info} queue processing: {str(e)}")
            sys.stdout.flush()
            logging.exception("Detailed error:")
            time.sleep(1)


def process_single_request(task_dict):
    import torch
    from trellis.utils import render_utils, postprocessing_utils

    """Process a single request from the queue"""
    request_id = task_dict["request_id"]
    request_output_dir = task_dict["request_output_dir"]
    os.makedirs(request_output_dir, exist_ok=True)
    
    # Get current thread name for better logging context
    thread_name = threading.current_thread().name
    
    # Log with more details including thread name and process ID
    logging.info(f"[Thread: {thread_name}] [PID: {os.getpid()}] Processing {task_dict['task_type']} request {request_id}")
    
    # Flush stdout to ensure logs are visible immediately
    sys.stdout.flush()

    try:
        if task_dict["task_type"] == TaskType.IMAGE_TO_3D.value:
            # Load and process image
            image_path = os.path.join(task_dict["input_path"])
            image = Image.open(image_path)

            if task_dict["is_dv_mode"]:
                binary_voxel = voxelize(task_dict["mesh_input_path"], resolution=64)
                outputs = _pipeline.run_detail_variation(
                    binary_voxel,
                    image,
                    seed=1,
                    sparse_structure_sampler_params={
                        "steps": task_dict["ss_sample_steps"] or 12,
                        "cfg_strength": task_dict["ss_cfg_strength"] or 7.5,
                    },
                    slat_sampler_params={
                        "steps": task_dict["slat_sample_steps"] or 12,
                        "cfg_strength": task_dict["slat_cfg_strength"] or 3.5,
                    },
                )
            else:
                outputs = _pipeline.run(
                    image,
                    seed=1,
                    sparse_structure_sampler_params={
                        "steps": task_dict["ss_sample_steps"] or 12,
                        "cfg_strength": task_dict["ss_cfg_strength"] or 7.5,
                    },
                    slat_sampler_params={
                        "steps": task_dict["slat_sample_steps"] or 12,
                        "cfg_strength": task_dict["slat_cfg_strength"] or 3.5,
                    },
                )
        else:  # TaskType.TEXT_TO_3D
            # detail variation / texture generation mode 
            input_text = task_dict["input_text"]
            # Load mesh
            if task_dict["is_dv_mode"]:
                mesh_input = task_dict["mesh_input_path"]
                outputs = _pipeline.run_detail_variation(
                    # notice that this should be z-up 
                    o3d.io.read_triangle_mesh(mesh_input),  
                    input_text,
                    seed=1,
                    sparse_structure_sampler_params={
                        "steps": task_dict["ss_sample_steps"] or 12,
                        "cfg_strength": task_dict["ss_cfg_strength"] or 7.5,
                    },
                    slat_sampler_params={
                        "steps": task_dict["slat_sample_steps"] or 12,
                        "cfg_strength": task_dict["slat_cfg_strength"] or 3.5,
                    },
                )
            else:
                outputs = _pipeline.run(
                    input_text,
                    seed=1,
                    sparse_structure_sampler_params={
                        "steps": task_dict["ss_sample_steps"] or 12,
                        "cfg_strength": task_dict["ss_cfg_strength"] or 7.5,
                    },
                    slat_sampler_params={
                        "steps": task_dict["slat_sample_steps"] or 12,
                        "cfg_strength": task_dict["slat_cfg_strength"] or 3.5,
                    },
                )
        torch.cuda.empty_cache()

        # By default don't render the video to save memory and time
        if task_dict.get("debug", False):
            video = render_utils.render_video(outputs["gaussian"][0])["color"]
            imageio.mimsave(os.path.join(request_output_dir, "gs.mp4"), video, fps=30)
            video = render_utils.render_video(outputs["mesh"][0])["normal"]
            imageio.mimsave(os.path.join(request_output_dir, "mesh.mp4"), video, fps=30)

        # we should return in the y-up mode
        trimesh_yup = postprocessing_utils.to_trimesh(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=task_dict.get("simplify_ratio", 0.95),
            texture_size=task_dict.get("texture_size", 1024),
            get_srgb_texture=False,
            fill_holes=True,
            texture_bake_mode=task_dict.get("texture_bake_mode", "fast"),
            render_resolution=512,
            debug=False,
            verbose=True,
        )

        # Save mesh
        mesh_path = os.path.join(request_output_dir, "output.glb")
        trimesh_yup.export(mesh_path)

    except Exception as e:
        logging.error(
            f"Error processing {task_dict['task_type']} request {request_id}: {str(e)}"
        )
        logging.exception("Detailed error:")
        raise


class AIWorker:
    """AI worker for processing 3D generation tasks"""

    def __init__(self, worker_id, gpu_id):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.pipeline_type = None
        self._pipeline_initialized = False
        self.process = None

    def start(self):
        """Initialize the pipeline and start processing tasks"""
        try:
            # Initialize CUDA context if GPU is available
            if self.gpu_id is not None:
                torch.cuda.init()
                torch.cuda.set_device(self.gpu_id)
                logging.info(
                    f"CUDA initialization successful for worker {self.worker_id}. Device: {self.gpu_id}, Count: {torch.cuda.device_count()}"
                )

            # Initialize pipeline based on worker type
            init_pipeline(gpu_id=self.gpu_id, pipeline_type=self.pipeline_type)
            logging.info(
                f"{self.pipeline_type.value} pipeline initialized successfully for worker {self.worker_id} on GPU {self.gpu_id}"
            )

            # Start processing tasks in a thread
            logging.info(
                f"Starting {self.pipeline_type.value} worker {self.worker_id} on GPU {self.gpu_id}"
            )
            # Here using the process instead of threads
            # self.process = multiprocessing.Process(
            #     target=process_queue,
            #     args=(self.pipeline_type, self.worker_id, self.gpu_id),
            #     daemon=True
            # )
            self.process = threading.Thread(
                target=process_queue,
                args=(self.pipeline_type, self.worker_id, self.gpu_id),
                daemon=True,
                name=f"{self.pipeline_type.value}-worker-{self.worker_id}-gpu-{self.gpu_id}"
            )
            self.process.start()
            return self.process
        except Exception as e:
            logging.error(f"Error starting worker {self.worker_id}: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            raise


class ImageWorker(AIWorker):
    """Worker specifically for image-to-3D tasks"""

    def __init__(self, worker_id, gpu_id):
        super().__init__(worker_id, gpu_id)
        self.pipeline_type = PipelineType.IMAGE


class TextWorker(AIWorker):
    """Worker specifically for text-to-3D tasks"""

    def __init__(self, worker_id, gpu_id):
        super().__init__(worker_id, gpu_id)
        self.pipeline_type = PipelineType.TEXT


# Global dictionary to track which GPU is running which type of worker
_gpu_worker_type_map = {}


class WorkerManager:
    """Manager for handling multiple AI workers"""

    def __init__(self, worker_class, num_workers, num_gpus):
        self.worker_class = worker_class
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.workers = []
        self.worker_type = (
            PipelineType.IMAGE if worker_class == ImageWorker else PipelineType.TEXT
        )
        worker_type_name = (
            "image-to-3D" if self.worker_type == PipelineType.IMAGE else "text-to-3D"
        )
        logging.info(
            f"Initializing {num_workers} {worker_type_name} workers with {num_gpus} GPUs"
        )

    def start_workers(self):
        """Start all workers and distribute them across available GPUs"""
        global _gpu_worker_type_map

        # Count of successfully started workers
        started_workers = 0
        worker_id = 1  # Worker IDs start at 1

        # Try to start the requested number of workers
        while (
            started_workers < self.num_workers and worker_id <= self.num_workers * 2
        ):  # Limit attempts
            # Find an available GPU that isn't running a different worker type
            gpu_id = None

            for i in range(self.num_gpus):
                # If GPU is not in use or is already running our worker type, we can use it
                if (
                    i not in _gpu_worker_type_map
                    or _gpu_worker_type_map[i] == self.worker_type
                ):
                    gpu_id = i
                    break

            # If no suitable GPU found, log error and break
            if gpu_id is None:
                logging.error(
                    f"No available GPUs for {self.worker_type.value} worker. "
                    f"All GPUs are running different worker types."
                )
                break

            # Create and start worker
            worker = self.worker_class(worker_id, gpu_id)

            try:
                # Start the worker (initializes pipeline and begins processing)
                worker.start()
                self.workers.append(worker)

                # Mark this GPU as running this worker type
                _gpu_worker_type_map[gpu_id] = self.worker_type

                logging.info(
                    f"Started worker {worker_id} of type {worker.pipeline_type.value} on GPU {gpu_id}"
                )

                started_workers += 1
            except Exception as e:
                logging.error(
                    f"Failed to start worker {worker_id} on GPU {gpu_id}: {str(e)}"
                )
                logging.error("Stack trace:", exc_info=True)

            worker_id += 1

        if started_workers < self.num_workers:
            logging.warning(
                f"Only started {started_workers}/{self.num_workers} requested {self.worker_type.value} workers"
            )

        return self.workers


def get_gpu_worker_allocation():
    """Get the current worker allocation on GPUs

    Returns:
        Dictionary mapping GPU IDs to worker types
    """
    return _gpu_worker_type_map.copy()


def start_image_workers(num_workers, num_gpus):
    """Start image-to-3D workers"""
    logging.info(f"Starting {num_workers} image-to-3D workers across {num_gpus} GPUs")
    manager = WorkerManager(ImageWorker, num_workers, num_gpus)
    return manager.start_workers()


def start_text_workers(num_workers, num_gpus):
    """Start text-to-3D workers"""
    logging.info(f"Starting {num_workers} text-to-3D workers across {num_gpus} GPUs")
    manager = WorkerManager(TextWorker, num_workers, num_gpus)
    return manager.start_workers()


if __name__ == "__main__":
    import multiprocessing
    import argparse

    # Set up PyTorch multiprocessing to use 'spawn' method for better compatibility
    # This is important for CUDA operations across processes
    multiprocessing.set_start_method("spawn")

    # Parse command line arguments to allow configuration at runtime
    parser = argparse.ArgumentParser(description="Start TRELLIS API workers")
    parser.add_argument(
        "--gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--image-workers-per-gpu",
        type=int,
        default=IMAGE_WORKERS_PER_GPU,
        help="Number of image-to-3D workers per GPU",
    )
    parser.add_argument(
        "--text-workers-per-gpu",
        type=int,
        default=TEXT_WORKERS_PER_GPU,
        help="Number of text-to-3D workers per GPU",
    )
    parser.add_argument(
        "--image-only", action="store_true", help="Only start image-to-3D workers"
    )
    parser.add_argument(
        "--text-only", action="store_true", help="Only start text-to-3D workers"
    )

    args = parser.parse_args()

    # Update the configuration based on command line arguments
    NUM_GPUS = args.gpus
    IMAGE_WORKERS_PER_GPU = args.image_workers_per_gpu
    TEXT_WORKERS_PER_GPU = args.text_workers_per_gpu
    NUM_IMAGE_WORKERS = NUM_GPUS * IMAGE_WORKERS_PER_GPU
    NUM_TEXT_WORKERS = NUM_GPUS * TEXT_WORKERS_PER_GPU

    logging.info(f"Starting TRELLIS API with {NUM_GPUS} GPUs")
    logging.info(
        f"Image workers per GPU: {IMAGE_WORKERS_PER_GPU}, total: {NUM_IMAGE_WORKERS}"
    )
    logging.info(
        f"Text workers per GPU: {TEXT_WORKERS_PER_GPU}, total: {NUM_TEXT_WORKERS}"
    )

    image_workers = []
    text_workers = []

    # Start text-to-3D workers
    if not args.image_only and NUM_TEXT_WORKERS > 0:
        text_workers = start_text_workers(NUM_TEXT_WORKERS, NUM_GPUS)
        logging.info(f"Started {len(text_workers)} text-to-3D workers")

    # Start image-to-3D workers
    if not args.text_only and NUM_IMAGE_WORKERS > 0:
        image_workers = start_image_workers(NUM_IMAGE_WORKERS, NUM_GPUS)
        logging.info(f"Started {len(image_workers)} image-to-3D workers")

    # Keep the main process running to maintain the worker threads
    try:
        # Wait for keyboard interrupt
        while True:
            time.sleep(10)
            logging.info("Workers running...")
    except KeyboardInterrupt:
        logging.info("Shutting down workers")
    finally:
        logging.info("Exiting...")
