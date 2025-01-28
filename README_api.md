# TRELLIS API Documentation
- **Update-01-26**: Switch to separate ai_workers and web_workers.
- **Update-01-19**: I have deployed a ready-to-use endpoint url on a cloud 3060 GPU and the low-vram mode. You can use that if you just want to have a try. Otherwise it's better to deploy by yourself.

## 1. Installation

### Prerequisites
- Python 3.8+
- Redis server
- CUDA-capable GPU (recommended)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/FishWoWater/TRELLIS.git
cd TRELLIS
```

2. Install Python dependencies:
```bash
# install requirements according to offical repo
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# requirements for the API 
pip install flask flask_cors open3d redis gunicorn
# start the AI workers (edit config params in it)
python ai_worker.py 
# start the web workers (process web requests and send them to ai workers)
python web_worker.py 
```

3. Install and start Redis server:
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# MacOS
brew install redis
brew services start redis
```

4. Start the API server:
```bash
# for development mode (uses GPU 0)
python api.py
# for production mode (automatically uses all available GPUs)
gunicorn -c gunicorn_config.py wsgi:app
```

The server will run on port 6006 by default. In production mode, it will automatically detect the number of available GPUs and create one worker per GPU for optimal resource utilization.

## 2. API Endpoints

### 2.1 Image to 3D Conversion
**Endpoint**: `/image_to_3d`  
**Method**: POST  
**Content-Type**: multipart/form-data

**Parameters:**
- `image` (file, required): Input image file
- `sparse_structure_sample_steps` (int, optional): Number of sampling steps for sparse structure generation (default: 25)
- `sparse_structure_cfg_strength` (float, optional): Guidance strength for sparse structure (default: 8.5)
- `slat_sample_steps` (int, optional): Number of sampling steps for structured latent (default: 25)
- `slat_cfg_strength` (float, optional): Guidance strength for structured latent (default: 4.0)
- `simplify_ratio` (float, optional): Mesh simplification ratio (default: 1.0)
- `texture_size` (int, optional): Size of generated texture (default: 1024)
- `texture_bake_mode` (str, optional): Mode for texture baking (default: "pbr")

**Response:**
```json
{
    "request_id": "string",
    "status": "queued"
}
```

### 2.2 Get Request Status
**Endpoint**: `/my_requests`  
**Method**: GET

**Response:**
```json
{
    "requests": [
        {
            "request_id": "string",
            "status": "string",  // "queued", "processing", "complete", or "error"
            "image_name": "string",
            "output_files": ["string"],  // Only present if status is "complete"
            "error": "string"  // Only present if status is "error"
        }
    ]
}
```

### 2.3 Queue Status
**Endpoint**: `/queue/status`  
**Method**: GET

**Parameters:**
- `request_id` (string, optional): Get status of specific request

**Response:**
```json
{
    "queue_length": "int",
    "processing": "int",
    "client_queue_length": "int",
    "client_processing": "int"
}
```

### 2.4 Download Result
**Endpoint**: `/outputs/<ip_address>/<request_id>/<filename>`  
**Method**: GET

Downloads a specific output file from a completed request.

## 3. Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found
- 500: Internal Server Error

Error responses include a message explaining the error:
```json
{
    "error": "Error message"
}
```

## Rate Limiting

- Each IP address is limited to 10 recent requests in history
- Requests are processed in a FIFO queue
- The server maintains separate output directories for each IP address

## Implementation Notes

- The API uses Redis for request queue management
- All requests are processed asynchronously
- Output files are stored in IP-specific directories
- The server supports both CPU and GPU processing (GPU recommended)
