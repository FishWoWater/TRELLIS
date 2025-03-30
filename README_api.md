# TRELLIS API Documentation
A simple API backend of TRELLIS, supporting text-to-3d and image-to-3d textured mesh generation. Also equipped with simple task management and worker system. 

## Core Features 
* Text-to-3D textured mesh generation 
* Image-to-3D textured mesh generation 
* Text-Conditioned detail variation(texture generation)
* Image-Conditioned detail variation(texture generation)
* Task queue management (queued/processing/complete/error)

## Installation

### Prerequisites
- Python 3.8+
- sqlalchemy
- CUDA-capable GPU (12GB+ or 8GB+ with low-vram mode)

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
pip install flask flask_cors open3d gunicorn sqlalchemy
```

2. Edit the endpoint url/port and basic config params in [config](trellis_api/config.py)

3. Start the API server:
```bash
# notice that currently a single GPU can ONLY be used to run a single text worker or a single image worker 
# if you have only 1 gpu, sum of them should be at most 1
python trellis_api/ai_worker.py --text-workers-per-gpu 1 --image-workers-per-gpu 0
# start the web workers (process web requests and send them to ai workers)
python trellis_api/web_server.py 
```

## Client examples 
#### Python
```python 
python trellis_api/test_client.py
```

#### Curl 
```bash
./trellis_api/test_client.sh
```

## API Endpoints

### Image to 3D Conversion
**Endpoint**: `/image_to_3d`  
**Method**: POST  
**Content-Type**: application/json

**Parameters:**
- `image` (base64 encoded string): Input image
- `mesh` (base64 encoded string, optional): Input mesh (if provided will enter detail variation mode)
- `sparse_structure_sample_steps` (int, optional): Number of sampling steps for sparse structure generation (default: 12)
- `sparse_structure_cfg_strength` (float, optional): Guidance strength for sparse structure (default: 7.5)
- `slat_sample_steps` (int, optional): Number of sampling steps for structured latent (default: 12)
- `slat_cfg_strength` (float, optional): Guidance strength for structured latent (default: 3.5)
- `simplify_ratio` (float, optional): Mesh simplification ratio (default: 0.95)
- `texture_size` (int, optional): Size of generated texture (default: 1024)
- `texture_bake_mode` (enum, optional): Mode for texture baking (default: "fast")

**Response:**
```json
{
    "request_id": "string",
    "status": "queued"
}
```

### Text to 3D Conversion
**Endpoint**: `/text_to_3d`  
**Method**: POST  
**Content-Type**: application/json

**Parameters:**
- `text` (string): Prompt on the object to generate
- `mesh` (base64 encoded string, optional): Input mesh (if provided will enter detail variation mode)
- `negative_text` (string, optional): Negative prompt
- `sparse_structure_sample_steps` (int, optional): Number of sampling steps for sparse structure generation (default: 12)
- `sparse_structure_cfg_strength` (float, optional): Guidance strength for sparse structure (default: 7.5)
- `slat_sample_steps` (int, optional): Number of sampling steps for structured latent (default: 12)
- `slat_cfg_strength` (float, optional): Guidance strength for structured latent (default: 3.5)
- `simplify_ratio` (float, optional): Mesh simplification ratio (default: 0.95)
- `texture_size` (int, optional): Size of generated texture (default: 1024)
- `texture_bake_mode` (str, optional): Mode for texture baking (default: "pbr")

**Response:**
```json
{
    "request_id": "string",
    "status": "queued"
}
```
### Get Task Status
**Endpoint**: `/task/<request_id>`  
**Method**: GET

**Response:**
```json
{
    "request_id": "string",
    "status": "string",  // "queued", "processing", "complete", or "error"
    "image_name": "string",
    "input_text": "string", 
    "output_files": ["string"],  // Only present if status is "complete"
    "error": "string"  // Only present if status is "error"
}
```

### Get All Requests
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

### Queue Status
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

### Download Result
**Endpoint**: `/outputs/<ip_address>/<request_id>/<filename>`  
**Method**: GET

Downloads a specific output file from a completed request.

## Error Handling

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

