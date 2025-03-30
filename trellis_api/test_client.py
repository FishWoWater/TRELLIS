#!/usr/bin/env python3
"""
Test client for the TRELLIS API server
This script tests both image-to-3D and text-to-3D endpoints
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path

# Configuration
DEFAULT_SERVER_URL = "http://localhost:6006"  # Default server URL
DEFAULT_IMAGE_PATH = "assets/example_image/typical_creature_dragon.png"
DEFAULT_TEXT_PROMPT = "A majestic dragon with detailed scales, sharp claws, and wings"


def test_image_to_3d(server_url, image_path, verbose=False):
    """Test the image-to-3D endpoint"""
    endpoint = f"{server_url}/image_to_3d"

    # Prepare form data
    files = {
        "image": (os.path.basename(image_path), open(image_path, "rb"), "image/png")
    }

    form_data = {
        "is_dv_mode": "false",
        "ss_sample_steps": "25",
        "ss_cfg_strength": "8.5",
        "slat_sample_steps": "25",
        "slat_cfg_strength": "4.0",
    }

    print(f"Sending image {image_path} to {endpoint}...")
    response = requests.post(endpoint, files=files, data=form_data)

    if response.status_code == 200:
        result = response.json()
        request_id = result.get("request_id")
        print(f"Request submitted successfully. Request ID: {request_id}")

        if verbose:
            print(json.dumps(result, indent=2))

        # Poll for status
        poll_task_status(server_url, request_id)
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_text_to_3d(server_url, text_prompt, verbose=False):
    """Test the text-to-3D endpoint"""
    endpoint = f"{server_url}/text_to_3d"

    # Prepare JSON data
    json_data = {
        "text": text_prompt,
        "ss_sample_steps": 30,
        "ss_cfg_strength": 7.5,
        "slat_sample_steps": 30,
        "slat_cfg_strength": 4.5,
    }

    print(f"Sending text prompt to {endpoint}...")
    print(f"Prompt: {text_prompt}")

    response = requests.post(endpoint, json=json_data)

    if response.status_code == 200:
        result = response.json()
        request_id = result.get("request_id")
        print(f"Request submitted successfully. Request ID: {request_id}")

        if verbose:
            print(json.dumps(result, indent=2))

        # Poll for status
        poll_task_status(server_url, request_id)
    else:
        print(f"Error: {response.status_code} - {response.text}")


def poll_task_status(server_url, request_id, interval=5, max_attempts=5):
    """Poll for task status until completion or error"""
    endpoint = f"{server_url}/my_requests"
    attempts = 0

    print(f"Polling for task status (request ID: {request_id})...")

    while attempts < max_attempts:
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                tasks = response.json().get("requests", [])
                # Find the task with our request ID
                task = next(
                    (t for t in tasks if t.get("request_id") == request_id), None
                )

                if task:
                    status = task.get("status")
                    print(f"Status: {status}")

                    if status == "complete":
                        print("Task completed successfully!")
                        return True
                    elif status == "error":
                        print(
                            f"Task failed with error: {task.get('error', 'Unknown error')}"
                        )
                        return False
                else:
                    print(f"Task with request ID {request_id} not found")
            else:
                print(
                    f"Error checking status: {response.status_code} - {response.text}"
                )
        except Exception as e:
            print(f"Error polling for status: {str(e)}")

        attempts += 1
        time.sleep(interval)

    print(f"Gave up after {max_attempts} attempts")
    return False


def check_worker_status(server_url, verbose=False):
    """Check the current worker allocation on GPUs

    Args:
        server_url: Base URL of the TRELLIS API server
        verbose: Whether to print detailed information

    Returns:
        Dictionary with worker status information or None if request failed
    """
    try:
        response = requests.get(f"{server_url}/worker_status", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Worker status retrieved successfully")

            # Print GPU allocation information
            gpu_allocation = status_data.get("gpu_allocation", {})
            if gpu_allocation:
                print("\nGPU Worker Allocation:")
                print("-" * 30)
                for gpu_id, worker_type in gpu_allocation.items():
                    print(f"GPU {gpu_id}: {worker_type} worker")
            else:
                print("No workers currently allocated to GPUs")

            if verbose:
                print("\nRaw response:")
                print(json.dumps(status_data, indent=2))

            return status_data
        else:
            print(f"❌ Failed to get worker status: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking worker status: {e}")
        return None


def check_server_status(server_url):
    """Check if the server is running"""
    try:
        response = requests.get(f"{server_url}/status")
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.ConnectionError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Test TRELLIS API endpoints")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="Server URL")
    parser.add_argument(
        "--mode",
        choices=["image", "text", "both", "status", "workers"],
        default="both",
        help="Test mode: image-to-3D, text-to-3D, both, server status, or worker status",
    )
    parser.add_argument(
        "--image", default=DEFAULT_IMAGE_PATH, help="Path to image for image-to-3D test"
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT_PROMPT, help="Text prompt for text-to-3D test"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()

    # For status-only checks, don't verify server is running first
    if args.mode == "status":
        print(f"Checking TRELLIS API server status at {args.server}")
        if check_server_status(args.server):
            print(f"✅ Server at {args.server} is running")
        else:
            print(f"❌ Server at {args.server} is not responding")
        return

    # For worker status checks
    if args.mode == "workers":
        print(f"Checking worker status at {args.server}")
        if check_server_status(args.server):
            check_worker_status(args.server, args.verbose)
        else:
            print(f"❌ Server at {args.server} is not responding")
        return

    # For other modes, check if server is running
    if not check_server_status(args.server):
        print(f"Error: Server at {args.server} is not responding")
        print("Make sure the web server and AI workers are running")
        sys.exit(1)

    # Verify image path exists
    if args.mode in ["image", "both"]:
        if not os.path.isfile(args.image):
            print(f"Error: Image file {args.image} not found")
            sys.exit(1)

    print(f"Testing TRELLIS API at {args.server}")

    # Check worker status before running tests
    # check_worker_status(args.server, args.verbose)

    if args.mode in ["image", "both"]:
        test_image_to_3d(args.server, args.image, args.verbose)

    if args.mode in ["text", "both"]:
        test_text_to_3d(args.server, args.text, args.verbose)


if __name__ == "__main__":
    main()
