import os
from datetime import datetime
from trellis_api.models import Session, Task, TaskType, TaskStatus
from trellis_api.config import (
    BASE_URL, INPUT_DIR, OUTPUT_DIR, 
    IP_HISTORY_LIMIT, AI_WORKER, MAIN_WORKER
)


def get_ip_output_dir(ip_address):
    """Get the output directory for a specific IP address"""
    ip_dir = os.path.join(OUTPUT_DIR, ip_address.replace(":", "_"))
    os.makedirs(ip_dir, exist_ok=True)
    return ip_dir


def get_next_task(task_type=None):
    """Get the next task from the queue, optionally filtered by task type

    Args:
        task_type: Optional TaskType enum value to filter tasks by type
        
    Returns:
        A dictionary with task data if a task is found, otherwise None
    """
    session = Session()
    try:
        query = session.query(Task).filter(Task.status == TaskStatus.QUEUED.value)

        # If task_type is specified, filter by task type
        if task_type:
            query = query.filter(Task.task_type == task_type)

        # Order by submission time (oldest first)
        task = query.order_by(Task.start_time.asc()).first()

        if task:
            # Get dictionary with all fields for task processing
            task_dict = task.to_dict(include_all_fields=True)
            
            # Update the task status in the database
            task.status = TaskStatus.PROCESSING.value
            task.start_time = datetime.now()
            session.commit()
            
            return task_dict
        return None
    finally:
        session.close()


def update_task_status(request_id, status, error=None, worker_pid=None):
    """Update the status of a task"""
    session = Session()
    try:
        task = session.query(Task).filter(Task.request_id == request_id).first()
        if task:
            task.status = status
            if status == TaskStatus.COMPLETE.value or status == TaskStatus.ERROR.value:
                task.finish_time = datetime.now()
            if error:
                task.error = error
            if worker_pid:
                task.worker_pid = worker_pid
            session.commit()
    finally:
        session.close()


def get_tasks_for_ip(client_ip):
    """Get tasks for a specific IP address
    
    Args:
        client_ip: IP address to filter tasks by
        
    Returns:
        List of task dictionaries for the specified IP address
    """
    session = Session()
    try:
        tasks = (
            session.query(Task)
            .filter(Task.client_ip == client_ip)
            .order_by(Task.start_time.desc())
            .limit(IP_HISTORY_LIMIT)
            .all()
        )
        # Convert Task objects to dictionaries
        return [task.to_dict() for task in tasks]
    finally:
        session.close()


def get_processing_tasks():
    """Get all tasks that are currently processing
    
    Returns:
        List of task dictionaries for all currently processing tasks
    """
    session = Session()
    try:
        tasks = (
            session.query(Task).filter(Task.status == TaskStatus.PROCESSING.value).all()
        )
        # Convert Task objects to dictionaries
        return [task.to_dict() for task in tasks]
    finally:
        session.close()


def get_task_by_id(request_id):
    """Get a specific task by request ID
    
    Args:
        request_id: The unique request ID of the task
        
    Returns:
        Task dictionary if found, None otherwise
    """
    session = Session()
    try:
        task = session.query(Task).filter(Task.request_id == request_id).first()
        if task:
            return task.to_dict(include_all_fields=True)
        return None
    finally:
        session.close()


def create_task(request_data, client_ip):
    """Create a new task in the database"""
    session = Session()
    try:
        # Get task_type from the request data
        task_type_value = request_data.get("task_type")
        task_type = None

        # Convert string task_type to enum value
        if task_type_value == TaskType.IMAGE_TO_3D.value:
            task_type = TaskType.IMAGE_TO_3D
        elif task_type_value == TaskType.TEXT_TO_3D.value:
            task_type = TaskType.TEXT_TO_3D
        else:
            # Default to image-to-3D if not specified
            task_type = TaskType.IMAGE_TO_3D

        task = Task(
            request_id=request_data["request_id"],
            task_type=task_type,  # Set the task type
            status=TaskStatus.QUEUED.value,
            client_ip=client_ip,
            input_path=request_data["input_path"],
            mesh_input_path=request_data.get("mesh_input_path", ""),
            request_output_dir=request_data["request_output_dir"],
            is_dv_mode=request_data["is_dv_mode"],
            image_name=request_data.get("image_name", ""),
            input_text=request_data.get("input_text", ""),  # Added for text-to-3D
            ss_sample_steps=request_data.get("ss_sample_steps"),
            ss_cfg_strength=request_data.get("ss_cfg_strength"),
            slat_sample_steps=request_data.get("slat_sample_steps"),
            slat_cfg_strength=request_data.get("slat_cfg_strength"),
        )
        session.add(task)
        session.commit()
    finally:
        session.close()
