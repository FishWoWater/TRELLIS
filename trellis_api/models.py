import enum
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Boolean,
    Float,
    create_engine,
    Enum,
)
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


class TaskStatus(enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class TaskType(enum.Enum):
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_3D = "text_to_3d"


class Task(Base):
    __tablename__ = "tasks"

    request_id = Column(String, primary_key=True)
    task_type = Column(Enum(TaskType), nullable=False)
    status = Column(String, nullable=False)
    client_ip = Column(String, nullable=False)

    # Input fields
    input_path = Column(String)  # For image-to-3D tasks
    input_text = Column(String)  # For text-to-3D tasks
    mesh_input_path = Column(String)  # For detail variation mode

    # Common fields
    request_output_dir = Column(String, nullable=False)
    is_dv_mode = Column(Boolean, default=False)
    image_name = Column(String)
    start_time = Column(DateTime)
    finish_time = Column(DateTime)
    error = Column(String)
    worker_pid = Column(Integer)

    # Optional parameters
    ss_sample_steps = Column(Integer)
    ss_cfg_strength = Column(Float)
    slat_sample_steps = Column(Integer)
    slat_cfg_strength = Column(Float)

    def to_dict(self, include_all_fields=False):
        """Convert Task object to dictionary
        
        Args:
            include_all_fields: If True, include all fields needed for processing.
                              If False, include only fields needed for API responses.
        
        Returns:
            Dictionary with task data
        """
        # Base fields for API responses
        result = {
            "request_id": self.request_id,
            "task_type": self.task_type.value,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "image_name": self.image_name,
            "text": self.input_text, 
            "error": self.error,
            "worker_pid": self.worker_pid,
        }
        
        # Additional fields needed for task processing
        if include_all_fields:
            result.update({
                "client_ip": self.client_ip,
                "input_path": self.input_path,
                "input_text": self.input_text,
                "mesh_input_path": self.mesh_input_path,
                "request_output_dir": self.request_output_dir,
                "is_dv_mode": self.is_dv_mode,
                "ss_sample_steps": self.ss_sample_steps,
                "ss_cfg_strength": self.ss_cfg_strength,
                "slat_sample_steps": self.slat_sample_steps,
                "slat_cfg_strength": self.slat_cfg_strength,
            })
        
        return result


# Database setup
engine = create_engine("sqlite:///trellis_api/trellis.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
