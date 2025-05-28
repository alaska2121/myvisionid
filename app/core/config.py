import os
import logging
from datetime import datetime
from typing import Optional

class Config:
    """Configuration class for the application."""
    
    def __init__(self):
        self._setup_logging()
        self._setup_model_paths()
        self.matting_model = os.getenv("MATTING_MODEL", "birefnet-v1-lite")
        self.face_detect_model = os.getenv("FACE_DETECT_MODEL", "retinaface-resnet50")
        
        # More conservative defaults for cloud deployment
        self.max_concurrent_workers = int(os.getenv("MAX_CONCURRENT_WORKERS", "1"))  # Default to 1 for Railway
        self.memory_threshold_mb = int(os.getenv("MEMORY_THRESHOLD_MB", "800"))  # Reduced for Railway
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "2"))
        
        # Detect if running on Railway
        self.is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None
        
        if self.is_railway:
            # Even more conservative settings for Railway
            self.max_concurrent_workers = min(self.max_concurrent_workers, 1)
            self.memory_threshold_mb = min(self.memory_threshold_mb, 400)  # Very conservative: 400MB max for Railway
            logging.info("Railway environment detected - using conservative settings")
        
        # Validate configuration
        if self.max_concurrent_workers < 1:
            self.max_concurrent_workers = 1
        if self.max_concurrent_workers > 3:
            self.max_concurrent_workers = 3  # Cap to prevent memory issues
        
        logging.info(f"Config initialized: workers={self.max_concurrent_workers}, memory_threshold={self.memory_threshold_mb}MB, railway={self.is_railway}")
    
    def _get_log_file_path(self) -> str:
        """Get the path for the log file."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = self._get_log_file_path()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_model_paths(self) -> dict[str, str]:
        """Get the paths for all model files."""
        return {
            'retinaface': "retinaface/RetinaFace-R50.pth",
            'modnet': "modnet_photographic_portrait_matting/modnet_photographic_portrait_matting.ckpt",
            'onnx': "hivision/creator/weights/birefnet-v1-lite.onnx"
        }
    
    def _validate_model_path(self, path: str, model_name: str) -> bool:
        """Validate if a model file exists and log appropriate messages."""
        if not os.path.isfile(path):
            logging.error(f"{model_name} model not found at: {path}")
            dir_path = os.path.dirname(path)
            if os.path.exists(dir_path):
                logging.error(f"Directory contents of {dir_path}/: {os.listdir(dir_path)}")
            else:
                logging.error(f"Directory not found: {dir_path}")
            return False
        logging.info(f"{model_name} model found at: {path}")
        return True
    
    def _determine_matting_model(self) -> tuple[str, str]:
        """Determine which matting model to use based on available files."""
        paths = self._get_model_paths()
        
        if os.path.isfile(paths['onnx']):
            return "birefnet-v1-lite", paths['onnx']
        elif os.path.isfile(paths['modnet']):
            return "birefnet-v1-lite", paths['modnet']
        else:
            logging.warning("Warning: MODNet model not found. Falling back to hivision_modnet.")
            fallback_path = "hivision/creator/weights/hivision_modnet.onnx"
            if not os.path.isfile(fallback_path):
                logging.error(f"Fallback model not found at: {fallback_path}")
                dir_path = os.path.dirname(fallback_path)
                if os.path.exists(dir_path):
                    logging.error(f"Directory contents of {dir_path}/: {os.listdir(dir_path)}")
                else:
                    logging.error(f"Directory not found: {dir_path}")
                raise FileNotFoundError(f"Fallback model not found at: {fallback_path}")
            logging.info(f"Fallback model found at: {fallback_path}")
            return "hivision_modnet", fallback_path
    
    def _setup_model_paths(self):
        """Set up and validate model paths."""
        # Log current directory information
        logging.info("Checking model files...")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Directory contents: {os.listdir('.')}")

        # Validate RetinaFace model
        paths = self._get_model_paths()
        if not self._validate_model_path(paths['retinaface'], "RetinaFace"):
            raise FileNotFoundError(f"RetinaFace model not found at: {paths['retinaface']}")

        # Determine and set matting model
        self.matting_model, self.onnx_model_path = self._determine_matting_model() 