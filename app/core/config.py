import json
import os

from typing import Any, Dict, Optional

class Settings:
    def __init__(self, config_path: str = "configs/config.json") -> None:
        if not os.path.exists(config_path):
            # Fallback to local path if running from within app/core
            config_path = os.path.join(os.path.dirname(__file__), "../../configs/config.json")
            
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @property
    def yolo_model_name(self) -> Optional[str]: return self.config.get("yolo_model_name")
    
    @property
    def sam_config_file(self) -> Optional[str]: return self.config.get("sam_config_file")
    
    @property
    def sam_ckpt_path(self) -> Optional[str]: return self.config.get("sam_ckpt_path")
    
    @property
    def model_path(self) -> Optional[str]: return self.config.get("model_path")
    
    @property
    def image_size(self) -> int: return self.config.get("image_size", 300)
    
    @property
    def threshold(self) -> float: return self.config.get("threshold", 0.5)
    
    @property
    def video_config(self) -> Dict[str, Any]: return self.config.get("video_config", {})

settings = Settings()
