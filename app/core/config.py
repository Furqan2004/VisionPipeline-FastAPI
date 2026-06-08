import json
import os

class Settings:
    def __init__(self, config_path="configs/config.json"):
        if not os.path.exists(config_path):
            # Fallback to local path if running from within app/core
            config_path = os.path.join(os.path.dirname(__file__), "../../configs/config.json")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def yolo_model_name(self): return self.config.get("yolo_model_name")
    
    @property
    def sam_config_file(self): return self.config.get("sam_config_file")
    
    @property
    def sam_ckpt_path(self): return self.config.get("sam_ckpt_path")
    
    @property
    def model_path(self): return self.config.get("model_path")
    
    @property
    def image_size(self): return self.config.get("image_size")
    
    @property
    def threshold(self): return self.config.get("threshold")
    
    @property
    def video_config(self): return self.config.get("video_config")

settings = Settings()
