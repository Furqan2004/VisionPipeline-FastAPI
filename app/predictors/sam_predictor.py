import torch
import numpy as np
from typing import Union, List, Any
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from app.core.logger_utils import CustomLogger

class SAM2Predictor:
    def __init__(self, config_file: str, ckpt_path: str) -> None:
        self.logger: CustomLogger = CustomLogger("sam_predictor")
        self.config_file: str = config_file
        self.ckpt_path: str = ckpt_path
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"Initializing SAM2 on device: {self.device}")
        try:
            self.model: Any = build_sam2(
                config_file = self.config_file,
                ckpt_path   = self.ckpt_path,
                device      = self.device,
            )
            self.mask_generator: SAM2AutomaticMaskGenerator = SAM2AutomaticMaskGenerator(
                model                  = self.model,
                points_per_side        = 32,
                pred_iou_thresh        = 0.85,
                stability_score_thresh = 0.90,
                min_mask_region_area   = 500,
            )
            self.logger.info("SAM2 loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SAM2: {e}")
            raise

    def get_head_segmentation(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
                image_np = np.array(image)
            else:
                image_np = image_input

            h, w = image_np.shape[:2]

            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32):
                all_masks = self.mask_generator.generate(image_np)

            all_masks = sorted(all_masks, key=lambda x: x["area"], reverse=True)

            helmet_mask_data = None
            for m in all_masks:
                seg = m["segmentation"]
                ys, _ = np.where(seg)
                if len(ys) == 0: continue
                cy = ys.mean() / h
                if cy < 0.45 and helmet_mask_data is None:
                    helmet_mask_data = m
                    break

            if helmet_mask_data is None:
                return np.zeros((h, w), bool)

            return helmet_mask_data["segmentation"]
        except Exception as e:
            self.logger.error(f"Error during SAM segmentation: {e}")
            return np.zeros((1, 1), bool)
