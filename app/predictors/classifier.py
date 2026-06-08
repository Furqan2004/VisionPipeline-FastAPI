import torch
from torchvision import transforms
from PIL import Image
import os
from app.core.logger_utils import CustomLogger

class EfficientNetClassifier:
    def __init__(self, model_path: str, image_size: int, threshold: float):
        self.logger = CustomLogger("classifier")
        self.threshold  = threshold
        self.image_size = image_size
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        try:
            if os.path.exists(model_path):
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                self.logger.info("Classifier model loaded.")
            else:
                self.logger.error(f"Model not found: {model_path}")
                self.model = None
        except Exception as e:
            self.logger.error(f"Failed to load classifier: {e}")
            self.model = None

    def predict(self, image_input) -> int:
        if self.model is None: return -1
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input).convert("RGB") if hasattr(image_input, 'shape') else image_input
            
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(tensor)).item()

            label = 1 if prob >= self.threshold else 0
            return label
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return -1
