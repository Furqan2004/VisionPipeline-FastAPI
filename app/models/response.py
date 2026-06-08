from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionResult(BaseModel):
    id: int
    label: int
    bbox: List[int]

class ImageResponse(BaseModel):
    labels: List[PredictionResult]
    output_image: str

class HealthStatus(BaseModel):
    status: str
    models: Dict[str, str]
