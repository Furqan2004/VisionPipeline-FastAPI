# VisionPipeline-FastAPI

A modular and professional FastAPI-based vision pipeline that orchestrates multiple AI models: **YOLO** for object detection, **SAM2** for head/helmet segmentation, and **EfficientNet** for final classification.

This project is designed to handle both high-quality image processing and optimized video inference with frame-skipping capabilities.

## 🚀 Getting Started

Follow these steps to set up the project locally. This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 1. Clone the Repository
```bash
git clone https://github.com/Furqan2004/VisionPipeline-FastAPI.git
cd VisionPipeline-FastAPI
```

### 2. Install Dependencies
Ensure you have `uv` installed, then run:
```bash
uv sync
```

### 3. Place Model Weights
Create a `weights/` directory and place your model files there:
- `weights/best.pt` (YOLO)
- `weights/sam2.1_hiera_large.pt` (SAM2)
- `weights/efficientnet_b3_traced.pt` (EfficientNet Classifier)

### 4. Run the Application
```bash
uv run uvicorn main:app --reload
```
The API will be available at `http://0.0.0.0:8000`. You can access the interactive Swagger documentation at `http://0.0.0.0:8000/docs`.

---

## 📂 Project Structure

```
Structure_Work/
├── configs/            # JSON configuration for model paths and thresholds
├── weights/            # AI Model weight files (.pt) - Git ignored
├── app/                # Application source code
│   ├── api/            # FastAPI Routes and dependency injection
│   ├── core/           # Configuration loader and logging utilities
│   ├── models/         # Pydantic request/response schemas
│   └── predictors/     # Individual AI model implementations (YOLO, SAM2, Classifier)
├── logs/               # Detailed per-module log files
├── output/             # Processed images and videos
├── main.py             # FastAPI entry point
├── pyproject.toml      # Project metadata and dependencies
└── .gitignore          # Rules to exclude weights and environments
```

---

## 🛠 API Endpoints

### `GET /health`
Returns the status of the API and checks if all AI models are loaded correctly.

### `POST /predict-image`
- **Input**: Image file (JPG, PNG, etc.)
- **Process**: Detects objects -> Segments head region -> Classifies segment.
- **Output**: JSON with labels and path to the segmented result image.

### `POST /predict-video`
- **Input**: Video file (MP4, AVI, MOV, etc.)
- **Process**: Frame-by-frame inference using `frame_interval` for speed.
- **Output**: A processed `.avi` video file with bboxes, masks, and labels overlaid.

---

## ⚙️ Configuration

Settings are managed in `configs/config.json`. You can adjust:
- `frame_interval`: Skip frames during video inference (e.g., `5` means process every 5th frame).
- `threshold`: Confidence threshold for the classifier.
- `image_size`: Internal resize dimension for the classifier.

---

## 📝 Logging
Detailed logs are maintained for every module in the `logs/` directory, allowing for granular tracking of model loading, inference performance, and error handling.
