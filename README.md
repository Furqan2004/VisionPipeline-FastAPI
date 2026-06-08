# Structure Work - Vision Pipeline

A modular FastAPI-based vision pipeline combining YOLO for object detection, SAM2 for segmentation, and EfficientNet for classification.

## Project Structure

```
Structure_Work/
├── configs/            # Configuration files
├── weights/            # AI Model weight files (.pt)
├── app/                # Application source code
│   ├── api/            # API Routes and dependencies
│   ├── core/           # Config and logging utilities
│   ├── models/         # Pydantic models
│   └── predictors/     # AI Model implementations
├── main.py             # FastAPI entry point
```

## Installation

```bash
pip install .
```

## Usage

1. Update `config.json` with your model paths and settings.
2. Run the pipeline:

```bash
python main.py
```
