"""
Feature extraction via YOLOv8.
Detecta objetos relevantes ao preço: piscina, banheira, lareira, camas, sofás, etc.
"""
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger
from PIL import Image
from ultralytics import YOLO

IMAGES_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))

# Classes COCO relevantes para acomodações
RELEVANT_CLASSES = {
    "bed": "has_bed",
    "couch": "has_couch",
    "chair": "has_chair",
    "dining table": "has_dining_table",
    "tv": "has_tv",
    "refrigerator": "has_refrigerator",
    "microwave": "has_microwave",
    "oven": "has_oven",
    "sink": "has_sink",
    "toilet": "has_toilet",
    "laptop": "has_laptop",
    "potted plant": "has_plant",
    "clock": "has_clock",
    "vase": "has_vase",
    "book": "has_book",
}

# Score de "riqueza" dos objetos detectados (peso subjetivo)
LUXURY_WEIGHTS = {
    "bed": 0.5,
    "couch": 0.6,
    "dining table": 0.4,
    "tv": 0.3,
    "refrigerator": 0.2,
    "potted plant": 0.1,
    "vase": 0.2,
    "book": 0.1,
}


class YOLOFeatureExtractor:
    def __init__(self, model_size: str = "yolov8n.pt"):
        logger.info(f"Loading YOLO model: {model_size}")
        self.model = YOLO(model_size)  # baixa automaticamente se não existir

    def _process_image(self, path: Path) -> dict:
        features = {v: 0 for v in RELEVANT_CLASSES.values()}
        features["object_count"] = 0
        features["yolo_luxury_score"] = 0.0
        features["bed_count"] = 0

        try:
            results = self.model(str(path), verbose=False, conf=0.3)
            detected = defaultdict(int)

            for result in results:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls)]
                    detected[cls_name] += 1

            features["object_count"] = sum(detected.values())

            for cls_name, count in detected.items():
                feature_name = RELEVANT_CLASSES.get(cls_name)
                if feature_name:
                    features[feature_name] = min(count, 1)  # binário

            features["bed_count"] = detected.get("bed", 0)
            features["yolo_luxury_score"] = sum(
                LUXURY_WEIGHTS.get(cls, 0) * min(cnt, 1)
                for cls, cnt in detected.items()
            )

        except Exception as e:
            logger.warning(f"YOLO failed on {path}: {e}")

        return features

    def run(self, batch_size: int = 1) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(IMAGES_PATH.glob("*.jpg"))

        if not image_paths:
            logger.warning("No images found. Skipping YOLO extraction.")
            return str(PROCESSED_DATA_PATH / "yolo_features.parquet")

        logger.info(f"Processing {len(image_paths)} images with YOLO...")
        records = []
        for path in image_paths:
            feats = self._process_image(path)
            feats["listing_id"] = int(path.stem)
            records.append(feats)

        df = pd.DataFrame(records)
        output_path = PROCESSED_DATA_PATH / "yolo_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"YOLO features saved to {output_path} — shape: {df.shape}")
        return str(output_path)
