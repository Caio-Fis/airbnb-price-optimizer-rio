"""
Feature extraction via CLIP (HuggingFace).
Gera scores semânticos (luxo, limpeza, iluminação, estilo) e embeddings reduzidos.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

IMAGES_PATH = Path(os.getenv("IMAGES_PATH", "data/images"))
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 20  # dimensões após PCA

# Prompts para scoring semântico
QUALITY_PROMPTS = {
    "luxury_score": [
        "a luxurious, high-end apartment with elegant decoration",
        "a cheap, poorly decorated room",
    ],
    "cleanliness_score": [
        "a clean, well-organized and tidy room",
        "a messy, dirty, disorganized space",
    ],
    "brightness_score": [
        "a bright room with natural light and large windows",
        "a dark, dimly lit room",
    ],
    "professional_photo_score": [
        "a professional real estate photograph with perfect composition",
        "a blurry, amateur photo taken with a phone",
    ],
    "modern_style_score": [
        "a modern, contemporary apartment with stylish furniture",
        "an old-fashioned, outdated interior design",
    ],
}


class CLIPFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.eval()

    def _load_image(self, path: Path) -> Image.Image | None:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open image {path}: {e}")
            return None

    def _compute_semantic_scores(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> dict[str, list[float]]:
        scores = {key: [] for key in QUALITY_PROMPTS}

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            for score_name, (pos_prompt, neg_prompt) in QUALITY_PROMPTS.items():
                inputs = self.processor(
                    text=[pos_prompt, neg_prompt],
                    images=batch,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits_per_image  # shape: (batch, 2)
                    probs = logits.softmax(dim=1)[:, 0].cpu().numpy()  # prob do prompt positivo

                scores[score_name].extend(probs.tolist())

        return scores

    def _compute_embeddings(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                embs = self.model.get_image_features(**inputs)
                embs = embs / embs.norm(dim=-1, keepdim=True)  # normalizar L2
            all_embeddings.append(embs.cpu().numpy())

        return np.vstack(all_embeddings)

    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(IMAGES_PATH.glob("*.jpg"))

        if not image_paths:
            logger.warning("No images found. Skipping CLIP extraction.")
            return str(PROCESSED_DATA_PATH / "clip_features.parquet")

        logger.info(f"Processing {len(image_paths)} images with CLIP...")
        listing_ids, valid_images = [], []
        for p in image_paths:
            img = self._load_image(p)
            if img is not None:
                listing_ids.append(int(p.stem))
                valid_images.append(img)

        # Scores semânticos
        scores = self._compute_semantic_scores(valid_images)

        # Embeddings + PCA
        embeddings = self._compute_embeddings(valid_images)
        pca = PCA(n_components=EMBEDDING_DIM, random_state=42)
        emb_reduced = pca.fit_transform(embeddings)
        emb_cols = {f"clip_emb_{i}": emb_reduced[:, i] for i in range(EMBEDDING_DIM)}

        df = pd.DataFrame({"listing_id": listing_ids, **scores, **emb_cols})

        output_path = PROCESSED_DATA_PATH / "clip_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"CLIP features saved to {output_path} — shape: {df.shape}")
        return str(output_path)
