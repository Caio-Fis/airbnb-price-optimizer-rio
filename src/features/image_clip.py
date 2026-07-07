"""
Feature extraction via CLIP (HuggingFace).
Gera scores semânticos (luxo, limpeza, iluminação, estilo) e embeddings reduzidos.

Otimizado para CPU: cada imagem é codificada UMA única vez; os scores saem do
cosseno entre o embedding da imagem e os embeddings dos prompts (pré-codificados),
que é matematicamente idêntico ao softmax de logits_per_image do CLIP.
As imagens são processadas em streaming (batches lidos do disco), nunca todas na RAM.
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
BATCH_SIZE = 32

# Prompts para scoring semântico (positivo, negativo)
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

    def _encode_prompts(self) -> torch.Tensor:
        """Embeddings L2-normalizados dos 2×N prompts, na ordem de QUALITY_PROMPTS.

        Usa pooler + projeção explicitamente (o retorno de get_text_features
        mudou no transformers 5.x).
        """
        texts = [p for pair in QUALITY_PROMPTS.values() for p in pair]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.text_model(**inputs)
            embs = self.model.text_projection(out.pooler_output)
        return embs / embs.norm(dim=-1, keepdim=True)  # (2N, 512)

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            embs = self.model.visual_projection(out.pooler_output)
        return embs / embs.norm(dim=-1, keepdim=True)  # (B, 512)

    def run(self) -> str:
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(IMAGES_PATH.glob("*.jpg"))
        output_path = PROCESSED_DATA_PATH / "clip_features.parquet"

        if not image_paths:
            logger.warning("No images found. Skipping CLIP extraction.")
            return str(output_path)

        logger.info(f"Processing {len(image_paths)} images with CLIP (batch={BATCH_SIZE})...")
        text_embs = self._encode_prompts()
        logit_scale = self.model.logit_scale.exp().item()

        listing_ids: list[int] = []
        all_scores: list[np.ndarray] = []
        all_embs: list[np.ndarray] = []

        batch_imgs: list[Image.Image] = []
        batch_ids: list[int] = []

        def _flush():
            if not batch_imgs:
                return
            img_embs = self._encode_image_batch(batch_imgs)          # (B, 512)
            # logits_per_image = logit_scale * img @ txt.T → softmax por par (pos, neg)
            logits = logit_scale * img_embs @ text_embs.T            # (B, 2N)
            logits = logits.reshape(len(batch_imgs), len(QUALITY_PROMPTS), 2)
            probs = logits.softmax(dim=-1)[:, :, 0]                  # prob do prompt positivo
            all_scores.append(probs.cpu().numpy())
            all_embs.append(img_embs.cpu().numpy())
            listing_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

        processed = 0
        for path in image_paths:
            img = self._load_image(path)
            if img is None:
                continue
            batch_imgs.append(img)
            batch_ids.append(int(path.stem))
            if len(batch_imgs) >= BATCH_SIZE:
                _flush()
            processed += 1
            if processed % 2000 == 0:
                logger.info(f"{processed:,}/{len(image_paths):,} imagens processadas")
        _flush()

        scores = np.vstack(all_scores)  # (N, 5)
        score_cols = {name: scores[:, i] for i, name in enumerate(QUALITY_PROMPTS)}

        # Embeddings + PCA
        embeddings = np.vstack(all_embs)  # (N, 512)
        pca = PCA(n_components=EMBEDDING_DIM, random_state=42)
        emb_reduced = pca.fit_transform(embeddings)
        emb_cols = {f"clip_emb_{i}": emb_reduced[:, i] for i in range(EMBEDDING_DIM)}

        df = pd.DataFrame({"listing_id": listing_ids, **score_cols, **emb_cols})
        df.to_parquet(output_path, index=False)
        logger.info(f"CLIP features saved to {output_path} — shape: {df.shape}")
        return str(output_path)
