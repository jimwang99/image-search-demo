from __future__ import annotations

import numpy as np


class OpenCLIPEmbeddingServer:
    """OpenCLIP embedding server
    https://github.com/mlfoundations/open_clip
    """

    def __init__(self, model_name: tuple[str, str]):
        self.model_name = model_name

    def generate_embedding_for_image(self, image: np.ndarray) -> np.ndarray:
        pass

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        pass

    def generate_embedding_for_text(self, text: str) -> np.ndarray:
        pass
