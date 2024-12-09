from __future__ import annotations

import torch
import numpy as np
import open_clip

from loguru import logger


class OpenCLIPEmbeddingServer:
    """OpenCLIP embedding server
    https://github.com/mlfoundations/open_clip

    >>> svr = OpenCLIPEmbeddingServer()
    >>> svr.get_embedding_dimension()
    512
    >>> img0 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    >>> emb0 = svr.generate_embedding_for_image(img0)
    >>> emb0.shape
    (512,)
    >>> emb0.dtype
    dtype('float32')
    >>> img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    >>> emb1 = svr.generate_embedding_for_image(img1)
    >>> emb1.shape
    (512,)
    >>> emb0.dtype
    dtype('float32')
    """

    def __init__(self, model_name: tuple[str, str] = ("ViT-B-32-quickgelu", "openai")):
        self.model_name = model_name
        assert (
            self.model_name in open_clip.list_pretrained()
        ), f"Invalid model name: {model_name}"

        logger.info(f"Creating OpenCLIP model: {model_name}")
        logger.warning("This may take a while to download model from HuggingFace ...")
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.model_name[0], self.model_name[1]
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name[0])

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        e = self.generate_embedding_for_text("Hello, world!")
        self.embedding_dimension = e.shape[-1]

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def generate_embedding_for_image(self, image: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            logger.trace(f"Preprocessing image {image.shape=}")
            pp = self.preprocess(torch.from_numpy(image)).unsqueeze(0).to(self.device)
            logger.trace(f"Encode image {pp.shape=}")
            e = self.model.encode_image(pp, normalize=True).squeeze().cpu().numpy()
        return e

    def generate_embedding_for_text(self, text: str) -> np.ndarray:
        with torch.inference_mode():
            t = self.tokenizer([text]).to(self.device)
            e = self.model.encode_text(t, normalize=True).squeeze().cpu().numpy()
        return e


if __name__ == "__main__":
    import doctest

    doctest.testmod()
