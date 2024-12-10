from __future__ import annotations

import torch
import numpy as np
import open_clip
from PIL import Image

from loguru import logger


class OpenCLIPEmbeddingServer:
    """OpenCLIP embedding server
    https://github.com/mlfoundations/open_clip

    >>> svr = OpenCLIPEmbeddingServer()
    >>> svr.get_embedding_dimension()
    768
    >>> img0 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    >>> emb0 = svr.generate_embedding_for_image(img0)
    >>> emb0.shape[0] == svr.get_embedding_dimension()
    True
    >>> emb0.dtype
    dtype('float32')
    >>> img1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    >>> emb1 = svr.generate_embedding_for_image(img1)
    >>> emb0.shape[0] == svr.get_embedding_dimension()
    True
    >>> emb0.dtype
    dtype('float32')
    """

    def __init__(
        self, model_name: tuple[str, str] = ("ViT-L-14-336-quickgelu", "openai")
    ):
        """Initialize OpenCLIP embedding server

        Args:
            model_name (tuple[str, str], optional): prtrained model name and author. Defaults to ("ViT-L-14-336-quickgelu", "openai").
        """
        self.model_name = model_name
        assert (
            self.model_name in open_clip.list_pretrained()
        ), f"Invalid model name: {model_name}"

        logger.info(f"Creating OpenCLIP model: {model_name}")
        logger.warning(
            "This may take a while to download model from HuggingFace for the first time"
        )
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
        """Get the embedding dimension according to the model

        Returns:
            int: embedding dimension
        """
        return self.embedding_dimension

    def generate_embedding_for_image(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding for an image

        Args:
            image (np.ndarray): numpy array of the image

        Returns:
            np.ndarray: numpy array of the embedding
        """
        with torch.inference_mode():
            pp = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
            e = self.model.encode_image(pp, normalize=True).squeeze().cpu().numpy()
        return e

    def generate_embedding_for_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text

        Args:
            text (str): text string

        Returns:
            np.ndarray: numpy array of the embedding
        """
        with torch.inference_mode():
            t = self.tokenizer([text]).to(self.device)
            e = self.model.encode_text(t, normalize=True).squeeze().cpu().numpy()
        return e


if __name__ == "__main__":
    import doctest

    doctest.testmod()
