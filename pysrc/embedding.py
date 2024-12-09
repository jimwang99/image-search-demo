from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class MultiModalEmbeddingServer(ABC):
    """Multi-modal embedding server, abstract class"""

    @abstractmethod
    def generate_embedding_for_image(self, image: torch.Tensor) -> torch.Tensor:
        """Generate embedding for image

        Args:
            image (torch.Tensor): image tensor

        Returns:
            torch.Tensor: embedding tensor
        """
        pass

    @abstractmethod
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for embedding generation

        Args:
            image (torch.Tensor): image tensor

        Returns:
            torch.Tensor: preprocessed image tensor
        """
        pass

    @abstractmethod
    def generate_embedding_for_text(self, text: str) -> torch.Tensor:
        """Generate embedding for text

        Args:
            text (str): text string

        Returns:
            torch.Tensor: embedding tensor
        """
        pass


# class GoogleCloudEmbeddingServer(MultiModalEmbeddingServer):
#     """Google Cloud embedding server
#     https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
#     """

#     def generate_embedding_for_image(self, image: torch.Tensor) -> torch.Tensor:
#         pass

#     def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
#         pass

#     def generate_embedding_for_text(self, text: str) -> torch.Tensor:
#         pass


class OpenCLIPEmbeddingServer(MultiModalEmbeddingServer):
    """OpenCLIP embedding server
    https://github.com/mlfoundations/open_clip
    """

    def generate_embedding_for_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def generate_embedding_for_text(self, text: str) -> torch.Tensor:
        pass
