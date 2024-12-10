import pprint
from pathlib import Path

import cv2
import numpy as np
import torch
from config import Config
from database_server import MilvusLocalServer
from embedding_server import OpenCLIPEmbeddingServer
from image_server import ImageLocalServer
from loguru import logger


class BackendServer:
    def __init__(self, config: Config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config)
        self.config = config

        logger.info("Initialize embedding server")
        if self.config.use_open_clip:
            self.embedding_server = OpenCLIPEmbeddingServer(
                self.config.open_clip_model_name
            )
        else:
            raise ValueError("Only support OpenCLIP for now")

        logger.info("Initialize database server")
        if self.config.use_milvus and self.config.use_local_database:
            self.database_server = MilvusLocalServer(
                self.config.local_database_fpath,
                self.embedding_server.get_embedding_dimension(),
            )
        else:
            raise ValueError("Only support local Milvus for now")

        logger.info("Initialize image server")
        if self.config.use_local_image:
            self.image_server = ImageLocalServer(self.config.local_image_dpath)
        else:
            raise ValueError("Only support local images for now")

        self._check()

    def _check(self):
        assert (
            self.database_server.size() == self.image_server.size()
        ), f"Database size {self.database_server.size()} != Image size {self.image_server.size()}"

    def get_database_size(self) -> int:
        return self.database_server.size()

    def load_image(self, fpath: Path) -> np.ndarray:
        assert fpath.is_file(), f"Invalid image file path: {fpath}"
        image = cv2.imread(str(fpath))
        assert image is not None, f"Invalid image: {image}"
        return image

    def insert_image(self, image: np.ndarray | Path) -> int:
        """Insert an image

        Args:
            image (np.ndarray | Path): numpy array of the image or image path

        Returns:
            int: image ID
        """
        if isinstance(image, Path):
            image = self.load_image(image)

        assert isinstance(image, np.ndarray), f"Invalid image type: {type(image)}"
        assert len(image.shape) == 3, f"Invalid image shape: {image.shape}"
        assert image.shape[-1] == 3, f"Invalid image channel: {image.shape[-1]}"
        assert image.dtype == np.uint8, f"Invalid image dtype: {image.dtype}"

        emb = self.embedding_server.generate_embedding_for_image(image)
        id = self.database_server.insert(emb)
        _ = self.image_server.insert(image, id)

        return id

    def delete_image(self, id: int) -> None:
        self.database_server.delete(id)
        self.image_server.delete(id)

    def get_image(self, id: int) -> np.ndarray:
        """Get the image content in numpy array

        Args:
            id (int): image ID

        Returns:
            np.ndarray: numpy array of the image
        """
        return self.image_server.get(id)

    def search_with_image(self, image: np.ndarray | Path, top_k: int) -> list[int]:
        if isinstance(image, Path):
            image = self.load_image(image)
        emb = self.embedding_server.generate_embedding_for_image(image)
        results = self.database_server.search(emb, top_k)
        return [r[0] for r in results]

    def search_with_text(self, text: str, top_k: int) -> list[int]:
        emb = self.embedding_server.generate_embedding_for_text(text)
        results = self.database_server.search(emb, top_k * 2, distance_threshold=0.0)
        ids = [r[0] for r in results]
        distances = torch.tensor([100.0 * r[1] for r in results], dtype=torch.float32)
        softmaxs = torch.nn.functional.softmax(distances, dim=0)
        result_ids = []
        for id, dist, sm in zip(ids, distances, softmaxs):
            logger.trace(f"{id=} distance={dist} softmax={sm}")
            if sm >= 0.01:
                result_ids.append(id)
        return result_ids[:top_k]


if __name__ == "__main__":
    model_name = "ViT-L-14-336-quickgelu"
    model_author = "openai"
    config = Config(
        root_dpath=Path(__file__).parent.parent,
        use_local_database=True,
        local_database_relative_fpath=Path(f"data/{model_name}.{model_author}.db"),
        use_local_image=True,
        local_image_relative_dpath=Path("data/images"),
        use_open_clip=True,
        open_clip_model_name=(model_name, model_author),
        test_image_relative_dpath=Path("data/inputs/val2017"),
    )
    server = BackendServer(config)

    from tqdm import tqdm

    for image_fpath in tqdm(config.test_image_dpath.glob("*.jpg")):
        id = server.insert_image(image_fpath)
