import pprint

import numpy as np

from config import Config
from image_server import LocalImageServer
from embedding_server import OpenCLIPEmbeddingServer
from database_server import MilvusDatabaseServer

from loguru import logger


class BackendServer:
    def __init__(self, config: Config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(config)
        self.config = config

        logger.info("Initialize image server")
        if self.config.use_local_image:
            self.image_server = LocalImageServer(self.config.local_image_dpath)
        else:
            raise ValueError("Only support local images for now")

        logger.info("Initialize embedding server")
        if self.config.use_open_clip:
            self.embedding_server = OpenCLIPEmbeddingServer(
                self.config.open_clip_model_name
            )
        else:
            raise ValueError("Only support OpenCLIP for now")

        logger.info("Initialize database server")
        if self.config.use_milvus and self.config.use_local_database:
            self.database_server = MilvusDatabaseServer(
                uri=str(self.config.local_database_fpath)
            )
        else:
            raise ValueError("Only support local Milvus for now")

    def get_database_size(self) -> int:
        return self.database_server.get_size()

    def insert_image(self, uri: str, ndarry: np.ndarray) -> None:
        for fpath, image in self.image_server.get_all_images():
            embedding = self.embedding_server.generate_embedding_for_image(image)
            self.database_server.insert(embedding, str(fpath))
        pass

    def delete_image(self, uri: str) -> None:
        pass

    def search_with_image(self, ndarray: np.ndarray, top_k: int) -> list[str]:
        results: list[str] = []
        return results

    def search_wth_text(self, text: str, top_k: int) -> list[str]:
        results: list[str] = []
        return results
