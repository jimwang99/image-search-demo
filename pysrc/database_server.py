from pathlib import Path

import numpy as np
from loguru import logger
from pymilvus import MilvusClient


class MilvusLocalServer:
    """A local Milvus server for storing multimodal embeddings
    >>> svr = MilvusLocalServer(Path("/tmp/test_milvus.db"), 768, overwrite=True)
    >>> svr.get_database_size()
    0
    >>> from embedding_server import OpenCLIPEmbeddingServer
    >>> embedding_server = OpenCLIPEmbeddingServer()
    >>> img0 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    >>> emb0 = embedding_server.generate_embedding_for_image(img0)
    >>> svr.insert(emb0, "test0.jpg")
    >>> svr.get_database_size()
    1
    >>> img1 = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    >>> emb1 = embedding_server.generate_embedding_for_image(img1)
    >>> svr.insert(emb1, "test1.jpg")
    >>> svr.get_database_size()
    2
    """

    def __init__(
        self, database_fpath: Path, embedding_dimension: int, overwrite: bool = False
    ) -> None:
        logger.info(f"Initialize Milvus database with local file {database_fpath}")
        if database_fpath.exists():
            logger.warning(f"Database file {database_fpath} already exists")
            if overwrite:
                logger.warning(f"Overwrite the existing database file {database_fpath}")
                database_fpath.unlink()
            else:
                logger.warning(f"Use the existing database file {database_fpath}")
        self.client = MilvusClient(uri=str(database_fpath))

        self.collection_name = "multimodal_embeddings"
        if not self.client.has_collection(collection_name="multimodal_embeddings"):
            logger.info(f"Create a collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vector_field_name="embedding",
                dimension=embedding_dimension,
                auto_id=True,
                metric_type="COSINE",
                enable_dynamic_filed=True,
            )
        else:
            logger.warning(
                f"Collection {self.collection_name} already exists, use the existing collection"
            )

    def get_database_size(self) -> int:
        result = self.client.get_collection_stats(self.collection_name)
        return result["row_count"]

    def insert(self, embedding: np.ndarray, image_uri: str) -> None:
        result = self.client.insert(
            self.collection_name, {"embedding": embedding, "uri": image_uri}
        )
        logger.debug(f"Insert result: {result}")

    def search(self, embedding: np.ndarray, top_k: int) -> None:
        pass

    def delete(self, image_id: str) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_size(self) -> int:
        return 0


if __name__ == "__main__":
    import doctest

    doctest.testmod()
