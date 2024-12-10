import os
import time
from pathlib import Path

import itertools
import numpy as np
from loguru import logger
from pymilvus import MilvusClient


class MilvusLocalServer:
    """A local Milvus server for storing multimodal embeddings
    >>> svr = MilvusLocalServer(Path("/tmp/test_milvus.db"), 768)
    >>> svr.size()
    0
    >>> from embedding_server import OpenCLIPEmbeddingServer
    >>> embedding_server = OpenCLIPEmbeddingServer()
    >>> img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    >>> emb = embedding_server.generate_embedding_for_image(img)
    >>> id = svr.insert(emb)
    >>> svr.size()
    1
    >>> img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    >>> emb = embedding_server.generate_embedding_for_image(img)
    >>> id = svr.insert(emb)
    >>> svr.size()
    2
    >>> emb = embedding_server.generate_embedding_for_text("Hello!")
    >>> id_hello = svr.insert(emb)
    >>> svr.size()
    3
    >>> emb = embedding_server.generate_embedding_for_text("How are you?")
    >>> id_how_are_you = svr.insert(emb)
    >>> svr.size()
    4
    >>> emb = embedding_server.generate_embedding_for_text("Give me a lever long enough and a fulcrum on which to place it, and I shall move the world.")
    >>> id = svr.insert(emb)
    >>> svr.size()
    5
    >>> emb = embedding_server.generate_embedding_for_text("Knowledge is power.")
    >>> id = svr.insert(emb)
    >>> svr.size()
    6
    >>> emb = embedding_server.generate_embedding_for_text("Hello!")
    >>> results = svr.search(embedding=emb, top_k=3)
    >>> len(results)
    2
    >>> results[0][0] == id_hello
    True
    >>> results[1][0] == id_how_are_you
    True
    >>> svr.size()
    6
    >>> svr.delete(id_hello)
    >>> svr.size()
    5
    >>> os.unlink("/tmp/test_milvus.db")
    """

    def __init__(self, database_fpath: Path, embedding_dimension: int) -> None:
        """Initialize Milvus database with local file

        Args:
            database_fpath (Path): file path of the Milvus database
            embedding_dimension (int): dimension of the embedding
        """
        logger.info(f"Initialize Milvus database with local file {database_fpath}")
        if database_fpath.exists():
            logger.warning(f"Database file {database_fpath} already exists")
        else:
            os.makedirs(database_fpath.parent, exist_ok=True)
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
                enable_dynamic_filed=False,
            )
        else:
            logger.warning(
                f"Collection {self.collection_name} already exists, use the existing collection"
            )

    def size(self) -> int:
        """Get the number of entities in the database

        Returns:
            int: number of entities
        """
        result = self.client.get_collection_stats(self.collection_name)
        return result["row_count"]

    def insert(self, embedding: np.ndarray) -> int:
        """Insert an embedding into the database

        Args:
            embedding (np.ndarray): numpy array of the embedding to be inserted

        Returns:
            int: ID of the inserted embedding
        """
        logger.trace(f"Inserting embedding {embedding[0]=}")
        result = self.client.insert(self.collection_name, {"embedding": embedding})
        assert result["insert_count"] == 1, f"Insert failed: {result}"
        return result["ids"][0]

    def search(
        self, embedding: np.ndarray, top_k: int, distance_threshold: float = 0.9
    ) -> list[int, float]:
        """Search for embeddings in the database

        Args:
            embedding (np.ndarray): numpy array of the embedding to be searched
            top_k (int): number of results to return
            distance_threshold (float, optional): threshold of the distance metric. Defaults to 0.9.

        Returns:
            list[int]: list of IDs of the search results
        """
        groups = self.client.search(
            self.collection_name,
            data=[embedding],
            group_size=top_k,
            strict_group_size=True,
        )
        itr = (
            (hit["id"], hit["distance"])
            for group in groups
            for hit in group
            if hit["distance"] >= distance_threshold
        )
        results = list(itertools.islice(itr, top_k))
        if len(results) == 0 or len(results) < top_k:
            logger.warning(
                f"No enough results found for {embedding[0]=}, and here are all the hits"
            )
            for group in groups:
                for hit in group:
                    logger.warning(f"{hit}")
        logger.trace(f"{top_k=} {distance_threshold=} {results=}")
        return results

    def delete(self, id: int) -> None:
        """Delete an entity from the database

        Args:
            id (int): ID of the entity to be deleted
        """
        self.client.delete(self.collection_name, ids=[id])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
