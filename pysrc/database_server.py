from numpy import np
from loguru import logger


class MilvusDatabaseServer:
    def __init__(self, uri: str) -> None:
        pass

    def connect(self) -> None:
        pass

    def insert(self, embedding: np.ndarray, image_uri: str) -> None:
        pass

    def search(self, embedding: np.ndarray, top_k: int) -> None:
        pass

    def delete(self, image_id: str) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_size(self) -> int:
        return 0
