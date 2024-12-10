import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from loguru import logger


class ImageLocalServer:
    """A local image server for storing images

    >>> # doc test
    >>> import os
    >>> os.makedirs("/tmp/test_images", exist_ok=True)
    >>> svr = ImageLocalServer("/tmp/test_images")
    >>> svr.size()
    0
    >>> img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    >>> id = 1
    >>> uri = svr.insert(img, id)
    >>> svr.size()
    1
    >>> svr.has(id)
    True
    >>> img1 = svr.get(id)
    >>> np.array_equal(img, img1)
    True
    >>> svr.delete(id)
    >>> svr.size()
    0
    """

    def __init__(self, root_dpath: str):
        """Initialize image server on local file system

        Args:
            root_dpath (str): root directory path
        """
        self.root_dpath = Path(root_dpath)
        os.makedirs(self.root_dpath, exist_ok=True)
        assert self.root_dpath.is_dir(), f"Invalid directory: {root_dpath}"
        logger.info(f"Image root directory = {self.root_dpath}")

        self._all_fpaths = list(self.root_dpath.glob("*.png"))
        logger.info(f"Found {len(self._all_fpaths)} images")

    def size(self) -> int:
        """Get the number of images on the server

        Returns:
            int: number of images
        """
        return len(self._all_fpaths)

    def has(self, id: int) -> bool:
        """Check if the image exists

        Args:
            id (int): image ID

        Returns:
            bool: True if the image exists
        """
        p = self.root_dpath / f"{id}.png"
        return p in self._all_fpaths

    def get_uri(self, id: int) -> str:
        """Get the URI of the image, which is the file path

        Args:
            id (int): image ID

        Returns:
            str: file path of the image
        """
        return str(self.root_dpath / f"{id}.png")

    def get_id(self, uri: str) -> int:
        """Get the ID of the image, given the file path

        Args:
            uri (str): file path of the image

        Returns:
            int: image ID
        """
        return int(Path(uri).stem)

    def insert(self, image: np.ndarray, id: int) -> str:
        """Insert an image to the server

        Args:
            image (np.ndarray): numpy array of the image
            id (int): image ID, from database server

        Returns:
            str: file path of the image
        """
        assert len(image.shape) == 3, f"Invalid image shape: {image.shape}"
        assert image.shape[-1] == 3, f"Invalid image channel: {image.shape[-1]}"
        assert image.dtype == np.uint8, f"Invalid image dtype: {image.dtype}"

        uri = self.get_uri(id)
        cv2.imwrite(str(uri), image)
        self._all_fpaths.append(Path(uri))
        logger.trace(f"Inserted image: {uri}, count = {self.size()}")
        return str(uri)

    def delete(self, id: int) -> None:
        """Delete an image from the server by ID

        Args:
            id (int): image unique ID
        """
        assert self.has(id), f"Image not found: {id=}"
        p = Path(self.get_uri(id))
        p.unlink()
        self._all_fpaths.remove(p)
        logger.trace(f"Deleted image: {p}, count = {self.size()}")

    def get(self, id: int) -> np.ndarray:
        """Get the image from the server by ID

        Args:
            id (int): image ID

        Returns:
            np.ndarray: numpy array of the image, 3D with shape (height, width, channel)
        """
        assert self.has(id), f"Image not found: {id=}"
        return cv2.imread(self.get_uri(id))

    def all_uri(self) -> Iterable[str]:
        """Get all images' URIs, which are the file paths

        Yields:
            Iterator[str]: file path of the image in iterator form
        """
        for uri in map(str, self._all_fpaths):
            yield uri

    def all_id(self) -> Iterable[int]:
        """Get all images' IDs

        Yields:
            Iterator[int]: image ID in iterator form
        """
        for p in self._all_fpaths:
            yield int(p.stem)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
