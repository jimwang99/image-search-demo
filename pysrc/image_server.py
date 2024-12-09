import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from loguru import logger


class LocalImageServer:
    def __init__(self, root_dpath: str):
        self.root_dpath = Path(root_dpath)
        assert self.root_dpath.is_dir(), f"Invalid directory: {root_dpath}"
        logger.info(f"Image root directory = {self.root_dpath}")

        self._all_fpaths = list(self.root_dpath.glob("*.jpg"))
        logger.info(f"Found {len(self._all_fpaths)} images")

    def __len__(self):
        return len(self._all_fpaths)

    def get_all_images(self) -> Iterable[np.ndarray]:
        for fpath in self._all_fpaths:
            yield (fpath, cv2.imread(str(fpath)))

    def get_one_random_image(self) -> np.ndarray:
        fpath = random.choices(self._all_fpaths)[0]
        return (fpath, cv2.imread(str(fpath)))
