import unittest

import cv2
import numpy as np
from pathlib import Path
from config import Config
from backend import BackendServer


class TestBackend(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            root_path=Path(__file__).parent,
            use_local_database=True,
            local_database_relative_fpath=Path("data/database"),
            use_local_image=True,
            local_image_relative_dpath=Path("data/images/val2017"),
            test_with_empty_database=True,
            test_image_relative_dpath=Path("data/images/val2017"),
        )
        self.backend_server = BackendServer(self.config)
        self.image_server = self.backend_server.image_server

    def test_insert_and_delete_image(self):
        uris = []
        for _ in range(10):
            uri, ndarray = self.image_server.get_random_image()
            self.backend_server.insert_image(uri, ndarray)
            uris.append(uri)
            cnt = self.backend_server.get_database_size()
            self.assertEqual(cnt, len(uris))

        while len(uris) > 0:
            uri = uris.pop()
            self.backend_server.delete_image(uri)

            cnt = self.backend_server.get_database_size()
            self.assertEqual(cnt, len(uris))

    def _insert_images(self, num_images: int) -> list[tuple[str, np.ndarray]]:
        images = []
        for _ in range(num_images):
            uri, ndarray = self.image_server.get_random_image()
            self.backend_server.insert_image(uri, ndarray)
            images.append((uri, ndarray))
        return images

    def test_search_with_existing_image(self):
        self.assertEqual(self.backend_server.get_database_size(), 0)

        images = self._insert_images(10)
        for uri, ndarray in images:
            results = self.backend_server.search_with_image(ndarray, top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], uri)

    def test_search_with_text(self):
        self.assertEqual(self.backend_server.get_database_size(), 0)

        images = self._insert_images(10)

        # add target image into database
        image_fpath = self.config.test_image_dpath / "000000001000.jpg"
        if str(image_fpath) not in [image[0] for image in images]:
            self.backend_server.insert_image(cv2.imread(str(image_fpath)))
        self.assertEqual(self.backend_server.get_database_size(), 11)

        results = self.backend_server.search_with_text(
            """a group of kids posing for a picture on a tennis court.
a group of young children standing next to each other.
a large family poses for picture on tennis court
a group of people that are standing near a tennis net.
the people are posing for a group photo.""",
            top_k=5,
        )
        self.assertEqual(len(results), 5)
        self.assertIn(str(image_fpath), [result[0] for result in results])
