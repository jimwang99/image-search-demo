import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "pysrc"))

from backend import BackendServer
from config import Config


def test_backend():
    config = Config(
        root_dpath=Path(__file__).parent.parent,
        local_database_relative_fpath=Path("data/test/test.db"),
        local_image_relative_dpath=Path("data/test/images"),
        use_open_clip=True,
        open_clip_model_name=("ViT-L-14-336-quickgelu", "openai"),
        test_with_empty_database=True,
        test_image_relative_dpath=Path("data/inputs/val2017"),
    )
    backend_server = BackendServer(config)
    test_image_fpaths = list(config.test_image_dpath.glob("*.jpg"))

    assert backend_server.get_database_size() == 0

    # test insert image
    ids = []
    for fpath in test_image_fpaths[:10]:
        id = backend_server.insert_image(fpath)
        ids.append(id)
        cnt = backend_server.get_database_size()
        assert cnt == len(ids)

    # test search with existing image
    for id in ids:
        img = backend_server.get_image(id)
        results = backend_server.search_with_image(img, top_k=1)
        assert len(results) == 1
        assert results[0] == id

    # test search with text
    ## add target image into database
    image_fpath = config.test_image_dpath / "000000001000.jpg"
    assert image_fpath.is_file(), f"Invalid image file path: {image_fpath}"
    img = backend_server.load_image(image_fpath)
    id = backend_server.insert_image(img)
    ids.append(id)

    results = backend_server.search_with_text(
        """a group of kids posing for a picture on a tennis court.
        a group of young children standing next to each other.
        a large family poses for picture on tennis court
        a group of people that are standing near a tennis net.
        the people are posing for a group photo.""",
        top_k=5,
    )
    assert id in results

    # test delete image
    while len(ids) > 0:
        id = ids.pop()
        backend_server.delete_image(id)

        cnt = backend_server.get_database_size()
        assert cnt == len(ids)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main(["-s", "-vv", __file__]))
