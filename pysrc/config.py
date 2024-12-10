from __future__ import annotations
import tempfile
import os

from pathlib import Path
from pydantic.dataclasses import dataclass
from pydantic.types import DirectoryPath


@dataclass
class Config:
    root_dpath: DirectoryPath
    # database
    use_milvus: bool = True
    use_local_database: bool = True
    local_database_relative_fpath: Path | None = None
    # images
    use_local_image: bool = True
    local_image_relative_dpath: Path | None = None
    # embeddings
    use_open_clip: bool = True
    open_clip_model_name: tuple[str, str] = ("ViT-L-14-336-quickgelu", "openai")
    # tests
    test_with_empty_database: bool = False
    test_image_relative_dpath: Path | None = None

    def __post_init__(self):
        if self.use_local_database:
            assert self.local_database_relative_fpath is not None
            self.local_database_fpath = (
                self.root_dpath / self.local_database_relative_fpath
            )
        else:
            self.local_database_fpath = None
            raise ValueError("Only support local database for now")

        if self.use_local_image:
            assert self.local_image_relative_dpath is not None
            self.local_image_dpath = self.root_dpath / self.local_image_relative_dpath
        else:
            self.local_image_dpath = None
            raise ValueError("Only support local images for now")

        if self.test_image_relative_dpath is not None:
            self.test_image_dpath = self.root_dpath / self.test_image_relative_dpath
        else:
            self.test_image_dpath = None
