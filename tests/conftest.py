import os
from pathlib import Path

import pytest


@pytest.fixture
def output_path() -> Path:
    output_dir = Path("tests_output")
    if not output_dir.exists():
        os.makedirs(output_dir)
    yield output_dir
