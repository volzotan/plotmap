from pathlib import Path

import requests
import shutil

def download_file(url: str, filename: Path) -> None:

    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

