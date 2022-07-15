from logging import getLogger

import os
from pathlib import Path
import requests  # type: ignore[import]

from tqdm import tqdm

log = getLogger(os.path.relpath(__file__))


def create_folder_safe(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def download_to_file(url: Path | str, out_file: Path | str):
    if type(url) is Path:
        url = url.as_posix()

    if type(out_file) is Path:
        out_file = out_file.as_posix()

    create_folder_safe(os.path.dirname(out_file))
    response = requests.get(url, stream=True)  # type: ignore[arg-type]
    size = int(response.headers.get("content-length", 0))
    progress = tqdm(total=size, unit="iB", unit_scale=True)

    with open(out_file, "wb") as f:
        for data in response.iter_content(chunk_size=1024):
            progress.update(len(data))
            f.write(data)

    if os.path.getsize(out_file) != size:
        log.critical("{} is corrupt".format(out_file))
        exit(50)
