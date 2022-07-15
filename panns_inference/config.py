from logging import getLogger

from typing import List

from pathlib import Path
import os
import csv

from utils import download_to_file


log = getLogger(__name__)

labels: List[str]
panns_path: Path
sample_rate = 32000
classes_num: int
lb_to_ix: dict
ix_to_lb: dict
id_to_ix: dict
labels_csv_path: Path


def init(panns_path_in: Path):
    global labels, panns_path, classes_num, labels_csv_path
    global lb_to_ix, ix_to_lb, id_to_ix
    panns_path = panns_path_in
    log.info(f"panns_path: {panns_path.as_posix()}")

    labels_csv_path = panns_path / "class_labels_indices.csv"

    # Download labels if not exist
    if not os.path.isfile(labels_csv_path):
        __download_labels()

    lines = __load_labels()

    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)

    classes_num = len(labels)
    log.info("Loaded PANNS labels")

    lb_to_ix = {label: i for i, label in enumerate(labels)}
    ix_to_lb = {i: label for i, label in enumerate(labels)}

    id_to_ix = {id: i for i, id in enumerate(ids)}


def __download_labels():
    global labels_csv_path
    log.info("Downloading PANNS labels to {}".format(labels_csv_path.parent))
    os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)

    url = (
        "http://storage.googleapis.com/"
        + "us_audioset/youtube_corpus/v1/csv/"
        + "class_labels_indices.csv"
    )
    download_to_file(url, labels_csv_path)


def __load_labels():
    global labels_csv_path
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        return list(reader)
