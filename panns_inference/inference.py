from logging import getLogger

import os
from pathlib import Path

import torch


# from gcloud import upload_file
from .config import labels, classes_num, panns_path
from .pytorch_utils import move_data_to_device
from .models import Cnn14, Cnn14_DecisionLevelMax

log = getLogger(__name__)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split("/")[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def download_to_file(url: Path | str, out_file: Path | str):
    if type(url) is Path:
        url = url.as_posix()


class AudioTagging(object):
    def __init__(self, model=None, checkpoint_path=None, model_url=None, device="cuda"):
        """Audio tagging inference wrapper."""

        if not model_url:
            model_url = (
                "https://zenodo.org"
                + "/record/3987831/files/"
                + "Cnn14_mAP%3D0.431.pth?download=1"
            )
        if not checkpoint_path:
            checkpoint_path = panns_path / "Cnn14_mAP=0.431.pth"
        log.info("Checkpoint path: {}".format(checkpoint_path))
        if (
            not os.path.exists(checkpoint_path)
            or os.path.getsize(checkpoint_path) < 3e8
        ):
            download_to_file(model_url, checkpoint_path)
            # if config.gcloud.gcloud:
            #    #upload_file(checkpoint_path, config.gcloud.panns_path)

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=self.classes_num,
            )
        else:
            self.model = model

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        # Parallel
        if "cuda" in str(self.device):
            self.model.to(self.device)
            log.info("GPU number: {}".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            log.info("Using CPU.")

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        clipwise_output = output_dict["clipwise_output"].data.cpu().numpy()
        embedding = output_dict["embedding"].data.cpu().numpy()

        return clipwise_output, embedding


class SoundEventDetection(object):
    def __init__(self, model=None, checkpoint_path=None, device="cuda"):

        # assert False, "This needs to be fixed first"

        """Sound event detection inference wrapper."""
        if not checkpoint_path:
            checkpoint_path = "{}/panns_data/Cnn14_DecisionLevelMax.pth".format(
                str(Path.home())
            )
        print("Checkpoint path: {}".format(checkpoint_path))
        url = (
            "https://zenodo.org/"
            + "record/3987831/files/"
            + "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
        )

        if (
            not os.path.exists(checkpoint_path)
            or os.path.getsize(checkpoint_path) < 3e8
        ):
            download_to_file(url, checkpoint_path)

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14_DecisionLevelMax(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=self.classes_num,
            )
        else:
            self.model = model

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

        # Parallel
        if "cuda" in str(self.device):
            self.model.to(self.device)
            print("GPU number: {}".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print("Using CPU.")

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        framewise_output = output_dict["framewise_output"].data.cpu().numpy()

        return framewise_output
