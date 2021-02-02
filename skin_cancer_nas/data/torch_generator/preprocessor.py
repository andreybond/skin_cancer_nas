import os
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Union, Dict
import re
from collections import Counter
from tqdm import tqdm
import logging as log
logger = log.getLogger('nni')

from config import ROOT_PATH, MELANOMA_CLASSES, PROCESSED_PATH, POS_PATH, NEG_PATH

FOLDER_NAMES = os.listdir(str(ROOT_PATH))

os.makedirs(str(PROCESSED_PATH), exist_ok=True)
os.makedirs(str(POS_PATH), exist_ok=True)
os.makedirs(str(NEG_PATH), exist_ok=True)

D_RE = re.compile('([0-9]+)')


def group_samples(source_zip: bool = True):
    """
    Moves (can copy) samples from original source to positive (melanoma) and negative (others) folders
    under PROCESSED_PATH. Renames samples by unique IDs.
    :param source_zip: flag indicating if data read from zip source or not
    """
    file_id = 1
    temp_dir = tempfile.mkdtemp()
    for folder_name in FOLDER_NAMES:
        folder_path = ROOT_PATH / folder_name
        if source_zip:
            with zipfile.ZipFile(folder_path) as zip_file:
                zip_file.extractall(temp_dir)
                for subfolder in os.listdir(temp_dir):
                    file_id = _move_folder(file_id, subfolder, temp_dir)
        else:
            for subfolder in os.listdir(folder_path):
                file_id = _move_folder(file_id, subfolder, folder_path)

    num_pos = len(os.listdir(POS_PATH))
    num_neg = len(os.listdir(NEG_PATH))
    pos_pct = round(num_pos / (num_pos + num_neg) * 100, 1)
    neg_pct = round(num_neg / (num_pos + num_neg) * 100, 1)
    logger.info("# positive: {} ({} %)".format(num_pos, pos_pct))
    logger.info("# negative: {} ({} %)".format(num_neg, neg_pct))


def _move_folder(file_id: int, subfolder: str, src: Union[Path, str]) -> int:
    """
    Auxiliary function for moving file from source to destination folder
    :param file_id: unique file ID
    :param subfolder: folder name
    :param src: source path
    :return: next file ID
    """
    subfolder_path = os.path.join(src, subfolder)
    if subfolder.lower() in MELANOMA_CLASSES:
        for sample in os.listdir(subfolder_path):
            # logger.info('melanoma sample copying: {}'.format(os.path.join(subfolder_path, sample)))
            shutil.copytree(os.path.join(subfolder_path, sample), POS_PATH / str(file_id))
            file_id += 1
    else:
        for sample in os.listdir(subfolder_path):
            shutil.copytree(os.path.join(subfolder_path, sample), NEG_PATH / str(file_id))
            file_id += 1
    return file_id


def clean_samples():
    """
    Removes redundant images and calculates number of channels per each sample
    """
    samples_num = {}
    for class_path in [POS_PATH, NEG_PATH]:
        for sample in tqdm(os.listdir(class_path)):
            sample_path = class_path/sample
            for channel in os.listdir(sample_path):
                channel_path = sample_path/channel
                if not os.path.isdir(channel_path):
                    os.remove(channel_path)
                    continue
                if len(os.listdir(channel_path)) > 1:
                    _select_image(channel_path)
            samples_num[sample] = len(os.listdir(sample_path))
    for num_channels, num_samples in Counter(samples_num.values()).items():
        logger.info('{} channels occurs in {} samples'.format(num_channels, num_samples))


def _select_image(channel_path: Path):
    """
    Auxiliary function for removing redundant files by specified logic (leaves the first in alphanumeric order)
    :param channel_path: directory
    """
    images = sorted(os.listdir(channel_path), key=alpha_numeric_key)
    for image in images[1:]:
        os.remove(channel_path/image)


def alpha_numeric_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(D_RE, s)]


if __name__ == '__main__':
    group_samples(source_zip=False)
    clean_samples()
