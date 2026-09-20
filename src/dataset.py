"""
Functions to load the NUAA dataset, run MTCNN face alignment,
and build the train/validation/test split.
"""

import os
from PIL import Image
from facenet_pytorch import MTCNN

from src.config import (
    CLIENT_TRAIN_LIST,
    CLIENT_TEST_LIST,
    IMPOSTER_TRAIN_LIST,
    IMPOSTER_TEST_LIST,
    NUAA_DIR,
    MTCNN_IMAGE_SIZE,
    MTCNN_MARGIN,
    VALIDATION_SUBJECTS,
    LABEL_LIVE,
    LABEL_SPOOF,
)


def _read_protocol_file(path):
    """
    Reads one of the NUAA official protocol .txt files.
    Each line is a path like:
        /kaggle/input/nuaaaa/raw/ClientRaw/0010/0010_01_05_03_115.jpg
    Returns a list of (subject_id, relative_path) tuples, where
    relative_path is relative to NUAA_DIR (e.g. "ClientRaw/0010/....jpg").
    """
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    entries = []
    for line in lines:
        filename = os.path.basename(line)
        subject_id = filename.split('_')[0]

        # Keep only the part starting at "ClientRaw" or "ImposterRaw",
        # so it works regardless of where the dataset was originally stored.
        parts = line.replace('\\', '/').split('/')
        if 'ClientRaw' in parts:
            idx = parts.index('ClientRaw')
        elif 'ImposterRaw' in parts:
            idx = parts.index('ImposterRaw')
        else:
            raise ValueError(f"Unexpected path format in protocol file: {line}")

        relative_path = '/'.join(parts[idx:])
        entries.append((subject_id, relative_path))

    return entries


def load_protocol():
    """
    Loads the four official NUAA protocol files and returns a dict:
        {
            'train': [(absolute_path, label, subject_id), ...],
            'test':  [(absolute_path, label, subject_id), ...],
        }
    Labels follow config.LABEL_LIVE / config.LABEL_SPOOF.
    """
    client_train = _read_protocol_file(CLIENT_TRAIN_LIST)
    client_test = _read_protocol_file(CLIENT_TEST_LIST)
    imposter_train = _read_protocol_file(IMPOSTER_TRAIN_LIST)
    imposter_test = _read_protocol_file(IMPOSTER_TEST_LIST)

    def to_entries(pairs, label):
        return [
            (os.path.join(NUAA_DIR, rel_path), label, subject_id)
            for subject_id, rel_path in pairs
        ]

    train = to_entries(client_train, LABEL_LIVE) + to_entries(imposter_train, LABEL_SPOOF)
    test = to_entries(client_test, LABEL_LIVE) + to_entries(imposter_test, LABEL_SPOOF)

    return {'train': train, 'test': test}


def split_train_validation(train_entries, validation_subjects=None):
    """
    Splits the official training entries into a smaller training set
    and a validation set, with no subject overlap between the two.

    validation_subjects: list of subject_id strings to hold out for
    validation. Defaults to config.VALIDATION_SUBJECTS.
    """
    if validation_subjects is None:
        validation_subjects = VALIDATION_SUBJECTS

    validation_subjects = set(validation_subjects)

    train_final = [e for e in train_entries if e[2] not in validation_subjects]
    validation = [e for e in train_entries if e[2] in validation_subjects]

    return train_final, validation


def build_mtcnn():
    """Returns an MTCNN instance configured with the project's fixed settings."""
    return MTCNN(image_size=MTCNN_IMAGE_SIZE, margin=MTCNN_MARGIN)


def align_face(mtcnn, image_path):
    """
    Runs MTCNN on a single image.
    Returns a torch.Tensor of shape (3, MTCNN_IMAGE_SIZE, MTCNN_IMAGE_SIZE),
    or None if no face was detected.
    """
    img = Image.open(image_path).convert('RGB')
    return mtcnn(img)