"""
Central project configuration.
Any constant used in more than one place should live here,
to avoid divergent values across notebooks/experiments.
"""

import os

# ---------------------------------------------------------------------------
# Paths (Google Drive, mounted in Colab)
# ---------------------------------------------------------------------------
DRIVE_ROOT = '/content/drive/MyDrive/TCC'

RAW_DIR = os.path.join(DRIVE_ROOT, 'Datasets', 'NUAA', 'raw')
NUAA_DIR = RAW_DIR

CLIENT_RAW_DIR = os.path.join(NUAA_DIR, 'ClientRaw')
IMPOSTER_RAW_DIR = os.path.join(NUAA_DIR, 'ImposterRaw')

# NUAA official protocol files (shipped with the dataset)
CLIENT_TRAIN_LIST = os.path.join(NUAA_DIR, 'client_train_raw.txt')
CLIENT_TEST_LIST = os.path.join(NUAA_DIR, 'client_test_raw.txt')
IMPOSTER_TRAIN_LIST = os.path.join(NUAA_DIR, 'imposter_train_raw.txt')
IMPOSTER_TEST_LIST = os.path.join(NUAA_DIR, 'imposter_test_raw.txt')

PROCESSED_DIR = os.path.join(DRIVE_ROOT, 'data', 'processed')

TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'validation')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

MODELS_DIR = os.path.join(DRIVE_ROOT, 'models')
REPORTS_DIR = os.path.join(DRIVE_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

EXPERIMENT_LOG_PATH = os.path.join(DRIVE_ROOT, 'experiment_log.csv')

# ---------------------------------------------------------------------------
# Preprocessing (MTCNN)
# ---------------------------------------------------------------------------
MTCNN_IMAGE_SIZE = 160
MTCNN_MARGIN = 40  # decision recorded in docs/decisions.md

# ---------------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------------
# Subjects held out from the official training set to form the
# validation set. Chosen with no overlap against the remaining
# training subjects.
VALIDATION_SUBJECTS = ['0008', '0009']

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_LIVE = 0
LABEL_SPOOF = 1