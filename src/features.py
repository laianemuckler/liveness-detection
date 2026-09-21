"""
Feature extraction functions: LBP and HOG.

--- LBP ---
Three configurations, following Chingovska et al. (2012) and the
comparison method they reimplement from Maatta et al. [7]:

1. lbp_global      -> LBP(P=8, R=1), single histogram over the whole face
2. lbp_grid        -> LBP(P=8, R=1), image split into a grid of blocks,
                       histograms concatenated (per-block features)
3. lbp_maatta      -> concatenation of:
                       - LBP(P=16, R=2) over the whole face
                       - LBP(P=8, R=1) over 9 overlapping blocks
                       - LBP(P=8, R=1) over the whole face
                       (replicates the method compared against in
                       Chingovska et al., originally proposed by
                       Maatta, Hadid and Pietikainen, 2011)

--- HOG ---
Parameters follow Dalal & Triggs' (2005) original "default detector":
9 orientation bins (0-180 deg), 8x8 pixel cells, 2x2 cell blocks,
L2-Hys normalization. These match skimage.feature.hog's own defaults.
The original 64x128 detection window (designed for upright pedestrians)
does not apply here; the square 160x160 aligned face is used as-is.
"""

import numpy as np
from skimage.feature import local_binary_pattern, hog
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _to_grayscale(image_path):
    """Loads an image and converts it to grayscale as a numpy array."""
    img = Image.open(image_path).convert('L')
    return np.array(img)


def _lbp_histogram(gray_patch, P, R):
    """
    Computes a single normalized uniform-LBP histogram for one
    grayscale patch (can be the whole image or a block/region of it).
    """
    lbp = local_binary_pattern(gray_patch, P, R, method='uniform')
    n_bins = P + 2  # number of uniform patterns + 1 catch-all bin
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-8)  # avoid division by zero
    return hist


def _split_into_grid(gray_image, grid_size):
    """
    Splits a grayscale image into grid_size x grid_size non-overlapping
    blocks. Returns a list of 2D numpy arrays (one per block).
    """
    h, w = gray_image.shape
    block_h = h // grid_size
    block_w = w // grid_size

    blocks = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = gray_image[i * block_h:(i + 1) * block_h,
                                j * block_w:(j + 1) * block_w]
            blocks.append(block)
    return blocks


def _split_into_overlapping_blocks(gray_image, grid_size=3, overlap=0.5):
    """
    Splits a grayscale image into grid_size x grid_size OVERLAPPING
    blocks. overlap is the fraction of each block that overlaps with
    its neighbour (0.5 = 50% overlap).

    Note: the original Maatta et al. paper does not specify the exact
    overlap ratio used for the 9 overlapping blocks; 50% is a
    reasonable, commonly used default and is documented here as an
    assumption.
    """
    h, w = gray_image.shape
    step_h = int((h / grid_size) * (1 - overlap))
    step_w = int((w / grid_size) * (1 - overlap))
    block_h = h // grid_size + step_h
    block_w = w // grid_size + step_w

    blocks = []
    for i in range(grid_size):
        for j in range(grid_size):
            y0 = min(i * step_h, h - block_h) if h > block_h else 0
            x0 = min(j * step_w, w - block_w) if w > block_w else 0
            block = gray_image[y0:y0 + block_h, x0:x0 + block_w]
            blocks.append(block)
    return blocks


def lbp_global(image_path, P=8, R=1):
    """
    Configuration 1: single LBP histogram over the whole face.
    Equivalent to Chingovska et al.'s "per-image" LBP_{3x3} (P=8, R=1).
    Output size: P + 2 (10 for P=8).
    """
    gray = _to_grayscale(image_path)
    return _lbp_histogram(gray, P, R)


def lbp_grid(image_path, P=8, R=1, grid_size=3):
    """
    Configuration 2: image split into grid_size x grid_size
    non-overlapping blocks, one LBP histogram per block, concatenated.
    Equivalent to Chingovska et al.'s "per-block" LBP_{3x3} (P=8, R=1).
    Output size: grid_size^2 * (P + 2) (531 for P=8, grid_size=3, per Chingovska).
    """
    gray = _to_grayscale(image_path)
    blocks = _split_into_grid(gray, grid_size)
    histograms = [_lbp_histogram(block, P, R) for block in blocks]
    return np.concatenate(histograms)


def lbp_maatta(image_path):
    """
    Configuration 3: replicates the method by Maatta, Hadid and
    Pietikainen (2011), as reimplemented for comparison in
    Chingovska et al. (2012). Concatenates:
      - LBP(P=16, R=2) over the whole face
      - LBP(P=8, R=1) over 9 overlapping blocks
      - LBP(P=8, R=1) over the whole face
    Output size: 18 + 9*10 + 10 = 118... (see note below)

    Note: the original paper reports 833 dimensions; the discrepancy
    likely comes from unspecified exact block sizes/overlap and
    possibly non-uniform LBP variants for some components. This
    implementation follows the general structure described in the
    paper as closely as possible from the published description.
    """
    gray = _to_grayscale(image_path)

    hist_16_2_global = _lbp_histogram(gray, P=16, R=2)

    blocks = _split_into_overlapping_blocks(gray, grid_size=3, overlap=0.5)
    hist_8_1_blocks = np.concatenate(
        [_lbp_histogram(block, P=8, R=1) for block in blocks]
    )

    hist_8_1_global = _lbp_histogram(gray, P=8, R=1)

    return np.concatenate([hist_16_2_global, hist_8_1_blocks, hist_8_1_global])


def hog_features(image_path, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys'):
    """
    Extracts a HOG (Histogram of Oriented Gradients) feature vector
    for one image, using the "default detector" parameters from
    Dalal & Triggs (2005): 9 orientation bins, 8x8 pixel cells,
    2x2 cell blocks, L2-Hys normalization. These are also
    skimage.feature.hog's own defaults, so passing no arguments
    reproduces the original configuration.
    """
    gray = _to_grayscale(image_path)
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    )
    return features


def _lbp_global_p8r1(image_path):
    return lbp_global(image_path, P=8, R=1)


def _lbp_grid_p8r1_3x3(image_path):
    return lbp_grid(image_path, P=8, R=1, grid_size=3)


# Registry so experiment notebooks can select a configuration by name.
# Named functions (not lambdas) are required here so they can be sent
# to worker processes during parallel extraction.
FEATURE_CONFIGS = {
    # LBP
    'lbp_global_p8r1': _lbp_global_p8r1,
    'lbp_grid_p8r1_3x3': _lbp_grid_p8r1_3x3,
    'lbp_maatta_p16r2_p8r1': lbp_maatta,
    # HOG
    'hog_default': hog_features,
}


def extract_features(image_paths, config_name, n_workers=None):
    """
    Extracts features for a list of image paths using the named
    configuration (one of FEATURE_CONFIGS keys, LBP or HOG).

    Runs in parallel across n_workers processes (defaults to all
    available CPU cores). This is safe: each image is processed
    independently by its own worker, with no shared state between
    them, and results are only combined (np.array) here, in the
    main process, after every worker has finished. No file writes
    happen inside the workers, so there's no risk of race conditions
    or one worker overwriting another's output.

    ProcessPoolExecutor.map preserves input order, so the returned
    array lines up with image_paths (and therefore with the labels
    list built alongside it).
    """
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(
            f"Unknown feature config '{config_name}'. "
            f"Available: {list(FEATURE_CONFIGS.keys())}"
        )

    extractor = FEATURE_CONFIGS[config_name]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features = list(
            tqdm(
                executor.map(extractor, image_paths),
                total=len(image_paths),
                desc=config_name,
            )
        )

    return np.array(features)