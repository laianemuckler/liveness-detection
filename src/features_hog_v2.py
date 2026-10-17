
"""
--- Decisões do PROJETO (o artigo não informa ou o código não permite) ---
- Tamanho da entrada: 224x224, o recorte do pré-processamento v2, sem redimensionar.
  O artigo usa janela de 64x128 (testa também 56x120 e 48x112, só reduzindo a margem em volta
  da pessoa, 6.5) e não trata de rostos nem de imagens quadradas.
  Vetor: 28x28 células -> 27x27 blocos x 36 = 26.244 dimensões.
- O scikit-image não tem a janela gaussiana de sigma=8 do artigo (a própria documentação
  lista isso como diferença em relação ao artigo). Aqui ela não é aplicada.
- epsilon da normalização: 1e-5, o valor do scikit-image. O artigo não dá o valor e diz que o
  resultado é insensível a ele numa faixa grande (6.4).
- Os pixels da borda da imagem ficam com gradiente zero no scikit-image; o artigo não trata disso.
- O scikit-image NÃO interpola os votos entre bins de orientação nem entre células, e o artigo
  interpola nos dois (6.3). Conferido com imagens sintéticas: um gradiente de 19 graus vai todo
  para o bin 0 e um de 21 graus todo para o bin 1; e o gradiente de uma célula não gera voto
  nas células vizinhas. Aqui os votos são contados sem interpolação.
- As características são guardadas em float32 (metade da memória; 26.244 dimensões x 9.123
  imagens de teste em float64 ocupariam cerca de 1,9 GB).
- Os parâmetros são passados explicitamente. O padrão do scikit-image 0.26.0 para blocos é 3x3
  (o artigo usa 2x2); a normalização padrão nessa versão já é L2-Hys, embora a documentação
  ainda cite L1 como diferença em relação ao artigo.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from skimage.feature import hog
from tqdm import tqdm

HOG_INPUT_SIZE = 224     # decisão do projeto: recorte v2, sem redimensionar


def _load_rgb(image_path):
    """Carrega o recorte v2 em RGB (3 canais, 224x224)."""
    img = Image.open(image_path).convert('RGB')
    if img.size != (HOG_INPUT_SIZE, HOG_INPUT_SIZE):
        raise ValueError(f"{image_path} tem tamanho {img.size}; o esperado no v2 é "
                         f"({HOG_INPUT_SIZE}, {HOG_INPUT_SIZE}).")
    return np.array(img)


def hog_paper_rgb(image_path):
    """HOG do detector padrão de Dalal & Triggs (2005), sem a janela gaussiana.
    RGB, 9 bins sem sinal, células 8x8, blocos 2x2, L2-Hys. 26.244 dimensões em 224x224."""
    rgb = _load_rgb(image_path)
    features = hog(
        rgb,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=False,      # sem correção gama (6.1)
        feature_vector=True,
        channel_axis=-1,           # RGB: gradiente de maior norma entre os canais (6.2)
    )
    return features.astype(np.float32)


FEATURE_CONFIGS_HOG_V2 = {
    'hog_paper_rgb_224': hog_paper_rgb,
}

EXPECTED_DIMS_HOG_V2 = {
    'hog_paper_rgb_224': 27 * 27 * 2 * 2 * 9,      # 26244
}


def extract_features_hog_v2(image_paths, config_name, n_workers=None):
    """Extrai as características de uma lista de imagens, em paralelo (a ordem é preservada).
    Nenhum arquivo é gravado aqui."""
    if config_name not in FEATURE_CONFIGS_HOG_V2:
        raise ValueError(
            f"Configuração desconhecida '{config_name}'. Disponíveis: {list(FEATURE_CONFIGS_HOG_V2)}"
        )
    extractor = FEATURE_CONFIGS_HOG_V2[config_name]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features = list(tqdm(executor.map(extractor, image_paths),
                             total=len(image_paths), desc=config_name))
    return np.array(features, dtype=np.float32)
