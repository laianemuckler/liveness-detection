"""
Extração de características HOG, versão 2 (usada com o pré-processamento v2).

Este arquivo é só do HOG. Não substitui src/features.py (v1) e não mexe em
src/features_v2.py (LBP).

Base: Dalal & Triggs (2005), "Histograms of Oriented Gradients for Human Detection"
(CVPR). O artigo estuda detecção de pedestres, em imagens de pessoas na rua com janela
de 64x128. Aqui o domínio é outro (foto de rosto real e foto de foto, NUAA).

--- O que vem do artigo (detector padrão, Seção 6) ---
- RGB, sem correção gama (6.1: o artigo usa cor quando disponível; cinza perde 1,5%
  a 1e-4 FPPW).
- Gradiente [-1, 0, 1] sem suavização (6.2). Em imagem colorida, o gradiente é calculado
  por canal e vale o de maior norma em cada pixel (6.2).
- 9 bins de orientação entre 0 e 180 graus (sem sinal) (6.3). Voto = magnitude do gradiente.
- Células de 8x8 pixels; blocos de 2x2 células (16x16 pixels) com passo de 8 pixels (Seção 6).
- Normalização de bloco L2-Hys: norma L2, corte em 0,2 e nova normalização (6.4).
- Classificador: SVM linear com C=0,01 (6.6). Isso fica no notebook, não aqui.

--- Decisões do PROJETO (o artigo não informa ou o código não permite) ---
- Tamanho da entrada: 224x224, o recorte do pré-processamento v2, sem redimensionar.
  O artigo usa 64x128 e não diz nada sobre outros tamanhos nem sobre rostos.
  Vetor: 28x28 células -> 27x27 blocos x 36 = 26.244 dimensões.
- O scikit-image não tem a janela gaussiana de sigma=8 do artigo (a própria documentação
  lista isso como diferença em relação ao artigo). Aqui ela não é aplicada.
- epsilon da normalização: 1e-5, o valor do scikit-image. O artigo não dá o valor e diz que o
  resultado é insensível a ele numa faixa grande (6.4).
- Os pixels da borda da imagem ficam com gradiente zero no scikit-image; o artigo não trata disso.
- Se o scikit-image interpola os votos entre bins e posições como o artigo (6.3): NÃO conferido.
- As características são guardadas em float32 (metade da memória; 26.244 dimensões x 9.123
  imagens de teste em float64 passaria de 1,9 GB).
- Os parâmetros são passados explicitamente, porque os padrões do scikit-image são outros
  (blocos 3x3 e normalização L1).
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
