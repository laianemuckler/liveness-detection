"""
Extração de características LBP, versão 2 (usada com o pré-processamento v2).

Este arquivo NÃO substitui src/features.py (v1): é um módulo novo, com nomes
de configuração novos, para que nada do v1 seja alterado.

Base: Maatta, Hadid & Pietikainen (2011), "Face spoofing detection from single
images using micro-texture analysis" (IJCB) e Chingovska, Anjos & Marcel (2012),
"On the Effectiveness of Local Binary Patterns in Face Anti-spoofing" (BIOSIG).

--- O que vem dos artigos ---
- LBP uniforme "u2": um rótulo para cada padrão uniforme e um único rótulo para
  todos os não uniformes. Dá 59 bins para P=8 e 243 bins para P=16
  (no scikit-image: method='nri_uniform'; o method='uniform' do features.py
  v1 é outra variante, com P+2 bins).
- Rosto normalizado para 64x64 antes do LBP (os dois artigos).
- Global (Chingovska, per-image): LBP(8,1), 59 dimensões.
- Grid (Chingovska, per-block): imagem dividida em 3x3 blocos, um histograma por
  bloco, concatenados: 9 x 59 = 531 dimensões.
- Maatta (Maatta et al., Seção 3.1 e Fig. 4): LBP(8,1) aplicado na imagem inteira,
  resultado dividido em 3x3 regiões sobrepostas (sobreposição de 14 pixels, em
  imagem 64x64), 9 x 59 = 531 bins; mais dois histogramas globais do rosto
  inteiro, LBP(8,2) com 59 bins e LBP(16,2) com 243 bins.
  Total: 531 + 59 + 243 = 833.

--- Decisões do PROJETO (os artigos não informam) ---
- O recorte de 224 (v2) é convertido para cinza e reduzido para 64x64 apenas
  dentro desta extração; os arquivos em disco não mudam. O método de redução
  (bilinear) é escolha nossa.
- Como a sobreposição é de 14 px em 64 px e as 3 regiões cobrem a imagem toda,
  cada região tem (64 + 2*14) / 3 = 30,67 px; os limites são arredondados:
  [0:31], [17:47], [33:64] (tamanhos 31, 30 e 31).
- Grid: o LBP é calculado dentro de cada bloco (como no diagrama da Fig. 2 do
  Chingovska e como no v1); os blocos cobrem os 64 px: [0:21], [21:43], [43:64].
  O artigo não diz se os blocos do per-block são sobrepostos; aqui não são.
- Cada histograma é normalizado individualmente (soma 1). Os artigos não dizem.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from tqdm import tqdm

LBP_INPUT_SIZE = 64      # tamanho do rosto nos dois artigos
OVERLAP_PX = 14          # sobreposição das regiões do Maatta, em imagem 64x64
_RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR   # decisão do projeto


def _load_gray_64(image_path):
    """Carrega o recorte, converte para cinza e reduz para 64x64."""
    img = Image.open(image_path).convert('L')
    img = img.resize((LBP_INPUT_SIZE, LBP_INPUT_SIZE), _RESAMPLE)
    return np.array(img)


def _n_bins(P):
    """Rótulos do LBP uniforme u2: P*(P-1)+3 (59 para P=8, 243 para P=16)."""
    return P * (P - 1) + 3


def _lbp_codes(gray, P, R):
    return local_binary_pattern(gray, P, R, method='nri_uniform').astype(np.int64)


def _hist(codes, P):
    """Histograma normalizado dos códigos LBP u2 (soma 1)."""
    h = np.bincount(codes.ravel(), minlength=_n_bins(P))
    if h.size != _n_bins(P):
        raise ValueError(
            f"LBP(P={P}) gerou rótulos além do esperado: {h.size} bins em vez de {_n_bins(P)}."
        )
    h = h.astype(float)
    return h / (h.sum() + 1e-8)


def _grid_bounds(size, n=3):
    """Limites de n blocos NÃO sobrepostos que cobrem toda a imagem."""
    edges = np.linspace(0, size, n + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


def _overlapping_bounds(size, n=3, overlap=OVERLAP_PX):
    """Limites de n regiões sobrepostas (overlap px entre vizinhas) que cobrem toda a imagem."""
    region = (size + (n - 1) * overlap) / n
    step = region - overlap
    return [(int(round(i * step)), int(round(i * step + region))) for i in range(n)]


def lbp_u2_global(image_path):
    """Config. 1 (Chingovska, per-image): LBP(8,1) u2 na imagem inteira. 59 dimensões."""
    gray = _load_gray_64(image_path)
    return _hist(_lbp_codes(gray, 8, 1), 8)


def lbp_u2_grid(image_path, grid_size=3):
    """Config. 2 (Chingovska, per-block): 3x3 blocos, LBP(8,1) u2 em cada bloco. 9 x 59 = 531."""
    gray = _load_gray_64(image_path)
    bounds = _grid_bounds(gray.shape[0], grid_size)
    hists = []
    for y0, y1 in bounds:
        for x0, x1 in bounds:
            hists.append(_hist(_lbp_codes(gray[y0:y1, x0:x1], 8, 1), 8))
    return np.concatenate(hists)


def lbp_u2_maatta(image_path):
    """Config. 3 (Maatta et al., 2011): 531 (9 regiões sobrepostas, LBP(8,1)) + 59 (8,2 global)
    + 243 (16,2 global) = 833 dimensões."""
    gray = _load_gray_64(image_path)
    size = gray.shape[0]

    codes_81 = _lbp_codes(gray, 8, 1)          # LBP na imagem inteira, depois divide
    bounds = _overlapping_bounds(size)
    regions = [_hist(codes_81[y0:y1, x0:x1], 8) for y0, y1 in bounds for x0, x1 in bounds]

    global_82 = _hist(_lbp_codes(gray, 8, 2), 8)
    global_162 = _hist(_lbp_codes(gray, 16, 2), 16)

    return np.concatenate(regions + [global_82, global_162])


FEATURE_CONFIGS_V2 = {
    'lbp_u2_global_p8r1': lbp_u2_global,
    'lbp_u2_grid_p8r1_3x3': lbp_u2_grid,
    'lbp_u2_maatta': lbp_u2_maatta,
}

EXPECTED_DIMS_V2 = {
    'lbp_u2_global_p8r1': 59,
    'lbp_u2_grid_p8r1_3x3': 9 * 59,            # 531
    'lbp_u2_maatta': 9 * 59 + 59 + 243,        # 833
}


def extract_features_v2(image_paths, config_name, n_workers=None):
    """Extrai as características de uma lista de imagens, em paralelo (a ordem é preservada).
    Nenhum arquivo é gravado aqui."""
    if config_name not in FEATURE_CONFIGS_V2:
        raise ValueError(
            f"Configuração desconhecida '{config_name}'. Disponíveis: {list(FEATURE_CONFIGS_V2)}"
        )
    extractor = FEATURE_CONFIGS_V2[config_name]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        features = list(tqdm(executor.map(extractor, image_paths),
                             total=len(image_paths), desc=config_name))
    return np.array(features)
