"""
Medição de tempo e registro de hardware, para ser usada por TODOS os notebooks
(LBP, HOG, CNN, ViT). Arquivo novo; não altera nenhum módulo existente.

Ideia: cada notebook mede o que é do seu método e grava uma linha no mesmo formato,
para a comparação ficar na mesma tabela.

Colunas (uma linha por experimento e etapa):
  exp_id, etapa ('validacao' ou 'teste'), n_imagens,
  extracao_ms_por_img   -> tudo antes do classificador, por imagem
                           (LBP/HOG: ler + reduzir + extrair característica;
                            CNN/ViT: ler + pré-processar)
  treino_s              -> tempo total de treino do classificador/rede
  predicao_ms_por_img   -> só o classificador/rede, por imagem
  inferencia_total_ms_por_img -> extracao + predicao
  hardware, data, obs

Embasamento: ver o registro de decisões. Os artigos Green AI (Schwartz et al.) e
The Efficiency Misnomer (Dehghani et al.) dizem que o tempo depende do hardware e que se deve
descrever o ambiente e separar treino de inferência. Quais colunas gravar é decisão do projeto.
"""
import csv
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

TIMING_COLS = ['exp_id', 'etapa', 'n_imagens', 'extracao_ms_por_img', 'treino_s',
               'predicao_ms_por_img', 'inferencia_total_ms_por_img', 'hardware', 'data', 'obs']


def hardware_info(usado='CPU'):
    """Descrição do ambiente: CPU, núcleos, GPU disponível e o que foi realmente usado."""
    import platform
    cpu = platform.processor() or 'CPU desconhecida'
    try:
        with open('/proc/cpuinfo') as f:
            for linha in f:
                if linha.startswith('model name'):
                    cpu = linha.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = 'nenhuma'
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return f"CPU: {cpu} ({os.cpu_count()} núcleos); GPU disponível: {gpu}; usado: {usado}"


def medir_s(fn):
    """Executa fn() e devolve (resultado, segundos)."""
    t0 = time.perf_counter()
    resultado = fn()
    return resultado, time.perf_counter() - t0


def medir_por_item(fn, itens, n_aquecimento=5):
    """Chama fn(item) em sequência (1 processo) e devolve ms por item.
    Os primeiros n_aquecimento itens não entram na medição."""
    itens = list(itens)
    for it in itens[:n_aquecimento]:
        fn(it)
    t0 = time.perf_counter()
    for it in itens:
        fn(it)
    return (time.perf_counter() - t0) / len(itens) * 1000


def copiar_amostra(paths, n, pasta_local, n_threads=16):
    """Copia n arquivos igualmente espaçados da lista para o disco local e devolve os novos caminhos.
    Serve para medir o tempo sem incluir a leitura lenta do Drive. Não altera os originais."""
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    os.makedirs(pasta_local, exist_ok=True)

    def _copia(i):
        destino = os.path.join(pasta_local, f"{i}_{os.path.basename(paths[i])}")
        if not os.path.exists(destino):
            shutil.copyfile(paths[i], destino)
        return destino

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(_copia, idx))


def carregar_tempos(csv_path):
    if not os.path.isfile(csv_path):
        return pd.DataFrame(columns=TIMING_COLS)
    return pd.read_csv(csv_path, dtype=str)


def tem_tempo(csv_path, exp_id, etapa):
    t = carregar_tempos(csv_path)
    return bool(((t['exp_id'] == exp_id) & (t['etapa'] == etapa)).any())


def gravar_tempo(csv_path, exp_id, etapa, n_imagens, extracao_ms=None, treino_s=None,
                 predicao_ms=None, hardware=None, obs=''):
    """Acrescenta uma linha. Nunca sobrescreve: se (exp_id, etapa) já existe, não grava e avisa."""
    if tem_tempo(csv_path, exp_id, etapa):
        print(f"[{exp_id}] tempo de '{etapa}' já registrado; não sobrescrevi.")
        return False
    total = None
    if extracao_ms is not None and predicao_ms is not None:
        total = extracao_ms + predicao_ms

    def fmt(v, casas):
        return '' if v is None else f"{float(v):.{casas}f}"

    linha = {
        'exp_id': exp_id, 'etapa': etapa, 'n_imagens': n_imagens,
        'extracao_ms_por_img': fmt(extracao_ms, 3), 'treino_s': fmt(treino_s, 2),
        'predicao_ms_por_img': fmt(predicao_ms, 4), 'inferencia_total_ms_por_img': fmt(total, 3),
        'hardware': hardware or hardware_info(), 'data': date.today().isoformat(), 'obs': obs,
    }
    existe = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TIMING_COLS)
        if not existe:
            w.writeheader()
        w.writerow(linha)
    return True


def gravar_tempo_teste(csv_path, exp_id, n_teste, predicao_total_s, hardware=None):
    """Linha do teste: a predição é medida no teste; extração e treino vêm da linha de validação
    do mesmo experimento (a extração é medida numa amostra da validação, não nas imagens do teste)."""
    t = carregar_tempos(csv_path)
    val = t[(t['exp_id'] == exp_id) & (t['etapa'] == 'validacao')]
    if val.empty:
        print(f"[{exp_id}] sem tempo de validação; meça a validação antes para ter extração e treino.")
        return False
    v = val.iloc[0]
    extracao = float(v['extracao_ms_por_img']) if v['extracao_ms_por_img'] not in ('', None) else None
    treino = float(v['treino_s']) if v['treino_s'] not in ('', None) else None
    ok = gravar_tempo(csv_path, exp_id, 'teste', n_teste, extracao_ms=extracao, treino_s=treino,
                      predicao_ms=predicao_total_s / n_teste * 1000, hardware=hardware,
                      obs='extração e treino copiados da linha de validação')
    if ok:
        print(f"[{exp_id}] teste: predição {predicao_total_s / n_teste * 1000:.3f} ms/img")
    return ok
