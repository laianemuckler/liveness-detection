"""
ViT v2 (ViT-B/16): modelo, treino e verificação de recursos, seguindo George & Marcel.

Arquivo novo; não altera src/modeling/cnn.py (v1) nem src/modeling/cnn_v2.py. As funções genéricas
(seed, dados, dataloaders, contagem de parâmetros, avaliação com tempo, pré-processamento de uma
imagem) continuam vindo de src.modeling.cnn e src.modeling.cnn_v2; aqui entra só o que depende do ViT.

--- O que vem do artigo (George & Marcel, "On the Effectiveness of Vision Transformers for Zero-shot
    Face Anti-Spoofing", arXiv 2011.08019) ---
- ViT "base", patch 16x16, imagem 224x224, implementação do timm (Wightman), com pesos pré-treinados.
- A cabeça original é removida e entra uma camada totalmente conectada (FC) de 768 -> 1 saída
  (sigmoide na saída), sem camada oculta e sem dropout.
- Só a FC treina; o resto do ViT fica congelado (nível "FC" do artigo).
- Adam, learning rate fixo de 1e-4, weight decay de 1e-5, perda BCE, 20 épocas, batch 16
  (o artigo diz que foi por limite de memória), flip horizontal com p = 0,5, e o melhor modelo é o de
  menor loss de validação.

--- Decisões do PROJETO (o artigo não informa ou o projeto muda) ---
- Pesos: o artigo não informa qual checkpoint do timm foi usado ("grandes quantidades de dados").
  Aqui o checkpoint é escolhido explicitamente em `weights_tag`. O padrão do notebook é
  `orig_in21k_ft_in1k`: pesos originais do ViT (código oficial do Google), pré-treinados no
  ImageNet-21k e ajustados no ImageNet-1k, segundo os comentários do timm 1.0.30.
- Normalização: a do checkpoint, lida da configuração dos pesos (nos pesos do timm usados aqui é
  média 0,5 e desvio 0,5, não a do ImageNet clássico que o CNN usa). Ver `get_norm`.
- Precisão mista (AMP) ligada, como no CNN; o artigo não informa.
- `patience` opcional: com `patience=None` (padrão) treina todas as épocas e só restaura o melhor
  modelo, como no artigo; com um número, para por early stopping como no CNN.
- Augmentation adicional (rotação, brilho) existe em `get_transforms_vit`, mas fica desligada por padrão.
- Medição de tempo: ver `evaluate_cnn_timed` em src.modeling.cnn_v2.

O ViT não tem BatchNorm (usa LayerNorm, sem estatísticas móveis), então não há o tratamento de BN
congelada do CNN. Os dropouts do timm ViT são 0 por padrão.

Rótulos como em src.config: LABEL_LIVE = 0, LABEL_SPOOF = 1.
"""

import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm.auto import tqdm

from src.modeling.cnn import evaluate_cnn
from src.modeling.cnn_v2 import evaluate_cnn_timed, preprocess_one  # noqa: F401  (reexportados para o notebook)

LEVELS = ('vit_b16_v2_01_fc',)

# Nome dos parâmetros da cabeça no timm (VisionTransformer.head).
HEAD_PREFIX = 'head.'


def _sync(device):
    """Espera a GPU terminar o que foi enviado (necessário para medir tempo de GPU)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_vit_v2(level, model_name='vit_base_patch16_224', weights_tag='orig_in21k_ft_in1k',
                 pretrained=True):
    """
    ViT do timm com uma cabeça nova de 1 saída (FC 768 -> 1) e o resto congelado.

      vit_b16_v2_01_fc : só a FC treina (George & Marcel)

    `weights_tag` escolhe o checkpoint do timm (ex.: 'orig_in21k_ft_in1k').
    Com pretrained=False os pesos não são baixados (serve para contar parâmetros e ler a normalização).
    """
    if level not in LEVELS:
        raise ValueError(f"Nível desconhecido '{level}'. Use um de {LEVELS}")

    import timm  # importado aqui para dar um erro claro se o timm não estiver instalado

    model = timm.create_model(f"{model_name}.{weights_tag}", pretrained=pretrained, num_classes=1)

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(HEAD_PREFIX)

    return model


def get_norm(model):
    """Média, desvio e tamanho de entrada dos pesos do modelo (da configuração do timm)."""
    from timm.data import resolve_data_config

    cfg = resolve_data_config(model=model)
    return tuple(cfg['mean']), tuple(cfg['std']), tuple(cfg['input_size'])


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def get_transforms_vit(mean, std, flip_p=0.5, rotation_degrees=0, brightness=0.0):
    """
    Devolve (train_transform, eval_transform).
    Por padrão o treino só tem flip horizontal com p = 0,5 (George & Marcel). Rotação e brilho
    (os do CNN) só entram se forem passados valores maiores que 0.
    A avaliação usa só a normalização do checkpoint.
    """
    normalize = transforms.Normalize(mean, std)

    train_steps = [transforms.RandomHorizontalFlip(flip_p)]
    if rotation_degrees:
        train_steps.append(transforms.RandomRotation(rotation_degrees))
    if brightness:
        train_steps.append(transforms.ColorJitter(brightness=brightness))
    train_steps += [transforms.ToTensor(), normalize]

    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.Compose(train_steps), eval_tf


# ---------------------------------------------------------------------------
# Checkpoints (só os parâmetros que treinam; o resto vem dos pesos pré-treinados)
# ---------------------------------------------------------------------------
def trainable_state(model):
    """Cópia (em CPU) dos parâmetros com requires_grad. No nível FC são só os da cabeça."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


def load_trainable_state(model, path):
    """
    Carrega um checkpoint salvo por train_vit_v2 num modelo recém-criado com os mesmos pesos
    pré-treinados (pretrained=True). Confere que as chaves são exatamente as dos parâmetros treináveis.
    """
    state = torch.load(path, map_location='cpu')
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != trainable:
        raise ValueError(f"As chaves de {path} não batem com os parâmetros treináveis do modelo: "
                         f"só no arquivo {sorted(set(state) - trainable)[:5]}, "
                         f"só no modelo {sorted(trainable - set(state))[:5]}.")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Chaves inesperadas ao carregar {path}: {result.unexpected_keys[:5]}")
    return model


# ---------------------------------------------------------------------------
# Treino (Adam + BCE; o melhor modelo é o de menor loss de validação)
# ---------------------------------------------------------------------------
def train_vit_v2(model, train_loader, val_loader, y_val, device,
                 lr_head=1e-4, lr_backbone=None, weight_decay=1e-5,
                 max_epochs=20, patience=None, checkpoint_path=None):
    """
    Treina todas as épocas (max_epochs) e, no fim, restaura o modelo de menor loss de validação
    (George & Marcel). Se `patience` for um número, para antes por early stopping (como no CNN).

    A cabeça usa lr_head. Se houver parâmetros do backbone descongelados (outros níveis), eles usam
    lr_backbone, que então é obrigatório. O checkpoint guarda só os parâmetros que treinam.

    Devolve (model, history DataFrame, info), onde info tem:
      train_s    -> tempo total do treino em segundos (inclui a validação a cada época;
                    não inclui restaurar os pesos nem salvar o checkpoint)
      epochs     -> épocas executadas
      best_epoch -> época com menor loss de validação
    """
    model.to(device)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(HEAD_PREFIX)]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith(HEAD_PREFIX)]
    if not head_params:
        raise ValueError("Nenhum parâmetro da cabeça está treinável.")

    param_groups = [{'params': head_params, 'lr': lr_head}]
    if backbone_params:
        if lr_backbone is None:
            raise ValueError("Há parâmetros do backbone treináveis: informe lr_backbone.")
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except AttributeError:  # PyTorch antigo
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float('inf')
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    history = []

    _sync(device)
    run_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoca {epoch:02d}", leave=False, unit='batch'):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * y.numel()
            n += y.numel()
        train_loss = running_loss / n
        _sync(device)
        train_time = time.perf_counter() - epoch_start

        val_res = evaluate_cnn(model, val_loader, y_val, device)
        epoch_time = time.perf_counter() - epoch_start
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_res['loss'],
            'val_HTER': val_res['HTER'],
            'val_AUC': val_res['AUC'],
            'train_time_s': train_time,
            'epoch_time_s': epoch_time,
        })
        print(f"epoca {epoch:02d} | loss treino {train_loss:.4f} | "
              f"loss val {val_res['loss']:.4f} | HTER val {val_res['HTER']*100:.2f}% | "
              f"treino {train_time:.1f}s | epoca {epoch_time:.1f}s")

        if val_res['loss'] < best_loss:
            best_loss = val_res['loss']
            best_epoch = epoch
            best_state = trainable_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(f"Early stopping na época {epoch} (melhor loss de validação {best_loss:.4f}, época {best_epoch})")
                break

    _sync(device)
    train_s = time.perf_counter() - run_start
    print(f"Treino concluído: {len(history)} épocas em {train_s/60:.1f} min "
          f"({train_s/len(history):.1f}s por época em média)")

    if best_state is None:
        raise RuntimeError("Nenhuma época teve loss de validação válida (NaN?). Treino interrompido.")
    model.load_state_dict(best_state, strict=False)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)

    info = {'train_s': train_s, 'epochs': len(history), 'best_epoch': best_epoch}
    return model, pd.DataFrame(history), info


# ---------------------------------------------------------------------------
# Verificação de memória e tempo antes de treinar
# ---------------------------------------------------------------------------
def resource_check(model, train_loader, device, lr_head=1e-4, weight_decay=1e-5, n_steps=6):
    """
    Roda poucos passos de treino (sem validação e sem salvar nada) e mede o pico de memória da GPU e o
    tempo por passo. O primeiro passo é de aquecimento e não entra na média.
    `est_epoch_s` é uma ESTIMATIVA do tempo de treino de uma época (passos × tempo por passo), sem a
    validação. Se faltar memória, devolve oom=True em vez de dar erro.
    Muda os pesos do modelo recebido: use um modelo só para isso.
    """
    if device.type != 'cuda':
        raise RuntimeError("A verificação precisa de GPU.")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr_head, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    try:
        for step, (x, y) in enumerate(train_loader):
            if step >= n_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _sync(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=True):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            times.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        return {'oom': True, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1024 ** 3}

    steps = times[1:] if len(times) > 1 else times
    s_per_step = sum(steps) / len(steps)
    return {
        'oom': False,
        'peak_mem_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        's_per_step': s_per_step,
        'est_epoch_s': s_per_step * len(train_loader),
    }
