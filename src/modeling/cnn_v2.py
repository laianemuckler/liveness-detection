"""
CNN v2 (ResNet-50): modelo com dois níveis de congelamento, treino com early stopping e
avaliação com medição de tempo.

Arquivo novo; não altera src/modeling/cnn.py (v1). As funções genéricas (seed, dados,
dataloaders, contagem de parâmetros, avaliação) continuam vindo de src.modeling.cnn.
O que muda aqui é o que depende da arquitetura (o v1 usa os nomes `features` e `classifier`
do DenseNet; o ResNet-50 do torchvision usa `layer1`..`layer4` e `fc`).

--- O que vem dos artigos ---
- Backbone ResNet-50 pré-treinado: Shinde et al. (2024) usam ResNet-50; Lee et al. comparam
  ResNet, EfficientNet, ViT e híbridos, e a ResNet foi a melhor CNN.
- Cabeça: o topo original é removido e entra uma camada densa de 512 unidades (Shinde et al.).
- Nível "head": backbone congelado, só a cabeça nova treina. O artigo diz "convoluções congeladas
  exceto as últimas 3 camadas" e dá 1.054.210 parâmetros treináveis. A conta do projeto (inferência,
  o artigo não a faz) fecha esse número com: densa 2048->512 (1.049.088) + saída de 2 unidades (1.026)
  + gamma e beta da última BatchNorm (4.096), supondo pooling global e Keras. Ou seja, na prática
  só a cabeça treina. Aqui a saída tem 1 logit, então a contagem do projeto é diferente.
- Otimizador Adam e learning rate 1e-4 para a cabeça (Shinde et al.).
- Nível "partial": princípio de Yosinski et al. (2014) (camadas iniciais gerais, finais específicas).

--- Decisões do PROJETO (os artigos não informam) ---
- Qual bloco é descongelado no nível parcial: `layer4`, o último bloco do ResNet-50.
- Learning rate das camadas descongeladas do backbone (`layer4`): 10x menor que o da cabeça.
- Ativação ReLU e dropout depois da camada densa; saída de 1 logit (P(spoof) = sigmoid(logit)).
- Perda BCE, early stopping na loss de validação (máx. de épocas e paciência), batch, augmentation.
- BatchNorm congelada fica em modo eval (senão as estatísticas continuariam mudando).
- Pesos do torchvision: IMAGENET1K_V2, a versão mais recente e o padrão do torchvision (nenhum dos
  artigos informa a versão; aqui a escolha é explícita). O torchvision avalia esses pesos com resize 232
  e recorte 224; aqui o recorte v2 de 224x224 entra inteiro, sem resize.
- Medição de tempo: ver `evaluate_cnn_timed`.

Rótulos como em src.config: LABEL_LIVE = 0, LABEL_SPOOF = 1.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from tqdm.auto import tqdm

from src.modeling.cnn import evaluate_cnn
from src.modeling.predict import compute_metrics

LEVELS = ('cnn_resnet50_v2_01_head', 'cnn_resnet50_v2_02_partial')

# Partes do ResNet-50 do torchvision descongeladas no nível parcial (a cabeça nova sempre treina).
PARTIAL_UNFROZEN = ('layer4',)


def _sync(device):
    """Espera a GPU terminar o que foi enviado (necessário para medir tempo de GPU)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_model_v2(level, hidden=512, dropout=0.5, weights_name='IMAGENET1K_V2', pretrained=True):
    """
    ResNet-50 com cabeça nova (densa de `hidden` unidades + ReLU + dropout + 1 saída)
    e os parâmetros congelados conforme o nível:

      cnn_resnet50_v2_01_head    : backbone todo congelado, só a cabeça treina
      cnn_resnet50_v2_02_partial : layer4 + cabeça treinam
    """
    if level not in LEVELS:
        raise ValueError(f"Nível desconhecido '{level}'. Use um de {LEVELS}")

    weights = getattr(models.ResNet50_Weights, weights_name) if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )

    # Congela o backbone (tudo menos a cabeça `fc`) conforme o nível.
    for name, module in model.named_children():
        if name == 'fc':
            continue
        trainable = (level == LEVELS[1]) and (name in PARTIAL_UNFROZEN)
        for p in module.parameters():
            p.requires_grad = trainable

    return model


def set_train_mode_v2(model):
    """
    model.train(), mas BatchNorm cujos parâmetros estão todos congelados fica em modo eval.
    Senão as estatísticas (média/variância) continuariam sendo atualizadas com dados do NUAA
    e a camada não estaria realmente congelada.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()


# ---------------------------------------------------------------------------
# Treino (Adam + BCE + early stopping na loss de validação)
# ---------------------------------------------------------------------------
def train_cnn_v2(model, train_loader, val_loader, y_val, device,
                 lr_head=1e-4, lr_backbone=1e-5,
                 max_epochs=30, patience=5, checkpoint_path=None):
    """
    A cabeça nova usa lr_head; as camadas descongeladas do backbone usam lr_backbone.
    Early stopping na loss de validação; os melhores pesos são restaurados no fim
    (e salvos em checkpoint_path, se dado).

    Devolve (model, history DataFrame, info), onde info tem:
      train_s    -> tempo total do treino em segundos (inclui a validação a cada época;
                    não inclui restaurar os pesos nem salvar o checkpoint)
      epochs     -> épocas executadas
      best_epoch -> época com menor loss de validação
    """
    model.to(device)

    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    backbone_params = [p for name, p in model.named_parameters()
                       if p.requires_grad and not name.startswith('fc.')]

    param_groups = [{'params': head_params, 'lr': lr_head}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    optimizer = torch.optim.Adam(param_groups)

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
        set_train_mode_v2(model)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping na época {epoch} (melhor loss de validação {best_loss:.4f}, época {best_epoch})")
                break

    _sync(device)
    train_s = time.perf_counter() - run_start
    print(f"Treino concluído: {len(history)} épocas em {train_s/60:.1f} min "
          f"({train_s/len(history):.1f}s por época em média)")

    if best_state is None:
        raise RuntimeError("Nenhuma época teve loss de validação válida (NaN?). Treino interrompido.")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)

    info = {'train_s': train_s, 'epochs': len(history), 'best_epoch': best_epoch}
    return model, pd.DataFrame(history), info


# ---------------------------------------------------------------------------
# Avaliação com medição de tempo
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_cnn_timed(model, loader, y_true, device, threshold=0.5, desc=None,
                       warmup_batches=3, input_size=224):
    """
    Mesmas métricas de src.modeling.cnn.evaluate_cnn (HTER com limiar fixo de 0,5 na saída
    sigmoide), mais o tempo de predição da rede.

    Tempo medido (campo 'forward_s'): soma, por lote, do tempo do `model(x)` com a GPU
    sincronizada antes e depois, com os dados já na GPU e AMP ligado. Não inclui ler a imagem,
    aplicar o pré-processamento nem copiar o lote para a GPU. Antes de medir, a rede roda
    `warmup_batches` lotes de zeros (aquecimento). Em CPU não há aquecimento e o tempo é o do
    próprio forward.
    O tempo por imagem é forward_s / n (média sobre lotes; o último lote pode ser menor).
    """
    model.eval()
    use_amp = device.type == 'cuda'

    if device.type == 'cuda':
        dummy = torch.zeros(loader.batch_size, 3, input_size, input_size, device=device)
        for _ in range(warmup_batches):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                model(dummy)
        _sync(device)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    scores, total_loss, n, forward_s = [], 0.0, 0, 0.0
    iterator = tqdm(loader, desc=desc, leave=False, unit='batch') if desc else loader
    for x, y in iterator:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x).squeeze(1)
        _sync(device)
        forward_s += time.perf_counter() - t0
        logits = logits.float()
        total_loss += criterion(logits, y).item()
        n += y.numel()
        scores.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(scores)
    y_pred = (scores >= threshold).astype(int)   # 1 = spoof
    results = compute_metrics(y_true, scores, y_pred)
    results['loss'] = total_loss / n
    results['forward_s'] = forward_s
    results['n'] = n
    return results


def preprocess_one(image_path, eval_transform):
    """Ler uma imagem do disco e aplicar o pré-processamento de avaliação.
    É o que se mede como 'extração' do CNN (a rede em si é medida em evaluate_cnn_timed)."""
    img = Image.open(image_path).convert('RGB')
    return eval_transform(img)
