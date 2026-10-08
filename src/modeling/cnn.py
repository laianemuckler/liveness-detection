"""
CNN (DenseNet201) functions: dataset, model construction with the three
freezing levels, training loop and evaluation.

Design decisions and where they come from:

- Backbone: DenseNet201 pre-trained on ImageNet. Choice supported by the
  architecture comparison in "Optimizing CNN Architectures for Face
  Liveness Detection" (CMES), where it ranked first on NUAA.
- Freezing levels (cnn_01_head / cnn_02_partial / cnn_03_full): rests on
  the generic-to-specific layer hierarchy and the frozen vs. fine-tuned
  comparison of Yosinski et al. (2014), "How transferable are features in
  deep neural networks?". Which part is unfrozen in the partial level
  (denseblock4 + norm5) is a project decision, not taken from a paper.
- Optimizer (Adam), loss (binary cross-entropy), dropout (0.5),
  augmentation types (flip, rotation, brightness) and early stopping on
  validation loss follow the CMES paper. Augmentation magnitudes,
  patience, max epochs, batch size and learning rates for unfrozen
  layers are project decisions (the paper does not report them).
- Input: 160x160 RGB, the same MTCNN crops used by LBP and HOG (no resize).

Labels follow src.config: LABEL_LIVE = 0, LABEL_SPOOF = 1. The model
outputs one logit; sigmoid(logit) = P(spoof).
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.modeling.predict import compute_metrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EXPERIMENTS = ('cnn_01_head', 'cnn_02_partial', 'cnn_03_full')

# Parts of torchvision's densenet201.features that are unfrozen in the
# partial experiment. The new classifier is always trainable.
PARTIAL_UNFROZEN = ('denseblock4', 'norm5')


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_transforms(rotation_degrees=10, brightness=0.2):
    """
    Returns (train_transform, eval_transform).
    Augmentation (flip, rotation, brightness) is applied only in training.
    Both use ImageNet normalization, which the pre-trained weights expect.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation_degrees),
        transforms.ColorJitter(brightness=brightness),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class FASDataset(Dataset):
    """
    Images are read from disk once and kept in memory as uint8 arrays,
    so epochs do not re-read from Google Drive (slow in Colab).
    ~210 MB for train, ~700 MB for test at 160x160x3.
    """

    def __init__(self, image_paths, labels, transform):
        self.labels = np.array(labels, dtype=np.float32)
        self.transform = transform
        self.images = [np.array(Image.open(p).convert('RGB')) for p in image_paths]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), self.labels[idx]


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(experiment, dropout=0.5, pretrained=True):
    """
    DenseNet201 with a new 1-output classifier (dropout + linear), and the
    parameters frozen according to the experiment:

      cnn_01_head    : everything frozen, only the new classifier trains
      cnn_02_partial : denseblock4 + norm5 + classifier train
      cnn_03_full    : everything trains
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'. Use one of {EXPERIMENTS}")

    weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet201(weights=weights)

    in_features = model.classifier.in_features  # 1920
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )

    # Freeze the backbone according to the experiment.
    for name, module in model.features.named_children():
        if experiment == 'cnn_03_full':
            trainable = True
        elif experiment == 'cnn_02_partial':
            trainable = name in PARTIAL_UNFROZEN
        else:  # cnn_01_head
            trainable = False
        for p in module.parameters():
            p.requires_grad = trainable

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _set_train_mode(model):
    """
    model.train(), but BatchNorm layers whose parameters are frozen stay in
    eval mode. Otherwise their running statistics would keep updating on
    NUAA data and the layer would not really be frozen.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()


# ---------------------------------------------------------------------------
# Evaluation (metrics come from src.modeling.predict.compute_metrics)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_scores(model, loader, device):
    """Returns (P(spoof) scores, mean BCE loss) over a loader."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    scores, total_loss, n = [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(x).squeeze(1)
        logits = logits.float()
        total_loss += criterion(logits, y).item()
        n += y.numel()
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores), total_loss / n


def evaluate_cnn(model, loader, y_true, device, threshold=0.5):
    """
    Runs the model on a loader and returns the same metrics dict as
    src.modeling.predict.evaluate (+ 'loss'), so it can go straight into
    log_experiment. HTER uses the fixed threshold (0.5 on the sigmoid
    output), the CNN equivalent of the SVM's default decision rule.
    """
    scores, loss = predict_scores(model, loader, device)
    y_pred = (scores >= threshold).astype(int)  # 1 = spoof
    results = compute_metrics(y_true, scores, y_pred)
    results['loss'] = loss
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_cnn(model, train_loader, val_loader, y_val, device,
              lr_head=1e-3, lr_backbone=1e-4,
              max_epochs=30, patience=5, checkpoint_path=None):
    """
    Trains with Adam and binary cross-entropy. The new classifier uses
    lr_head; unfrozen backbone layers use the smaller lr_backbone.
    Early stopping on validation loss; the best weights are restored at
    the end (and saved to checkpoint_path if given).

    Returns (model, history DataFrame).
    """
    model.to(device)

    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]

    param_groups = [{'params': head_params, 'lr': lr_head}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    optimizer = torch.optim.Adam(param_groups)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    history = []
    run_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        _set_train_mode(model)
        running_loss, n = 0.0, 0
        for x, y in train_loader:
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
        train_time = time.time() - epoch_start

        val_res = evaluate_cnn(model, val_loader, y_val, device)
        epoch_time = time.time() - epoch_start
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_res['loss'],
            'val_HTER': val_res['HTER'],
            'val_AUC': val_res['AUC'],
            'train_time_s': train_time,
            'epoch_time_s': epoch_time,
        })
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | "
              f"val loss {val_res['loss']:.4f} | val HTER {val_res['HTER']*100:.2f}% | "
              f"treino {train_time:.1f}s | epoca {epoch_time:.1f}s")

        if val_res['loss'] < best_loss:
            best_loss = val_res['loss']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss {best_loss:.4f})")
                break

    total_time = time.time() - run_start
    print(f"Treino concluido: {len(history)} epocas em {total_time/60:.1f} min "
          f"({total_time/len(history):.1f}s por epoca em media)")

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)

    return model, pd.DataFrame(history)
