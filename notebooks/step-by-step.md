---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: computer-vision
    language: python
    name: .venv
---

```python
%load_ext autoreload
%autoreload 2
%cd ..
```

```python
import logging
logging.basicConfig(level=logging.INFO)

import cvrsegmentation.io as io
```

```python
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(<path>)

img_paths = list(root.rglob('img.png'))*10 # HACK
label_paths = [ p.parent/'label.png' for p in img_paths]

imgs = list(map(io.read_image_normalized,tqdm(img_paths)))
labels = list(map(io.read_image_normalized,tqdm(label_paths)))
```

# Prepare DataLoaders

```python
import cvrsegmentation.dataset as ds

train_augumentation_fn = ds.setup_augumentation(
    patch_size = 256,
    elastic=True,
    brightness_contrast=True,
    blur_sharp_power=1,
    flip_horizontal=True,
    flip_vertical=True,
    noise_value=0.01,
    rotate_deg=45
    )
val_augumentation_fn = ds.setup_augumentation(patch_size = 256)

train_dl,val_dl = ds.prepare_dataloaders(
    imgs,
    labels,
    train_augumentation_fn,
    val_augumentation_fn,
    batch_size= 32,
    val_size = 0.33
)
```

# Model

```python
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=3,
    activation="sigmoid"
)
```

# Loss

```python
from cvrsegmentation.training import FocalLoss

loss = FocalLoss(alpha = .8,gamma=2)
```

# Training



```python
from cvrsegmentation.training import train

train_loss,val_loss = train(
    model,
    train_dl,
    val_dl,
    loss,
    checkpoint_path='/tmp/training/',
    patience=20
)
```
