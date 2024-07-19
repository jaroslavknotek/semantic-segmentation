---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
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

root = Path('<path to data>')
test_root = Path('<path to test data>')


img_paths = list(root.rglob('img.png'))*10 # HACK
#label_paths = [ p.parent/'label.png' for p in img_paths]
label_paths = [ p.parent/'mask.png' for p in img_paths]

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
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
from cvrsegmentation.training import train

train_loss,val_loss = train(
    model,
    train_dl,
    val_dl,
    loss,
    checkpoint_path='/tmp/training/delisa2023/checkpoint.pth',
    patience=20,
    scheduler_patience= 10
    device = device
)
```

# Test

```python
test_img_paths = list(test_root.rglob('img.png'))
test_label_paths = [ p.parent/'mask.png' for p in test_img_paths]

test_imgs = list(map(io.read_image_normalized,tqdm(test_img_paths)))
test_labels = list(map(io.read_image_normalized,tqdm(test_label_paths)))
```

```python
import cvrsegmentation.prediction as prd

preds = prd.segment_many(model,test_imgs,device = device)
```

```python
import cvrsegmentation.instance_matching as im

thr = .5

precisions = []
recalls = []
for img,label,pred in zip(test_imgs,test_labels,preds):
    foreground = pred[0]
    pred_thr = np.uint8(foreground>thr)
    label_thr = np.uint8(label)

    precision,recall,f1 = im.measure_precision_recall_f1(pred_thr,label_thr)
    precisions.append(precision)
    recalls.append(recall)
    _,(axi,axl,axp) = plt.subplots(1,3,figsize = (12,4))
    
    axi.imshow(img)
    axl.imshow(label)
    axp.imshow(pred_thr)
    plt.suptitle(f"{f1 =} {precision=} {recall=}")
    plt.show()
```

```python
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = 2* (mean_precision*mean_recall)/(mean_precision + mean_recall)

f"{mean_precision=} {mean_recall=} {mean_f1=}"
```
