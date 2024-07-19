import numpy as np
from torch.functional import F
from tqdm.auto import tqdm
import torch


def segment_many(model, imgs, device="cpu"):
    return [
        segment_image(model, img, device=device)
        for img in tqdm(imgs, desc="Segmenting", total=len(imgs))
    ]


def segment_image(model, img, device="cpu", pad_stride=32):
    img = _ensure_2d(img)
    img = img[None]  # To add channel dimension
    with torch.no_grad():
        tensor = torch.from_numpy(img).to(device)
        tensor = tensor[None]  # To add batch dimension
        padded_tensor, pads = pad_to(tensor, pad_stride)
        res_tensor = model(padded_tensor)
        res_unp = unpad(res_tensor, pads)
        return np.squeeze(res_unp.cpu().detach().numpy())


def _ensure_2d(img, ensure_float=True):
    match img.shape:
        case (_, _):
            img_2d = img
        case (_, _, _):
            img_2d = img[:, :, 0]
        case _:
            raise ValueError("Unexpected img shape")

    if ensure_float:
        max_val = np.max(img_2d)
        if max_val <= 1:
            return np.float32(img_2d)
        elif max_val <= 255:
            return np.float32(img_2d) / 255
        else:
            assumed_type_max = np.ceil(np.log2(max_val))
            return np.float32(img_2d) / assumed_type_max
    else:
        return img_2d


def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2] : -pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0] : -pad[1]]
    return x
