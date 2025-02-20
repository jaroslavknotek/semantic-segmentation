import numpy as np
import albumentations as A
import cv2
from typing import List, Tuple, Dict, Protocol
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sklearn.model_selection as ms
import numpy.typing as npt
from typing import TypeAlias

AugOutput: TypeAlias = dict[str, npt.NDArray]


class AugTransform(Protocol):
    def __call__(self, image: npt.NDArray, mask: npt.NDArray) -> AugOutput: ...


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: List[npt.NDArray],
        labels: List[npt.NDArray],
        transform: AugTransform,
        x_channels = 1,
    ):
        assert len(images) == len(labels), f"{len(images)=}!={len(labels)=}"
        
        self.images = [np.float32(img) for img in images]
        self.labels = [np.float32(label) for label in labels]
        
        self.transform = transform
        
        self.x_channels = x_channels

    def __len__(self) -> int:
        return len(self.images)

    def _transform(self, image, label) -> Tuple[npt.NDArray, npt.NDArray]:
        transformed = self.transform(image=image, mask=label)
        tr_image = transformed["image"]
        tr_label = transformed["mask"]
        return tr_image, tr_label

    def __getitem__(self, idx) -> Dict[str, npt.NDArray]:
        image = self.images[idx]
        label = self.labels[idx]
        image_aug, label_aug = self._transform(image, label)

        y = label_to_classes(label_aug)
        x = np.stack([image_aug]*self.x_channels)

        return {
            "x": x,
            "y": y,
        }


def label_to_classes(label: npt.NDArray) -> npt.NDArray:
    """
    Convert a label image to a 3-channel image with classes 'foreground',
    'background', 'border'.

    Parameters

    label: np.ndarray

    Returns

    np.ndarray
    """
    # label must be of type float32
    foreground = np.float32(label > 0)
    background = 1 - foreground
    border = _get_border(foreground)

    return np.stack([foreground, background, border])


def _get_border(foreground_label: npt.NDArray) -> npt.NDArray:
    fg_int = np.uint8(foreground_label)
    kernel = np.ones((3, 3))
    eroded = cv2.morphologyEx(fg_int, cv2.MORPH_ERODE, kernel)
    return np.float32(fg_int - eroded)


def setup_augumentation(
    patch_size: int,
    *,
    advanced: bool = False,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
    blur_sharp_power: float | None = None, 
    noise_value: float | None = None,  
    rotate_deg: int | None = None,  
    interpolation: int = cv2.INTER_CUBIC,
) -> AugTransform:
    patch_size_padded = int(patch_size * 1.5)
    transform_list = [
        A.PadIfNeeded(patch_size_padded, patch_size_padded),
        A.RandomCrop(patch_size_padded, patch_size_padded),
    ]
       
    if advanced:
        transform_list += [
            A.Perspective(p=.2),
            A.GridDistortion(p=.2,num_steps=17),
            A.RandomGridShuffle(p=.2,grid=(13,13)),
        ]
    if rotate_deg is not None:
        transform_list += [
            A.Rotate(limit=rotate_deg, interpolation=interpolation),
        ]
    
    # this potentially destroys images
    # if brightness_contrast:
    #     transform_list += [
    #         A.RandomBrightnessContrast(p=0.5), 
    #     ]
    if noise_value is not None:
        transform_list += [
            A.augmentations.transforms.GaussNoise(noise_value, p=0.5),
        ]

    if blur_sharp_power is not None:
        transform_list += [
            A.OneOf(
                [
                    A.Sharpen(p=1, alpha=(0.2, 0.2 * blur_sharp_power)),
                    A.AdvancedBlur(p=1),
                ],
                p=0.3,
            ),
        ]

    if flip_horizontal:
        transform_list += [
            A.HorizontalFlip(p=0.5),
        ]
    if flip_vertical:
        transform_list += [
            A.VerticalFlip(p=0.5),
        ]

    transform_list += [A.CenterCrop(patch_size, patch_size)]
    return A.Compose(transform_list)


def prepare_dataloaders(
    imgs: List[npt.NDArray],
    labels: List[npt.NDArray],
    train_augumentation_fn: AugTransform,
    val_augumentation_fn: AugTransform,
    batch_size: int = 32,
    val_size: float = 0.33,
    seed: int = 123,
    x_channels = 1,
) -> Tuple[DataLoader, DataLoader]:
    img_train, img_val, label_train, label_val = ms.train_test_split(
        imgs, labels, test_size=val_size, random_state=seed
    )

    dataset_train = SegmentationDataset(
        img_train,
        label_train,
        train_augumentation_fn,
        x_channels = x_channels
    )
    dataset_val = SegmentationDataset(
        img_val,
        label_val,
        val_augumentation_fn,
        x_channels,
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, val_dataloader
