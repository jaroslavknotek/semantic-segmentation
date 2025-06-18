
from pathlib import Path
import semseg.io as io
from tqdm.cli import tqdm
import semseg.imageutils as iu
import semseg.helpers as hpr
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
        transform: AugTransform|None = None,
        img_channels = 1,
    ):
        assert len(images) == len(labels), f"{len(images)=}!={len(labels)=}"
        
        self.images = [np.float32(img) for img in images]
        self.labels = [np.float32(label) for label in labels]
        
        self.transform = transform
        self.img_channels = img_channels

    def __len__(self) -> int:
        return len(self.images)

    def _transform(self, image, label) -> Tuple[npt.NDArray, npt.NDArray]:
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            tr_image = transformed["image"]
            tr_label = transformed["mask"]
            return tr_image, tr_label
        else:
            return image,label

    def __getitem__(self, idx) -> Dict[str, npt.NDArray]:
        image = self.images[idx]
        label = self.labels[idx]
        image_aug, y = self._transform(image, label)

        x = np.stack([image_aug]*self.img_channels)
        if len(y.shape) > 2:
            # e.g. channel last
            y = np.rollaxis(y,-1)
        
        return x,y



class DelisaSegmentationDataset(SegmentationDataset):
    # def __init__(
    #     self,
    #     images: List[npt.NDArray],
    #     labels: List[npt.NDArray],
    #     transform: AugTransform|None = None,
    #     x_channels = 1,
    # ):
    #     super().__init__(images,labels,transform=transform,x_channels = x_channels)

    def __getitem__(self, idx) -> Dict[str, npt.NDArray]:
        x,plain_y = super().__getitem__(self,ids)
        y = label_to_classes(plain_y)

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
    non_empty = True,
    erasing = False,
    square_sym = False,
    grid_shuffle = False
) -> AugTransform:
    patch_size_padded = int(patch_size * 1.5)
    transform_list = [
        A.PadIfNeeded(patch_size_padded, patch_size_padded),
    ]
    
    # crop to save performance for more advanced ops
    if non_empty:
        transform_list.append(
            A.CropNonEmptyMaskIfExists(
                height=patch_size_padded,
                width=patch_size_padded,
                ignore_values=None,
                ignore_channels=None,
                p=1,
            )
        )
    else:
        A.RandomCrop(patch_size_padded, patch_size_padded)
        
    if advanced:
        transform_list += [
            A.Perspective(p=.2),
            A.GridDistortion(p=.2,num_steps=17),
            A.RandomGridShuffle(p=.2,grid=(13,13)),
        ]
        
    # this potentially destroys images
    # if brightness_contrast:
    #     transform_list += [
    #         A.RandomBrightnessContrast(p=0.5), 
    #     ]
            
    if erasing:
        transform_list.append(
            A.Erasing(
                p=.5,
                fill_mask=0,
                fill = 'random_uniform',
                scale = (.05,.2)
            )
        )
    if rotate_deg is not None:
        transform_list += [
            A.Rotate(limit=rotate_deg, interpolation=interpolation),
        ]
    transform_list += [A.CenterCrop(patch_size, patch_size)]
    
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
        
    if square_sym:
        transform_list += [
            A.SquareSymmetry(p = .6),
        ]
    
    if grid_shuffle:
        transform_list += [
            A.RandomGridShuffle(grid=(3, 3), p=.8)
        ]
        
    return A.Compose(transform_list)


def prepare_dataloaders(
    imgs: List[npt.NDArray],
    labels: List[npt.NDArray],
    train_augumentation_fn: AugTransform,
    val_augumentation_fn: AugTransform,
    batch_size: int = 32,
    val_size: float = 0.33,
    seed: int = 123,
    img_channels = 1,
) -> Tuple[DataLoader, DataLoader]:
    img_train, img_val, label_train, label_val = ms.train_test_split(
        imgs, labels, test_size=val_size, random_state=seed
    )

    dataset_train = SegmentationDataset(
        img_train,
        label_train,
        train_augumentation_fn,
        img_channels = img_channels
    )
    dataset_val = SegmentationDataset(
        img_val,
        label_val,
        val_augumentation_fn,
        img_channels,
    )

    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers = 4,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        num_workers = 4,
        shuffle=False,
    )
    return train_dataloader, val_dataloader


def read_img_w_labels(
    data_root,
    image_name = "img.png", 
    label_name = "label.png", 
    use_tqdm = False,
    small_filter = None
):
    if small_filter is None:
        small_filter = 0
    
    img_paths = list(Path(data_root).rglob(image_name))
    label_paths = [ p.parent/'label.png' for p in img_paths]

    img_iter = tqdm(
        img_paths,
        disable = not use_tqdm, 
        desc = "Loading images"
    )
    imgs = list(map(io.read_image_normalized,img_iter))
    labels = map(io.read_image_normalized,label_paths)
    lbl_iter = tqdm(
        labels,
        total=len(imgs),
        disable = not use_tqdm, 
        desc = f"Loading and filtering {small_filter=} labels"
    )
    labels = [ iu.filter_small(label,small_filter) for label in lbl_iter]
    
    return {p.parent.stem:(img,label) for p,(img,label) in zip(img_paths, zip(imgs,labels))}


def load_train_test_images_w_labels(train_root,small_filter, test_filepath = None, use_tqdm=False):    
    train_dict = read_img_w_labels(train_root,small_filter = small_filter,use_tqdm = use_tqdm)

    test_img_names =  None
    if test_filepath is None:
        test_filepath = train_root/'test.txt'
        
    with open(test_filepath) as f:
        test_img_names = set([l.strip() for l in f.readlines()])
        

    tests = [ train_dict[name] for name in test_img_names]
    test_imgs, test_labels = hpr.unzip2(tests)
    test_imgs, test_labels = list(test_imgs), list(test_labels)

    imgs,labels = hpr.unzip2([ v for k,v in train_dict.items() if k not in test_img_names])
    imgs,labels = list(imgs),list(labels)
    if len(test_imgs) + len(imgs) != len(train_dict):
        logging.warning("Number of train + test does not match the number of all images. Most likely test contains names that were not found")
        
    return (imgs, labels),(test_imgs,test_labels)