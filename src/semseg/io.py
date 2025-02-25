import imageio
import logging
import numpy as np
import numpy.typing as npt
import os
from semseg.domain import NormalizedImage

logger = logging.getLogger("semseg.io")


def normalize_image(image: npt.NDArray) -> NormalizedImage:
    max_value = image.max()
    min_value = image.min()
    if max_value == min_value:
        logger.warning("Image is a constant!")
        return image
    return (image - min_value) / (max_value - min_value)


def read_image_normalized(
    image_path: os.PathLike,
    dtype=np.float32,
) -> NormalizedImage:
    image = imageio.imread(image_path)
    match image.shape:
        case (_, _):
            image= image.astype(dtype)
        case (_, _, 1):
            image= image[:, :, 0].astype(dtype)
        case (_, _, 2):
            # gray scale with alpha
            image= image[:, :, 0].astype(dtype)
        case _:
            image= rgb2gray(image)

    image= normalize_image(image)
    return image.astype(dtype)

def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])
