import imageio
import logging
import numpy as np
import numpy.typing as npt

import os

logger = logging.getLogger("cvrsegmentation.io")


def normalize_image(image: npt.NDArray) -> npt.NDArray[np.float32]:
    max_value = image.max()
    min_value = image.min()
    if max_value == min_value:
        logger.warning("Image is a constant!")
        return image
    return (image - min_value) / (max_value - min_value)


def read_image_normalized(
    image_path: os.PathLike,
    dtype=np.float32,
) -> npt.ArrayLike:
    image = imageio.imread(image_path)
    image = image.astype(np.float32)
    image = normalize_image(image)
    return image.astype(dtype)
