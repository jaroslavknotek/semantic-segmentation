import numpy.typing as npt
from typing import TypeAlias
import numpy as np

Prediction: TypeAlias = npt.NDArray[np.float32]
NormalizedImage: TypeAlias = npt.NDArray[np.float32]
BinMask: TypeAlias = npt.NDArray[np.uint8]


class ModelPrediction(np.ndarray):
    def __new__(
        cls,
        foreground: Prediction,
        **kwargs,
    ) -> "ModelPrediction":
        return super().__new__(
            cls,
            foreground.shape,
            dtype=foreground.dtype,
            buffer=foreground.flatten(),
        )

    def __init__(self, foreground, background=None, border=None, artifacts=None):
        self.background = background
        self.border = border
        self.foreground = foreground
        self.artifacts = artifacts

    def to_uint8_gs_image(self):
        return (self * 255).astype(np.uint8)
