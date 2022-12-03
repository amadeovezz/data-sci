from typing import Tuple

# 3rd party
from PIL import Image
import numpy as np
from numpy import linalg

# lib
from processing import filtering, normalize, kernels


class ImgCli:

    def __init__(self, filepath: str, display_size: Tuple[int, int] = None):
        """
        @param filepath: the location of the image
        @param grey: only support grey scale images for now (or conversions to grey)
        """
        # img is uint8
        img = Image.open(filepath).convert('L')
        self.img = np.asarray(img)
        self.display_size = display_size

    def display(self, img: np.ndarray):
        norm = normalize.normalize(img)
        i = Image.fromarray(norm, mode='L')
        if self.display_size is not None:
            i = i.resize(self.display_size)
        return i

    def show(self) -> Image:
        return self.display(self.img)

    def filter(self, kernel: np.ndarray, convolve: bool = False) -> Image:
        filtered_img = filtering.filter(self.img, kernel, convolve=convolve)
        return self.display(filtered_img)

    def smooth(self, method: str = 'uniform', size: int = 1):
        k = kernels.create_kernel(method, size)
        self.img = filtering.filter(self.img, k)

    def gradient_image(self):
        img = self.img
        grad_x = kernels.create_kernel('gradient_x')
        grad_y = kernels.create_kernel('gradient_y')
        img_x = filtering.filter(img, grad_x)
        img_y = filtering.filter(img, grad_y)
        # Compute magnitude
        combined = np.stack((img_x, img_y))
        magnitude = np.linalg.norm(combined, axis=0)
        return self.display(magnitude)
