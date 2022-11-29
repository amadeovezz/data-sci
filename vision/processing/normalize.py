import numpy as np


def normalize(img: np.ndarray, type: str = 'linear') -> np.ndarray:
    if type == 'linear':
        # Bring minimum to 0
        new_img = img - np.min(img)
        # Get scale value (Solve for max_intensity * scale_value = 255)
        # scale_value = 255 / max_intensity
        scale_value = 255.0 / np.max(new_img)
        # cast to uint8 here
        return (scale_value * new_img).astype(np.uint8)
