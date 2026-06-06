from PIL import Image
import numpy as np
import random
import torch

def set_seed(seed = None):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

def add_random_background(image, bg_color=None):
    bg_color = np.random.rand() * 255 if bg_color is None else bg_color
    image = np.array(image)
    rgb, alpha = image[..., :3], image[..., 3:]
    alpha = alpha.astype(np.float32) / 255.0
    image_new = rgb * alpha + bg_color * (1 - alpha)
    return Image.fromarray(image_new.astype(np.uint8))

def do_resize_content(image: Image.Image, scale_rate: float) -> Image.Image:
    """
    Resize image content while keeping the original canvas size.

    Args:
        image: PIL Image
        scale_rate: Scale factor in (0, 1].
                    1.0 = no change
                    0.8 = shrink content to 80%
                    0.5 = shrink content to 50%

    Returns:
        PIL Image with same size as input.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image")

    if not (0 < scale_rate <= 1):
        raise ValueError("scale_rate must be in (0, 1]")

    if scale_rate == 1:
        return image.copy()

    original_size = image.size

    # Calculate resized dimensions
    new_width = max(1, int(original_size[0] * scale_rate))
    new_height = max(1, int(original_size[1] * scale_rate))

    # High-quality resize
    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create transparent canvas for RGBA images
    if image.mode == "RGBA":
        canvas = Image.new(
            "RGBA",
            original_size,
            (0, 0, 0, 0)
        )

        paste_position = (
            (original_size[0] - new_width) // 2,
            (original_size[1] - new_height) // 2
        )

        canvas.paste(
            resized_image,
            paste_position,
            resized_image
        )

    else:
        # RGB, L, etc.
        canvas = Image.new(
            image.mode,
            original_size,
            0
        )

        paste_position = (
            (original_size[0] - new_width) // 2,
            (original_size[1] - new_height) // 2
        )

        canvas.paste(
            resized_image,
            paste_position
        )

    return canvas

def get_camera_for_index(data_index):
    """
    Camera viewpoints used in the dataset.

    Coordinate system:
        Elevation (ev):
            +90° = top view
            -90° = bottom view
             0° = horizontal view

        Azimuth (azimuth):
              0° = front view
            -90° = left view
             90° = right view
            180° = back view

    Dataset mapping:

        Index  View     Elevation  Azimuth
        -----  -------  ---------  -------
          0    Front       0°         0°
          1    Left        0°       -90°
          2    Bottom    -90°         0°
          3    Back        0°       180°
          4    Right       0°        90°
          5    Top        90°         0°
    """

    camera_views = {
        0: {"name": "front",  "elevation": 0,   "azimuth": 0},
        1: {"name": "left",   "elevation": 0,   "azimuth": -90},
        2: {"name": "bottom", "elevation": -90, "azimuth": 0},
        3: {"name": "back",   "elevation": 0,   "azimuth": 180},
        4: {"name": "right",  "elevation": 0,   "azimuth": 90},
        5: {"name": "top",    "elevation": 90,  "azimuth": 0},
    }

    return camera_views[data_index]