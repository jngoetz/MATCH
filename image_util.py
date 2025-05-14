import pickle
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, measure

from histograms import LuvHistogram

debug_enabled = False

dir = Path(__file__).parent


class CacheMode(Enum):
    DISABLED = 0
    ENABLED = 1
    CACHE_ONLY = 2

    def enabled(self):
        return self != CacheMode.DISABLED


def resize(
    input: Image.Image | Path | str, max_size: int, cache_mode=CacheMode.ENABLED
) -> Image.Image:
    """Resize an image to fit within a max_size x max_size square.
    Caches resized images in './resized' directory.
    """
    if isinstance(input, Path) or isinstance(input, str):
        input_path = Path(input)
        if input_path.exists():
            image = Image.open(input_path)
        else:
            image = Image.open(dir / input_path)
    else:
        image = input

    if image.width > image.height:
        size = (max_size, int(max_size * image.height / image.width))
    else:
        size = (int(max_size * image.width / image.height), max_size)

    if hasattr(image, "filename") and cache_mode.enabled():
        # Caching
        parent = Path(image.filename).parent
        resized_path = dir / "resized" / parent.name / Path(image.filename).name
        resized_path = resized_path.with_name(
            resized_path.stem + f"_{max_size}.png"
        )  # force png

        if resized_path.exists():
            # cache hit
            return Image.open(resized_path)
        if cache_mode == CacheMode.CACHE_ONLY:
            raise ValueError(
                f"input {image.filename} not found in cache ({resized_path})"
            )

        resized_path.parent.mkdir(exist_ok=True)
        resized = image.resize(size)
        resized.save(resized_path)
        return resized

    return image.resize(size)


def image_histo(
    input: Image.Image | Path | str, cache_mode=CacheMode.ENABLED
) -> LuvHistogram:
    """Return the LuvHistogram for an image.
    Caches histograms in './histograms' directory.
    """

    if isinstance(input, str):
        input = Path(input)
        name = Path(input)
    elif isinstance(input, Path):
        name = input
    elif hasattr(input, "filename"):
        name = Path(input.filename)
    else:
        name = None

    if name is not None and cache_mode.enabled():
        # Caching
        parent = name.parent
        histo_path = dir / "histograms" / parent.name / name.with_suffix(".p").name

        if histo_path.exists():
            # cache hit
            try:
                with histo_path.open("rb") as f:
                    return pickle.load(f)
            except:
                pass

        if cache_mode == CacheMode.CACHE_ONLY:
            raise ValueError(f"input {name} not found in cache ({histo_path})")

    if isinstance(input, Path):
        if input.exists():
            image = Image.open(input)
        else:
            image = Image.open(dir / input)
    else:
        image = input
    histo = LuvHistogram.from_image(image)

    if name is not None and cache_mode.enabled():
        histo_path.parent.mkdir(exist_ok=True)
        histo_path.unlink(missing_ok=True)  # remove old file if it exists
        with histo_path.open("wb") as f:
            pickle.dump(histo, f)

    return histo


def remove_background(input: Image.Image | Path) -> Image.Image:
    if isinstance(input, Path):
        image = Image.open(input)
    else:
        image = input

    image_array = np.array(image)
    gray_image = color.rgb2gray(image_array)
    if debug_enabled:
        Image.fromarray((gray_image * 255).astype(np.uint8)).save("debug/gray.png")

    # Threshold the image to obtain a binary mask
    binary_image = gray_image > 0.99
    if debug_enabled:
        Image.fromarray((binary_image * 255).astype(np.uint8)).save("debug/binary.png")

    # Find contours
    contours = measure.find_contours(binary_image, 0.5)

    # Create a mask to keep only non-edge contours
    non_edge_mask = binary_image.copy()

    if debug_enabled:
        full_debug = image.copy()

    for i, contour in enumerate(contours):
        # Convert float coordinates to integers
        contour = np.round(contour).astype(int)

        if debug_enabled:
            debug_image = image.copy()
            ImageDraw.Draw(debug_image).polygon(
                list(map(lambda e: (e[1], e[0]), contour)), outline="black", fill="red"
            )
            debug_image.save(f"debug/contour_{i}.png")
            ImageDraw.Draw(full_debug).polygon(
                list(map(lambda e: (e[1], e[0]), contour)), outline="red", fill="red"
            )

        # Draw filled contour on the mask
        ImageDraw.Draw(Image.fromarray(non_edge_mask)).polygon(
            contour.flatten(), outline=1, fill=1
        )

    if debug_enabled:
        full_debug.save("debug/all_contours.png")

    # Apply the non-edge mask to the original image
    result_image_array = image_array.copy()
    result_image_array = np.dstack(
        (result_image_array, (1 - non_edge_mask) * 255)
    ).astype(np.uint8)
    # result_image_array[:, :, 3] = (1 - non_edge_mask) * 255  # Set alpha channel

    # Convert the result array back to a PIL Image
    result_image = Image.fromarray(result_image_array, "RGBA")
    if hasattr(image, "filename"):
        result_image.filename = image.filename

    return result_image
