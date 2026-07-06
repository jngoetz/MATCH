from typing import Callable

import numpy as np
from PIL import Image

from histograms import Luv, LuvHistogram, Uv, luv_to_rgb, rgb_to_luv
from image_util import image_histo, resize
from mountain import Mountain, default_params

debug_enabled = False


def shift_mountain(image: Image.Image, m: Mountain, to: Uv):
    """Translate every color inside mountain `m` so its peak lands on `to`.
    This is a conservative shift, as it only changes colors that are inside the mountain, and leaves all other colors untouched.
    If a more general shift is desired, use `shift_all` instead, which translates every color by the same amount.
    """
    def filter_inside(coord):
        return (5 <= coord[0] <= 95) and (coord in m)
    
    shift_dist = np.asarray(to) - np.asarray(m.peak)

    return shift_to(image, shift_dist, filter_inside)

def dist_to(h: LuvHistogram, m: Mountain, to: Uv|Luv):
    if isinstance(to, Luv):
        # For Luv, we need the most common L for mountain's peak to get a full Luv shift vector,
        # since the mountain only encapsulates the uv.
        return np.asarray(to) - np.asarray(h.max_L(m.peak))
    else:
        return np.asarray(to) - np.asarray(m.peak)

def shift_all(image: Image.Image, dist: np.array):
    """Translate every color by `dist` (either Luv or Uv)
    Use `dist_to` helper for computing the shift distance for typical cases (e.g. to move a mountain's peak to a target color).

    For a more conservative shift that only moves colors inside a mountain, use `shift_mountain` instead.

    Note: Leaves black/white colors untouched, as they are often not part of the main color palette and can be considered noise. This is done by filtering out colors with L < 5 or L > 95.
    These often don't shift into proper RGB colors anyway and would get clamped back to black/white.
    """
    return shift_to(image, dist, lambda coord: 5 <= coord[0] <= 95)  # only shift colors with L in [5, 95]


def shift_to(image: Image.Image, dist: np.array, luv_filter: Callable[[np.array], bool]|None = None):
    """Translate every color matching `filter` by `dist` (either Luv or Uv)

    The optional `filter` argument is a callable that takes a Luv coordinate and returns True if the color should be shifted, or False if it should be left alone.
    This allows for more selective shifting of colors.
    """

    # Strategy: the work is done per *distinct color*, not per pixel. Take the
    # unique RGB values (`np.unique` with `return_inverse` for maping the results
    # back to pixels), convert that small set to CIELUV once with the vectorised
    # colorspace helpers, shift only the ones matching the filter, convert
    # those back once, then scatter the results to every pixel via `inverse`.

    arr = np.asarray(image)
    out = arr.copy()

    nch = arr.shape[2] # 3 for RGB, 4 for RGBA
    flat = arr.reshape(-1, nch)
    out_flat = out.reshape(-1, nch)

    has_alpha = nch >= 4
    opaque = flat[:, 3] != 0 if has_alpha else np.ones(len(flat), dtype=bool)

    # Convert each distinct color once, in the same D65 CIELUV space the
    # histograms (and therefore the mountains) are built in.
    uniq, inverse = np.unique(flat[opaque, :3], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)

    new_uniq = uniq.copy()
    changed = np.zeros(len(uniq), dtype=bool)

    if len(uniq):
        luv = rgb_to_luv(uniq.astype(float) / 255)

        inside = []
        for i, coord in enumerate(luv):
            if luv_filter is None or luv_filter(coord):
                inside.append(i)

        if inside:
            shifted = luv[inside].copy()

            if len(dist) == 2: # Uv mode
                shifted[:, 1:] += dist
            else: # Luv mode
                shifted += dist

            rgb = np.round((luv_to_rgb(shifted) * 255).clip(0, 255)).astype(uniq.dtype)
            new_uniq[inside] = rgb
            changed[inside] = np.any(rgb != uniq[inside], axis=1)


    # Scatter each unique color's result back to every opaque pixel that had it
    # (alpha/transparent pixels are left untouched via the initial copy).
    out_flat[opaque, :3] = new_uniq[inverse]
    converted = int(changed[inverse].sum())

    if debug_enabled:
        print(f"Converted {converted} pixels")

    return Image.fromarray(out), converted


def rotate_colors(image: Image.Image, deg: float):
    """Rotate every color by ``deg`` in the CIELUV u/v plane (leaving L fixed).

    Same per-distinct-color strategy as ``shift_mountain``: convert the unique
    RGB values to CIELUV once, apply the rotation to the whole batch, convert
    back, and scatter to pixels via the ``inverse`` index. Alpha is preserved.
    """
    rad = np.radians(deg)
    rot = np.array(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ]
    )

    arr = np.asarray(image)
    out = arr.copy()
    nch = arr.shape[2]
    flat = arr.reshape(-1, nch)
    out_flat = out.reshape(-1, nch)

    # Convert each distinct color once, rotate u/v (leave L and any alpha as-is).
    uniq, inverse = np.unique(flat[:, :3], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    luv = rgb_to_luv(uniq.astype(float) / 255)
    luv[:, 1:] = luv[:, 1:] @ rot.T  # each row -> rot @ [u, v]
    rgb = np.round((luv_to_rgb(luv) * 255).clip(0, 255)).astype(uniq.dtype)

    out_flat[:, :3] = rgb[inverse]
    return Image.fromarray(out)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("-f", "--from-peak", type=str, required=False)
    parser.add_argument("-t", "--to", type=str, required=True)

    args = parser.parse_args()

    image = resize(args.image, 160)
    histo = image_histo(image).without_white(10).smooth(3)
    uv = histo.collapse_L()

    if args.from_peak is not None:
        from_color = Uv(*map(int, args.from_peak.split(",")))
        mtn = uv.mountain(default_params, from_color)
    else:
        mtns = uv.mountains(default_params, 2)
        mtn = mtns[0] if (0, 0) not in mtns[0] or len(mtns) < 2 else mtns[1]

    print(f"Converting {mtn}")

    result, _ = shift_mountain(image, mtn, Uv(*map(int, args.to.split(","))))
    out_path = (Path(".") / args.image.name).with_name(
        args.image.stem + "_converted.png"
    )
    result.save(out_path)
