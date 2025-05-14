import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import LuvColor, sRGBColor
from PIL import Image

from histograms import Uv
from image_util import image_histo, resize
from mountain import Mountain, default_params

debug_enabled = False


def shift_mountain(image: Image.Image, m: Mountain, to: Uv):
    i2 = image.copy()

    dist = np.asarray(to) - m.peak
    if debug_enabled:
        print(f"Converting {m.peak} to {to} (dist: {dist})")

    converted = 0
    cache = {}
    for x in range(image.width):
        for y in range(image.height):
            rgb = image.getpixel((x, y))
            rgb_old = rgb
            luv = np.asarray(
                convert_color(
                    sRGBColor(*rgb, is_upscaled=True), LuvColor
                ).get_value_tuple()
            )
            if luv not in m:
                continue

            luv[1] += dist[0]
            luv[2] += dist[1]
            if tuple(luv) in cache:
                rgb = cache[tuple(luv)]
            else:
                rgb = convert_color(LuvColor(*luv), sRGBColor).get_value_tuple()
                rgb = (np.asarray(rgb) * 255).clip(0, 255)
                rgb = tuple(rgb.astype(int))
                cache[tuple(luv)] = rgb
            i2.putpixel((x, y), rgb)
            if rgb != rgb_old:
                converted += 1

    if debug_enabled:
        print(f"Converted {converted} pixels")

    return i2, converted


def rotate_colors(image: Image.Image, deg: float):
    i2 = image.copy()
    rad = np.radians(deg)
    rot = np.array(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ]
    )

    for x in range(image.width):
        for y in range(image.height):
            rgb = image.getpixel((x, y))
            luv = np.asarray(
                convert_color(
                    sRGBColor(*rgb, is_upscaled=True), LuvColor
                ).get_value_tuple()
            )
            luv[1:] = np.dot(rot, luv[1:])  # rotate the u and v values, leave L as-is
            rgb = convert_color(LuvColor(*luv), sRGBColor).get_value_tuple()
            rgb = (np.asarray(rgb) * 255).clip(0, 255)
            i2.putpixel((x, y), tuple(rgb.astype(int)))

    return i2


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
