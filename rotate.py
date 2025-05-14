import numpy as np
from PIL import Image

from histograms import Luv


def rotate_pix(rgb: tuple[int, int, int], rot: np.array):
    luv = Luv.from_rgb(rgb)
    u, v = np.dot(rot, (luv.u, luv.v))
    return Luv(luv.L, u, v).to_rgb()


def rotate_luv(luv: Luv, deg: float):
    ang = np.radians(deg)
    rot = np.array(
        [
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ]
    )
    u, v = np.dot(rot, (luv.u, luv.v))
    return Luv(luv.L, u, v)


def rotate(image: Image.Image, deg: float):
    ang = np.radians(deg)
    rot = np.array(
        [
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ]
    )
    cache = {}
    for x in range(image.width):
        for y in range(image.height):
            rgb = image.getpixel((x, y))
            if all(map(lambda v: v == 0, rgb)) or all(map(lambda v: v == 255, rgb)):
                continue
            if rgb in cache:
                rgb = cache[rgb]
            else:
                key = rgb
                rgb = rotate_pix(rgb, rot)
                rgb = (rgb * 255).clip(0, 255).astype(int)
                cache[key] = rgb
            image.putpixel((x, y), tuple(rgb))

    return image


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("-r", "--rotation", type=float, default=180)
    parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args()

    img = Image.open(args.image)
    img = rotate(img, args.rotation)
    if args.output:
        img.save(args.output)
    else:
        img.save(args.image.with_name(args.image.stem + "_rotated.png").name)
