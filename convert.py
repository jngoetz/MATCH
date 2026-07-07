from enum import Enum
from typing import Callable

import numpy as np
from PIL import Image

from histograms import Luv, LuvHistogram, Uv, luv_to_rgb, rgb_to_luv
from image_util import image_histo, resize
from mountain import Mountain, default_params

debug_enabled = False


class GamutMode(Enum):
    """How to resolve a shifted color that falls outside the sRGB gamut.

    A translation in CIELUV can land on a color sRGB cannot display: either the
    lightness leaves [0, 100] (e.g. a dark pixel shifted darker than black), or,
    because the gamut collapses toward a point as L -> 0, a low-lightness color
    simply can't hold that much chroma. `luv_to_rgb` then returns channels
    outside [0, 1] that must be brought back somehow. The mode decides how.

    Example: a dark maroon skin pixel `sRGB(29, 13, 13)` (L~=5)
    shifted by `Luv(-19, -6, -6)` (the target lands below L=0, out of gamut):

        mode        result sRGB   what happened
        ----------  ------------  --------------------------------------------
        CLIP        (0, 0, 83)    vivid blue -- hue destroyed (see below)
        DIRECTION   (19, 0, 4)    dark red -- hue kept, ~20% of the shift applied
        CHROMA      (0, 0, 0)     black -- hits target darkness, chroma dropped

    CLIP (naive, the historical behavior)
        Clamp each of R, G, B into [0, 1] independently. This does NOT find the
        nearest real color -- it slides the point along a face/edge of the RGB
        cube, arbitrarily changing hue and chroma. The example's unclamped RGB
        is `(-0.55, -0.23, +0.32)`; the negative R and G clip to 0 while the
        spurious positive B survives, turning a dark brown into pure blue. On a
        dark, noisy photo this sprays vivid blue/magenta/red confetti, because
        near-identical boundary pixels clip to wildly different hues. Kept only
        for comparison / backwards compatibility.

    DIRECTION (default)
        Keep the shift *direction* and binary-search the largest fraction
        `t in [0, 1]` of the vector that still lands in gamut -- i.e. the point
        where the segment `original -> shifted` pierces the gamut boundary.
        Preserves the original hue and lightness ordering, so shadow texture
        survives. Trade-off: pixels that can't reach the target stop partway, so
        the transform is no longer a rigid translation -- unreachable regions are
        deliberately under-shifted (the example only moves ~20% of the way).
        The in-gamut bulk (including the mountain peak) still lands exactly on
        target; only the out-of-gamut tails are scaled back.

    CHROMA
        Clamp L into [0, 100], then scale (u, v) toward the neutral axis until
        the color re-enters the gamut. Honors the target *lightness* exactly and
        only sacrifices saturation. Trade-off: colors whose target is at/near
        L=0 collapse to black, crushing shadow detail (the example goes to pure
        black because its target L was negative).

    Both DIRECTION and CHROMA leave hue intact and eliminate the confetti; they
    differ in what they give up for unreachable colors -- DIRECTION keeps
    lightness/detail but undershoots the target, CHROMA hits the target
    lightness but flattens dark regions toward black.
    """

    DIRECTION = "direction"
    CHROMA = "chroma"
    CLIP = "clip"


_GAMUT_TOL = 1e-4


def _in_gamut(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose sRGB is within [0, 1] on every channel."""
    return (rgb >= -_GAMUT_TOL).all(-1) & (rgb <= 1 + _GAMUT_TOL).all(-1)


def chroma_to_gamut(luv: np.ndarray, iters: int = 20) -> np.ndarray:
    """Map CIELUV rows into the sRGB gamut and return sRGB in [0, 1].

    Clamps L into [0, 100], then for any color still outside the gamut scales
    (u, v) toward the neutral axis (binary search) until it re-enters, leaving
    hue and lightness intact. In-gamut colors are returned unchanged (aside from
    the round-trip). This is the `GamutMode.CHROMA` strategy, exposed for
    transforms (e.g. rotations) that always want chroma-preserving gamut mapping
    rather than naive per-channel clamping. See `GamutMode` for the trade-offs.
    """
    luv = np.asarray(luv, dtype=float).copy()
    luv[:, 0] = luv[:, 0].clip(0, 100)
    rgb = luv_to_rgb(luv)

    oog = ~_in_gamut(rgb)
    if oog.any():
        s = luv[oog]
        lo = np.zeros(int(oog.sum()))
        hi = np.ones(int(oog.sum()))
        for _ in range(iters):
            mid = (lo + hi) / 2
            trial = s.copy()
            trial[:, 1:] = s[:, 1:] * mid[:, None]
            ok = _in_gamut(luv_to_rgb(trial))
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        mapped = s.copy()
        mapped[:, 1:] = s[:, 1:] * lo[:, None]
        rgb[oog] = luv_to_rgb(mapped)

    return rgb.clip(0, 1)


def direction_to_gamut(start: np.ndarray, delta: np.ndarray, iters: int = 20) -> np.ndarray:
    """Translate CIELUV rows `start` by `delta` and return sRGB in [0, 1],
    walking any out-of-gamut result back along the shift vector.

    `start` is (N, 3) CIELUV assumed in gamut (t=0), `delta` is a (3,) L/u/v
    offset. For colors whose full shift `start + delta` leaves the gamut,
    binary-search the largest fraction t in [0, 1] with `start + t*delta` still
    displayable -- the point where the segment pierces the gamut boundary --
    preserving the shift direction (hue). In-gamut colors take the full shift.
    This is the `GamutMode.DIRECTION` strategy; see `GamutMode` for trade-offs.
    """
    start = np.asarray(start, dtype=float)
    delta = np.asarray(delta, dtype=float)
    rgb = luv_to_rgb(start + delta)

    oog = ~_in_gamut(rgb)
    if oog.any():
        s = start[oog]
        lo = np.zeros(int(oog.sum()))
        hi = np.ones(int(oog.sum()))
        for _ in range(iters):
            mid = (lo + hi) / 2
            ok = _in_gamut(luv_to_rgb(s + mid[:, None] * delta))
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        rgb[oog] = luv_to_rgb(s + lo[:, None] * delta)

    return rgb.clip(0, 1)


def _fit_to_gamut(
    start: np.ndarray, delta: np.ndarray, mode: GamutMode, iters: int = 20
) -> np.ndarray:
    """Translate CIELUV rows `start` by `delta` and return sRGB in [0, 1],
    resolving any out-of-gamut results per `mode` (see `GamutMode`).

    `start` is (N, 3) CIELUV (assumed in gamut), `delta` is a (3,) L/u/v offset.
    Dispatches to the per-mode helpers, each of which only runs its gamut search
    on the rows that actually leave the gamut.
    """
    if mode is GamutMode.CLIP:
        return luv_to_rgb(start + delta).clip(0, 1)
    if mode is GamutMode.DIRECTION:
        return direction_to_gamut(start, delta, iters)
    if mode is GamutMode.CHROMA:
        return chroma_to_gamut(start + delta, iters)
    raise ValueError(f"unknown gamut mode: {mode!r}")  # pragma: no cover


def shift_mountain(
    image: Image.Image, m: Mountain, to: Uv, gamut: GamutMode = GamutMode.DIRECTION
):
    """Translate every color inside mountain `m` so its peak lands on `to`.
    This is a conservative shift, as it only changes colors that are inside the mountain, and leaves all other colors untouched.
    If a more general shift is desired, use `shift_all` instead, which translates every color by the same amount.

    `gamut` controls how colors that shift outside the sRGB gamut are resolved; see `GamutMode`.
    """
    def filter_inside(coord):
        return (5 <= coord[0] <= 95) and (coord in m)

    shift_dist = np.asarray(to) - np.asarray(m.peak)

    return shift_to(image, shift_dist, filter_inside, gamut)

def dist_to(h: LuvHistogram, m: Mountain, to: Uv|Luv):
    if isinstance(to, Luv):
        # For Luv, we need the most common L for mountain's peak to get a full Luv shift vector,
        # since the mountain only encapsulates the uv.
        return np.asarray(to) - np.asarray(h.max_L(m.peak))
    else:
        return np.asarray(to) - np.asarray(m.peak)

def shift_all(image: Image.Image, dist: np.array, gamut: GamutMode = GamutMode.DIRECTION):
    """Translate every color by `dist` (either Luv or Uv)
    Use `dist_to` helper for computing the shift distance for typical cases (e.g. to move a mountain's peak to a target color).

    For a more conservative shift that only moves colors inside a mountain, use `shift_mountain` instead.

    Note: Leaves black/white colors untouched, as they are often not part of the main color palette and can be considered noise. This is done by filtering out colors with L < 5 or L > 95.
    These often don't shift into proper RGB colors anyway and would get clamped back to black/white.

    `gamut` controls how colors that shift outside the sRGB gamut are resolved; see `GamutMode`.
    """
    return shift_to(image, dist, lambda coord: 5 <= coord[0] <= 95, gamut)  # only shift colors with L in [5, 95]


def shift_to(
    image: Image.Image,
    dist: np.array,
    luv_filter: Callable[[np.array], bool]|None = None,
    gamut: GamutMode = GamutMode.DIRECTION,
):
    """Translate every color matching `filter` by `dist` (either Luv or Uv)

    The optional `filter` argument is a callable that takes a Luv coordinate and returns True if the color should be shifted, or False if it should be left alone.
    This allows for more selective shifting of colors.

    `gamut` (see `GamutMode`) controls how colors that shift outside the sRGB
    gamut are brought back. The default `DIRECTION` walks each out-of-gamut color
    back along the shift vector until it re-enters the gamut, preserving hue and
    avoiding the saturated confetti that naive per-channel clamping (`CLIP`)
    produces on dark images.
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
            # Build the full (L, u, v) offset. Uv mode leaves L untouched.
            if len(dist) == 2:  # Uv mode
                delta = np.array([0.0, dist[0], dist[1]], dtype=float)
            else:  # Luv mode
                delta = np.asarray(dist, dtype=float)

            rgb01 = _fit_to_gamut(luv[inside], delta, gamut)
            rgb = np.round(rgb01 * 255).astype(uniq.dtype)
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
    """Rotate every color by `deg` in the CIELUV u/v plane (leaving L fixed).

    Works per *distinct color* rather than per pixel: convert the unique RGB
    values to CIELUV once, rotate the whole batch, map back, and scatter to
    pixels via the `inverse` index. Fully transparent pixels and pure
    black/white are left untouched (they aren't part of the color palette).

    Out-of-gamut results are always resolved with `GamutMode.CHROMA`
    (`chroma_to_gamut`): L is preserved and chroma is reduced toward the neutral
    axis until the color is displayable. This avoids the saturated hue artifacts
    that naive per-channel clamping produces when a rotation lands outside sRGB.
    """
    ang = np.radians(deg)
    rot = np.array(
        [
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)],
        ]
    )

    arr = np.asarray(image)
    out = arr.copy()
    nch = arr.shape[2]  # 3 for RGB, 4 for RGBA
    flat = arr.reshape(-1, nch)
    out_flat = out.reshape(-1, nch)

    has_alpha = nch >= 4
    opaque = flat[:, 3] != 0 if has_alpha else np.ones(len(flat), dtype=bool)

    uniq, inverse = np.unique(flat[opaque, :3], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    new_uniq = uniq.copy()

    if len(uniq):
        # Leave pure black / pure white alone (treated as non-palette noise).
        keep = ~(np.all(uniq == 0, axis=1) | np.all(uniq == 255, axis=1))
        idx = np.where(keep)[0]
        if len(idx):
            luv = rgb_to_luv(uniq[idx].astype(float) / 255)
            luv[:, 1:] = luv[:, 1:] @ rot.T  # each row -> rot @ [u, v]
            rgb = np.round(chroma_to_gamut(luv) * 255).astype(uniq.dtype)
            new_uniq[idx] = rgb

    out_flat[opaque, :3] = new_uniq[inverse]
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
