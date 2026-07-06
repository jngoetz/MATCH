from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from colorspace import sRGB
from colorspace.colorlib import CIELUV
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL.Image import Image
from scipy import ndimage as nd

from mountain import Mountain, MountainParams, default_params

debug_enabled = False


def rgb_to_luv(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> CIELUV (D65). Accepts any (..., 3) shape."""
    a = np.asarray(rgb, dtype=float)
    flat = a.reshape(-1, 3)
    c = sRGB(flat[:, 0].copy(), flat[:, 1].copy(), flat[:, 2].copy())
    c.to("CIELUV")
    d = c.get()
    out = np.stack([np.asarray(d["L"]), np.asarray(d["U"]), np.asarray(d["V"])], axis=1)
    return out.reshape(a.shape)


def luv_to_rgb(luv: np.ndarray) -> np.ndarray:
    """CIELUV (D65) -> sRGB in [0, 1] (unclamped). Accepts any (..., 3) shape."""
    a = np.asarray(luv, dtype=float)
    flat = a.reshape(-1, 3)
    c = CIELUV(flat[:, 0].copy(), flat[:, 1].copy(), flat[:, 2].copy())
    c.to("sRGB")
    d = c.get()
    out = np.stack([np.asarray(d["R"]), np.asarray(d["G"]), np.asarray(d["B"])], axis=1)
    return out.reshape(a.shape)


class Luv(NamedTuple):
    L: float  # 0 to 100
    u: float  # -134 to 220
    v: float  # -140 to 122

    @classmethod
    def from_rgb(cls, pixel: Sequence[int]):
        L, u, v = rgb_to_luv(np.asarray(pixel[:3], dtype=float) / 255)
        return cls(
            np.clip(L, 0, 100),
            np.clip(u, -134, 220),
            np.clip(v, -140, 122),
        )

    @classmethod
    def from_hex(cls, hex: str):
        if hex.startswith("#"):
            hex = hex[1:]
        rgb = np.array([int(hex[i : i + 2], 16) for i in (0, 2, 4)])
        return cls.from_rgb(rgb)

    def to_rgb(self):
        return luv_to_rgb([self.L, self.u, self.v]).clip(0, 1)

    def hex(self):
        rgb = self.to_rgb()
        rgb = (rgb * 255).clip(0, 255).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def distance(self, other: "Luv") -> float:
        return np.linalg.norm(np.array(self) - np.array(other))


class Uv(NamedTuple):
    u: int  # -134 to 220
    v: int  # -140 to 122


class Histogram(dict[int, int]):
    def __missing__(self, key):
        return 0

    def plot(self, ax: Axes, L: int = 100):
        x = list(self.keys())
        # Color each bar by its Luv value. Build the whole (N, 3) Luv array and
        # convert in one batched call rather than one conversion per bar. Which
        # axis the bar keys (x) map to depends on what this slice holds fixed.
        xa = np.asarray(x, dtype=float)
        n = len(xa)
        if n == 0:
            c = None
        elif hasattr(self, "u") and hasattr(self, "v"):
            c = luv_to_rgb(np.column_stack([xa, np.full(n, self.u), np.full(n, self.v)]))
        elif hasattr(self, "u"):
            c = luv_to_rgb(np.column_stack([np.full(n, L), np.full(n, self.u), xa]))
        elif hasattr(self, "v"):
            c = luv_to_rgb(np.column_stack([np.full(n, L), xa, np.full(n, self.v)]))
        else:
            c = None
        if c is not None:
            c = c.clip(0, 1)

        ax.bar(
            x=x,
            height=[self[k] for k in x],
            width=1,
            color=c,
        )

    def peak(self) -> Tuple[int, float]:
        return max(self.items(), key=lambda x: x[1])

    def mountain(
        self, params: MountainParams | None = None, p2: int | None = None
    ) -> tuple[Mountain, List[int]]:
        """Returns the mountain with the highest peak in the histogram."""
        if params is None:
            params = default_params
        if p2 is None:
            p2, _ = self.peak()
        keys = list(range(min(self.keys()), max(self.keys()) + 1))
        values = [self[k] for k in keys]

        try:
            # v3
            for i in range(keys.index(p2) + 1, len(keys)):
                if values[i] > values[i - 1]:
                    v3 = keys[i - 1]
                    break
            else:
                v3 = keys[-1]
            # p3
            for i in range(keys.index(v3) + 1, len(keys)):
                if values[i] < values[i - 1]:
                    p3 = keys[i - 1]
                    break
            else:
                p3 = keys[-1]
            # v4
            for i in range(keys.index(p3) + 1, len(keys)):
                if values[i] > values[i - 1]:
                    v4 = keys[i - 1]
                    break
            else:
                v4 = keys[-1]

            # v2
            for i in range(keys.index(p2) - 1, -1, -1):
                if values[i] > values[i + 1]:
                    v2 = keys[i + 1]
                    break
            else:
                v2 = keys[0]
            # p1
            for i in range(keys.index(v2) - 1, -1, -1):
                if values[i] < values[i + 1]:
                    p1 = keys[i + 1]
                    break
            else:
                p1 = keys[0]
            # v1
            for i in range(keys.index(p1) - 1, -1, -1):
                if values[i] > values[i + 1]:
                    v1 = keys[i + 1]
                    break
            else:
                v1 = keys[0]
        except ValueError as e:
            raise ValueError(f"Could not find point in keys: {keys}") from e

        upper = v3
        if params.is_type1(self[v3], (self[p2], self[p3])) or params.is_type2(
            self[v3], self[p2], self[p3], self[v4]
        ):
            upper = v4

        lower = v2
        if params.is_type1(self[v2], (self[p1], self[p2])) or params.is_type2(
            self[v2], self[p1], self[p2], self[v1]
        ):
            lower = v1

        return (lower, p2, upper), [v1, p1, v2, p2, v3, p3, v4]


class UvHistogram(dict[Uv, int]):
    def __init__(self, luv_histo: "LuvHistogram" = None):
        super().__init__()
        self.luv_histo = luv_histo

    def __missing__(self, key):
        return 0

    def total(self):
        return sum(self.values())

    def copy(self):
        uv = UvHistogram(self.luv_histo)
        uv.update(self)
        return uv

    def plot(self, ax: Axes, Ls: dict[Uv, int] = None):
        if Ls is None:
            Ls = self.luv_histo.most_common_Ls()

        items = list(self.items())
        u = [iu for (iu, iv), _ in items]
        v = [iv for (iu, iv), _ in items]
        count = [icount for _, icount in items]
        # Look up each bar's L, then convert all Luv values to RGB in one call.
        L = [Ls.get(Uv(iu, iv), 100) for (iu, iv), _ in items]
        colors = luv_to_rgb(np.column_stack([L, u, v])).clip(0, 1) if items else None

        bars = ax.bar3d(
            x=u,
            y=v,
            z=0,
            dx=1,
            dy=1,
            dz=count,
            color=colors,
        )
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        return bars

    def slice_v(self, target_u) -> Histogram:
        """Returns a histogram of v values for a given u value."""
        h = Histogram()
        for (u, v), count in self.items():
            if u == target_u:
                h[v] += count
        h.u = target_u
        return h

    def slice_u(self, target_v):
        """Returns a histogram of u values for a given v value."""
        h = Histogram()
        for (u, v), count in self.items():
            if v == target_v:
                h[u] += count
        h.v = target_v
        return h

    def peak(self) -> Tuple[Uv, float]:
        return max(self.items(), key=lambda x: x[1])

    def mountain(self, params: MountainParams | None = None, peak: Uv | None = None):
        if params is None:
            params = default_params
        if peak is None:
            peak, _ = self.peak()

        u = self.slice_u(peak[1])
        try:
            mtn_u = u.mountain(params, peak[0])[0]
        except ValueError as e:
            raise ValueError(f"Could not find u cluster for peak {peak}") from e

        v = self.slice_v(peak[0])
        try:
            mtn_v = v.mountain(params, peak[1])[0]
        except ValueError as e:
            raise ValueError(f"Could not find v cluster for peak {peak}") from e

        return Mountain(mtn_u, mtn_v)

    def without_mountain(self, mtn: Mountain) -> "UvHistogram":
        h = UvHistogram(self.luv_histo)
        for k, v in list(self.items()):
            if k not in mtn:
                h[k] = v
        return h

    def primary_mountain(self, params: MountainParams | None = None):
        if params is None:
            params = default_params

        mtns = self.mountains(count=2)
        if len(mtns) == 0:
            raise ValueError("No mountains found")

        m = mtns.pop(0)
        if (0, 0) not in m:
            return m
        elif len(mtns) == 0:
            raise ValueError("Only white mountain found")
        else:
            return mtns.pop(0)

    def mountains(self, params: MountainParams | None = None, count=-1):
        if params is None:
            params = default_params

        # Pre-sort peaks by count (descending)
        biggest_peaks = sorted(self.items(), key=lambda x: x[1], reverse=True)

        # Use top 80% as candidates, but limit to reasonable number for performance
        num_candidates = min(len(biggest_peaks) - (len(biggest_peaks) // 5), 1000)
        candidates = biggest_peaks[:num_candidates]

        # Pre-compute slices to avoid repeated calculations
        u_slices = {}
        v_slices = {}

        def get_u_slice(target_v):
            if target_v not in u_slices:
                u_slices[target_v] = self.slice_u(target_v)
            return u_slices[target_v]

        def get_v_slice(target_u):
            if target_u not in v_slices:
                v_slices[target_u] = self.slice_v(target_u)
            return v_slices[target_u]

        def candidate_mountain(peak):
            u_hist = get_u_slice(peak[1])
            v_hist = get_v_slice(peak[0])
            try:
                um, u_pts = u_hist.mountain(params, peak[0])
                vm, v_pts = v_hist.mountain(params, peak[1])
                return Mountain(um, vm)
            except ValueError:
                return None

        # Use set for faster membership testing and removal
        remaining_candidates = set(range(len(candidates)))

        def remove_noncandidates(mtn):
            to_remove = []
            count = 0
            for i in list(remaining_candidates):
                peak = candidates[i][0]
                if peak in mtn:
                    count += 1
                    if count > 4:  # arbitrary threshold to reduce computation
                        to_remove.append(i)
            # Remove in reverse order to maintain indices
            for i in sorted(to_remove, reverse=True):
                remaining_candidates.discard(i)

        # Build candidate mountains more efficiently
        candidate_mountains = []
        processed_indices = []

        for i in list(remaining_candidates):
            peak, _ = candidates[i]
            m = candidate_mountain(peak)
            if m is not None and not m.has_zero_axis():
                size = self.mountain_size(m)
                candidate_mountains.append((m, size))
                processed_indices.append(i)
                remove_noncandidates(m)

        # Remove processed candidates
        for i in processed_indices:
            remaining_candidates.discard(i)

        candidate_mountains.sort(key=lambda x: x[1], reverse=True)

        mountains: list[Mountain] = []

        # Check overlap against accepted mountains
        def has_overlap(mtn):
            for existing_mtn in mountains:
                if existing_mtn.overlaps(mtn):
                    return True
            return False

        # Process candidate mountains efficiently
        for m, size in candidate_mountains:
            if count >= 0 and len(mountains) >= count:
                break
            if not has_overlap(m):
                mountains.append(m)

        return mountains

    def mountain_size(self, mtn: Mountain):
        """Returns the number of pixels in the image that are in the given mountain."""
        # Use caching to avoid recalculating for the same mountain
        if not hasattr(self, "_mountain_size_cache"):
            self._mountain_size_cache = {}

        # Create a hashable key for the mountain
        mtn_key = (mtn.u, mtn.v)
        if mtn_key in self._mountain_size_cache:
            return self._mountain_size_cache[mtn_key]

        items = self.items()
        if hasattr(self, "raw_histo"):
            items = self.raw_histo.items()

        size = sum(v for k, v in items if k in mtn)
        self._mountain_size_cache[mtn_key] = size
        return size

    def smooth(self, sigma=1) -> "UvHistogram":
        # Convert (L, u, v) -> count to an array of counts indexed by L, u, v
        min_indices = np.array(list(self.keys())).min(axis=0)
        max_indices = np.array(list(self.keys())).max(axis=0) + 1
        shape = tuple(np.array(max_indices) - np.array(min_indices))
        a = np.zeros(shape)
        for (u, v), count in self.items():
            a[u - min_indices[0], v - min_indices[1]] = count

        # Apply gaussian filter
        smoothed = nd.gaussian_filter(a, sigma=sigma)

        # Convert back to a histogram
        h = UvHistogram(self.luv_histo)
        for (u, v), count in np.ndenumerate(smoothed):
            if count > 0:
                h[
                    Uv(
                        u + min_indices[0],
                        v + min_indices[1],
                    )
                ] = count

        if hasattr(self, "raw_histo"):
            h.raw_histo = self.raw_histo

        return h

    def threshold(self, threshold: float) -> "UvHistogram":
        h = UvHistogram(self.luv_histo)
        for k, v in list(self.items()):
            if v > threshold:
                h[k] = v
        return h


class LuvHistogram(dict[Luv, int]):
    def __missing__(self, key):
        return 0

    def total(self):
        return sum(self.values())

    def plot(self, ax: Axes, **kwargs):
        # Keys are (L, u, v); convert them all to RGB in one batched call.
        luv = np.array(list(self.keys()), dtype=float).reshape(-1, 3)
        L, u, v = luv[:, 0], luv[:, 1], luv[:, 2]
        colors = luv_to_rgb(luv).clip(0, 1) if len(luv) else None

        if "s" not in kwargs:
            kwargs["s"] = 2

        ax.scatter(xs=u, ys=v, zs=L, c=colors, **kwargs)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("L")

    def collapse_L(self) -> UvHistogram:
        h = UvHistogram(self)
        for (_, u, v), count in self.items():
            if count > 0:
                h[Uv(u, v)] += count

        assert h.total() == self.total(), f"new {h.total()} != old {self.total()}"

        if hasattr(self, "raw_histo"):
            h.raw_histo = self.raw_histo.collapse_L()

        return h

    def Ls(self, uv: Uv):
        h = Histogram()
        for (L, u, v), count in self.items():
            if Uv(u, v) == uv:
                h[L] += count
        h.u = uv.u
        h.v = uv.v
        return h

    def most_common_Ls(self) -> dict[Uv, int]:
        bounds = {
            "u": {"min": 0, "max": 0},
            "v": {"min": 0, "max": 0},
        }
        counts = {}
        for luv, count in self.items():
            bounds["u"]["min"] = min(bounds["u"]["min"], luv.u)
            bounds["u"]["max"] = max(bounds["u"]["max"], luv.u)
            bounds["v"]["min"] = min(bounds["v"]["min"], luv.v)
            bounds["v"]["max"] = max(bounds["v"]["max"], luv.v)
            counts.setdefault((luv.u, luv.v), {})[luv.L] = count
        arr = (
            np.zeros(
                (
                    bounds["u"]["max"] - bounds["u"]["min"] + 1,
                    bounds["v"]["max"] - bounds["v"]["min"] + 1,
                )
            )
            * np.nan
        )
        for (u, v), c in counts.items():
            if len(c) == 0:
                continue
            max_L, count = max(
                c.items(),
                key=lambda x: x[1],
            )
            arr[u - bounds["u"]["min"], v - bounds["v"]["min"]] = max_L

        ind = nd.distance_transform_edt(
            np.isnan(arr), return_distances=False, return_indices=True
        )
        arr = arr[tuple(ind)]

        Ls = {}
        for u, vs in enumerate(arr):
            for v, L in enumerate(vs):
                if not np.isnan(L):
                    Ls[Uv(u + bounds["u"]["min"], v + bounds["v"]["min"])] = L
        return Ls

    def peak(self) -> Tuple[Luv, float]:
        return max(self.items(), key=lambda x: x[1])

    def without_white(self, threshold: float) -> "LuvHistogram":
        removed = 0

        h = LuvHistogram()
        for k, v in list(self.items()):
            dist = np.linalg.norm(k[1:])
            if dist > threshold:
                h[k] = v
            else:
                removed += v
        # print(f"Removed {removed} white pixels")
        h.raw_histo = self
        return h

    def without_mountain(self, mtn: Mountain) -> "LuvHistogram":
        h = LuvHistogram()
        for luv, v in list(self.items()):
            if Uv(*luv[1:]) not in mtn:
                h[luv] = v
        return h

    def smooth(self, sigma=1) -> "LuvHistogram":
        # Convert (L, u, v) -> count to an array of counts indexed by L, u, v
        min_indices = np.array(list(self.keys())).min(axis=0)
        max_indices = np.array(list(self.keys())).max(axis=0) + 1
        shape = tuple(np.array(max_indices) - np.array(min_indices))
        a = np.zeros(shape)
        for (L, u, v), count in self.items():
            a[L - min_indices[0], u - min_indices[1], v - min_indices[2]] = count

        # Apply gaussian filter
        smoothed = nd.gaussian_filter(a, sigma=sigma)

        # Convert back to a histogram
        h = LuvHistogram()
        for (L, u, v), count in np.ndenumerate(smoothed):
            if count > 0:
                h[
                    Luv(
                        L + min_indices[0],
                        u + min_indices[1],
                        v + min_indices[2],
                    )
                ] = count
        return h

    @classmethod
    def from_image(cls, image: Image) -> "LuvHistogram":
        h = cls()

        arr = np.asarray(image.convert("RGBA")).reshape(-1, 4)
        keep = arr[:, 3] >= 128  # alpha/255 >= 0.5
        transparent = int((~keep).sum())

        if keep.any():
            luv = rgb_to_luv(arr[keep, :3].astype(float) / 255)
            coords = np.round(luv).astype(int)
            # Count identical Luv coordinates in one pass instead of per pixel.
            uniq, counts = np.unique(coords, axis=0, return_counts=True)
            for (L, u, v), count in zip(uniq, counts):
                h[Luv(int(L), int(u), int(v))] = int(count)

        print(
            f"{getattr(image, 'filename', 'unknown')}: Loaded {h.total()}; skipped {transparent} transparent pixels. {len(image.getdata())} total; {image.width * image.height} [{image.width}x{image.height}]"
        )
        return h

    def max_L(self, uv: Uv) -> Luv:
        """Returns the L value with the highest count for the given u,v."""
        ls = [(k, v) for k, v in self.items() if k[1:] == uv]
        if len(ls) == 0:
            ls = [
                (k, v)
                for k, v in self.items()
                if k[1] >= uv[0] - 1
                and k[1] <= uv[0] + 1
                and k[2] >= uv[1] - 1
                and k[2] <= uv[1] + 1
            ]
            if len(ls) == 0:
                raise ValueError(f"No L values found for {uv}")
        return max(ls, key=lambda x: x[1])[0]

    def chi_square(self, other: "LuvHistogram") -> float:
        """Compute the chi-square statistic between this histogram and another."""
        keys = set(self.keys()) | set(other.keys())
        chi2 = 0.0
        for k in keys:
            o = self[k]
            e = other[k]
            if o + e > 0:
                chi2 += (o - e) ** 2 / (o + e)
        return chi2


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt

    from image_util import CacheMode, image_histo
    from mountain import default_params
    from plots import uv_Ls

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--mountains", type=int, default=2)
    parser.add_argument("--linecolor", type=str, default="#D9AA41")
    parser.add_argument("--smooth", type=int, default=1)

    args = parser.parse_args()
    fpath = Path(args.image)
    outpath = Path(".") / fpath.name

    resized_name = fpath.with_name(
        fpath.stem + "_160.png"
    )  # pretend it's resized to check for the histo
    histo = image_histo(resized_name, cache_mode=CacheMode.CACHE_ONLY)

    h_fig = plt.figure(figsize=(9, 9))
    h_ax = h_fig.add_subplot(111, projection="3d")

    uv = histo.collapse_L()

    mtns = uv.smooth(1).mountains(default_params, args.mountains + 1)
    white = next((m for m in mtns if (0, 0) in m), None)
    mtns = [m for m in mtns if m != white][: args.mountains]

    if white is not None:
        uv = uv.without_mountain(white)
    uv = uv.smooth(args.smooth).threshold(0.2)

    height = max(uv.values()) * 1.1

    Ls = uv_Ls(histo, uv)
    bars = uv.plot(h_ax, Ls=Ls)

    markers, stems, base = h_ax.stem(
        x=[m.peak[0] for m in mtns],
        y=[m.peak[1] for m in mtns],
        z=np.ones(len(mtns)) * height,
    )
    markers.set_color("black")  # markers alway show against the white/grey background
    stems.set_color(args.linecolor)
    stems.set_linestyle("dashed")
    stems.set_zorder(1000)
    base.set_visible(False)

    h_fig.tight_layout()
    h_fig.savefig(outpath.with_name(outpath.stem + "_histo_mtns.png"))
