from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import LuvColor, sRGBColor
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL.Image import Image
from scipy import ndimage as nd

from mountain import Mountain, MountainParams, default_params

debug_enabled = False


class Luv(NamedTuple):
    L: int  # 0 to 100
    u: int  # -134 to 220
    v: int  # -140 to 122

    @classmethod
    def from_rgb(cls, pixel: Sequence[int]):
        luv = convert_color(
            sRGBColor(*pixel[:3], is_upscaled=True),
            LuvColor,
            target_illuminant="d65",
        )
        return cls(
            int(np.clip(np.round(luv.luv_l), 0, 100)),
            int(np.clip(np.round(luv.luv_u), -134, 220)),
            int(np.clip(np.round(luv.luv_v), -140, 122)),
        )

    @classmethod
    def from_hex(cls, hex: str):
        if hex.startswith("#"):
            hex = hex[1:]
        rgb = np.array([int(hex[i : i + 2], 16) for i in (0, 2, 4)])
        return cls.from_rgb(rgb)

    def to_rgb(self):
        return np.array(
            convert_color(
                LuvColor(self.L, self.u, self.v),
                sRGBColor,
            ).get_value_tuple()
        ).clip(0, 1)

    def hex(self):
        rgb = self.to_rgb()
        rgb = (rgb * 255).clip(0, 255).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class Uv(NamedTuple):
    u: int  # -134 to 220
    v: int  # -140 to 122


class Histogram(dict[int, int]):
    def __missing__(self, key):
        return 0

    def plot(self, ax: Axes, L: int = 100):
        x = list(self.keys())
        if hasattr(self, "u") and hasattr(self, "v"):
            c = [Luv(L, self.u, self.v).to_rgb() for L in x]
        elif hasattr(self, "u"):
            c = [Luv(L, self.u, v).to_rgb() for v in x]
        elif hasattr(self, "v"):
            c = [Luv(L, u, self.v).to_rgb() for u in x]
        else:
            c = None

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
        keys = sorted(self.keys())
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

    def plot(self, ax: Axes, Ls: dict[Uv, int]):
        u, v, count = [], [], []
        colours = []
        for (iu, iv), icount in self.items():
            u.append(iu)
            v.append(iv)
            count.append(icount)
            rgb = convert_color(LuvColor(Ls.get(Uv(iu, iv), 100), iu, iv), sRGBColor)
            colours.append(np.asarray(rgb.get_value_tuple()).clip(0, 1))

        bars = ax.bar3d(
            x=u,
            y=v,
            z=0,
            dx=1,
            dy=1,
            dz=count,
            color=colours,
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
        candidates = sorted(self.items(), key=lambda x: x[1], reverse=True)
        mountains: list[Mountain] = []

        if params is None:
            params = default_params

        def contains(point):
            for m in mountains:
                if point in m:
                    return True
            return False

        def overlaps(mtn):
            for m in mountains:
                if m.overlaps(mtn):
                    return True
            return False

        while len(candidates) > 0 and (count < 0 or len(mountains) < count):
            peak, _ = candidates.pop(0)
            if contains(peak):
                # fast check, because finding the boundary of the candidate
                continue
            u = self.slice_u(peak[1])
            v = self.slice_v(peak[0])

            um, u_pts = u.mountain(params, peak[0])
            vm, v_pts = v.mountain(params, peak[1])
            m = Mountain(um, vm)

            if m.u[1] in (m.u[0], m.u[2]) or m.v[1] in (m.v[0], m.v[2]):
                #     # one of the axes has a 0-radius, so it's not a valid mountain
                continue
            if overlaps(m):
                # don't let mountains overlap
                continue

            mountains.append(m)
            if debug_enabled:
                print(f"{m} : {self.mountain_size(m)}")
                fig, (ax_u, ax_v, ax_L) = plt.subplots(3)
                fig.suptitle(f"Peak: {peak}")
                fig.text(
                    0.5,
                    0.9,
                    f"height: {int(self[peak])}; size: {int(self.mountain_size(m))}",
                    ha="center",
                )
                c = [
                    "purple",
                    "orange",
                    "purple",
                    "black",
                    "purple",
                    "orange",
                    "purple",
                ]
                c2 = ["purple", "black", "purple"]

                u.plot(ax_u)
                ax_u.vlines(
                    u_pts, ymin=0, ymax=max(u.values()), colors=c, linestyles="dashed"
                )
                ax_u.vlines(m.u, ymin=0, ymax=max(u.values()), colors=c2)
                ax_u.set_xlabel("u")

                v.plot(ax_v)
                ax_v.vlines(
                    v_pts, ymin=0, ymax=max(v.values()), colors=c, linestyles="dashed"
                )
                ax_v.vlines(m.v, ymin=0, ymax=max(v.values()), colors=c2)
                ax_v.set_xlabel("v")

                L = self.luv_histo.Ls(peak)
                L.plot(ax_L)
                ax_L.set_xlabel("L")

                fig.savefig(f"debug/mountain_{len(mountains)}.png")
                plt.close(fig)

        return mountains

    def mountain_size(self, mtn: Mountain):
        """Returns the number of pixels in the image that are in the given mountain."""
        return sum(v for k, v in self.items() if k in mtn)

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

    def plot(self, ax: Axes):
        L, u, v = [], [], []
        colours = []
        for iL, iu, iv in self.keys():
            L.append(iL)
            u.append(iu)
            v.append(iv)
            rgb = convert_color(LuvColor(iL, iu, iv), sRGBColor)
            colours.append(np.asarray(rgb.get_value_tuple()).clip(0, 1))

        ax.scatter(
            xs=u,
            ys=v,
            zs=L,
            c=colours,
            s=1,
        )
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("L")

    def collapse_L(self) -> UvHistogram:
        h = UvHistogram(self)
        for (_, u, v), count in self.items():
            if count > 0:
                h[Uv(u, v)] += count

        assert h.total() == self.total(), f"new {h.total()} != old {self.total()}"

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
        print(f"Removed {removed} white pixels")
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
        tranparent = 0
        for pixel in image.getdata():
            if len(pixel) == 4 and pixel[3] == 0:
                # skip transparent pixels
                tranparent += 1
                continue
            luv = Luv.from_rgb(pixel)
            h[luv] += 1
        print(f"Skipped {tranparent} transparent pixels")
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
