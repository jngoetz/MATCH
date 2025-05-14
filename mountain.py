from typing import NamedTuple, Tuple

import numpy as np
from matplotlib.axes import Axes


class MountainParams(NamedTuple):
    t: float
    alpha: float
    beta: float

    def is_type1(self, valley, peaks):
        """Returns whether the valley is a type 1 valley. Order of peaks does not matter.
        In the Tominaga paper example, valley=v2, peaks=(p2, p1)
        """
        if valley == 0:
            return False
        return (peaks[0] / valley) > self.t and (peaks[1] / valley) > self.t

    def is_type2(self, valley, center_peak, side_peak, next_valley):
        """Returns whether the valley is a type 2 valley.
        In the Tominaga paper example, valley=v3, center_peak=p2, side_peak=p3, next_valley=v4
        """
        if side_peak == valley:
            return False
        return (center_peak - valley) / (side_peak - valley) > self.alpha and (
            side_peak - next_valley
        ) / (side_peak - valley) > self.beta


default_params = MountainParams(t=1.5, alpha=10, beta=10)


class Mountain(NamedTuple):
    u: Tuple[int, int, int]
    v: Tuple[int, int, int]

    @property
    def lower(self):
        return (self.u[0], self.v[0])

    @property
    def peak(self):
        return (self.u[1], self.v[1])

    @property
    def upper(self):
        return (self.u[2], self.v[2])

    @property
    def coordinates(self):
        return (
            np.asarray([self.u[0], self.u[1], self.u[2], self.u[1]]),
            np.asarray([self.v[1], self.v[0], self.v[1], self.v[2]]),
        )

    def axes_for_point(self, point):
        u, v = point[:2]

        # subtract the peak coordinates to get it centered
        u -= self.u[1]
        v -= self.v[1]

        # determine which radius to use based on the quadrant
        rx = self.u[0] if u < 0 else self.u[2]
        ry = self.v[0] if v < 0 else self.v[2]

        rx -= self.u[1]
        ry -= self.v[1]

        return abs(rx), abs(ry)

    def __contains__(self, point):
        u, v = point[-2:]

        # subtract the peak coordinates to get it centered
        u -= self.u[1]
        v -= self.v[1]

        # determine which radius to use based on the quadrant
        rx = self.u[0] if u < 0 else self.u[2]
        ry = self.v[0] if v < 0 else self.v[2]

        # convert the point to the radius, no need to correct the subtraction,
        # since we're squaring it anyway
        rx -= self.u[1]
        ry -= self.v[1]

        if rx == 0 and u != 0:
            return False
        if ry == 0 and v != 0:
            return False

        x = (u**2) / (rx**2) if rx != 0 else 0
        y = (v**2) / (ry**2) if ry != 0 else 0

        # ellipse function
        return x + y <= 1

    def edge_towards(self, point):
        u, v = point[-2:]
        rx, ry = self.axes_for_point(point)

        if rx == 0 or ry == 0:
            raise ValueError(f"axes are zero: {rx}, {ry}")

        u -= self.u[1]
        v -= self.v[1]
        phi = np.arctan2(v, u)
        if np.cos(phi) == 0:
            # tangent is undefined, so we're on one of the axes
            return (0, self.v[2]) if np.sin(phi) > 0 else (0, self.v[0])

        edge_u = (rx * ry) / np.sqrt(ry**2 + (rx**2) * (np.tan(phi) ** 2))
        if np.cos(phi) < 0:
            # since u can be +/-, the initial sign of edge_u is positive
            # then we flip it here if -pi/2 < phi < pi/2
            edge_u *= -1
        edge_v = edge_u * np.tan(phi)
        return edge_u + self.u[1], edge_v + self.v[1]

    def overlaps(self, other: "Mountain") -> bool:
        """Returns whether this mountain overlaps with another mountain."""
        edge = self.edge_towards(other.peak)
        return edge in other

    def ellipse(self):
        # distances from the peak
        u_off = np.abs(np.array(self.u) - self.peak[0])
        v_off = np.abs(np.array(self.v) - self.peak[1])

        e_u, e_v = [], []
        ang = np.linspace(0, 2 * np.pi, 100)
        for ang, cos, sin in zip(ang, np.cos(ang), np.sin(ang)):
            x = u_off[2] if cos > 0 else u_off[0]
            y = v_off[2] if sin > 0 else v_off[0]
            e_u.append(self.peak[0] + x * cos)
            e_v.append(self.peak[1] + y * sin)
        return e_u, e_v

    def plot(
        self,
        ax: Axes,
        ellipse_height: int = 0,
        ellipse_color: str = "black",
    ) -> tuple:
        e_u, e_v = self.ellipse()
        ax.plot(
            xs=e_u,
            ys=e_v,
            zs=ellipse_height,
            color=ellipse_color,
            zorder=3,
            linewidth=1,
        )
