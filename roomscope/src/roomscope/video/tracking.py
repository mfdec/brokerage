"""Following the coloured marker on the microphone: image blob -> room position (x, y) at mic height."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from roomscope.core.geometry import RoomCamera
from roomscope.core.modes import RoomDims


@dataclass
class Detection:
    u: float
    v: float
    area: float            # in full-resolution pixels
    circularity: float


class ColorMarkerTracker:
    """HSV colour-blob tracker.

    Pick the marker colour by clicking it; the tracker then follows the most plausible blob of that colour
    (compact, of sensible size, near where the marker was a moment ago).  Matte, saturated colours that do not
    occur elsewhere in the room work best (a hot-pink or lime-green ball / sticker on the mic stand).
    """

    def __init__(self, hue_tol: int = 12, sat_min: int = 70, val_min: int = 50, min_area_px: float = 5.0,
                 max_area_frac: float = 0.02, work_width: int = 640, adapt: bool = False):
        self.hue_tol, self.sat_min, self.val_min = hue_tol, sat_min, val_min
        self.min_area_px, self.max_area_frac, self.work_width = min_area_px, max_area_frac, work_width
        self.adapt = adapt
        self.target: np.ndarray | None = None            # HSV (OpenCV: H 0..179)
        self.last: tuple[float, float] | None = None
        self.lost = 0
        self.last_mask: np.ndarray | None = None

    # ------------------------------------------------------------------ target
    def set_target_bgr(self, bgr) -> None:
        self.target = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float64)
        self.last, self.lost = None, 0

    def set_target_from_click(self, frame: np.ndarray, u: float, v: float, radius: int = 4) -> np.ndarray:
        """Take the marker colour from a small patch around the click (median of the saturated pixels)."""
        h, w = frame.shape[:2]
        x0, x1 = int(max(u - radius, 0)), int(min(u + radius + 1, w))
        y0, y1 = int(max(v - radius, 0)), int(min(v + radius + 1, h))
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        keep = patch[:, 1] >= np.percentile(patch[:, 1], 40)              # ignore washed-out highlight pixels
        hsv = np.median(patch[keep], axis=0)
        # hue is circular: use a circular mean over the kept pixels
        ang = patch[keep, 0] * (2 * np.pi / 180.0)
        hsv[0] = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
        self.target = hsv
        self.last, self.lost = (float(u), float(v)), 0
        return hsv

    # ------------------------------------------------------------------ detection
    def _ranges(self):
        h, s, v = self.target
        s_lo = int(max(self.sat_min, 0.45 * s))
        v_lo = int(max(self.val_min, 0.40 * v))
        lo, hi = int(round(h - self.hue_tol)), int(round(h + self.hue_tol))
        if lo < 0:
            hs = [(0, hi), (180 + lo, 179)]
        elif hi > 179:
            hs = [(lo, 179), (0, hi - 180)]
        else:
            hs = [(lo, hi)]
        return [((a, s_lo, v_lo), (b, 255, 255)) for a, b in hs]

    def mask(self, small_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(cv2.GaussianBlur(small_bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self._ranges():
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        # odd, symmetric kernels only: an even kernel (2x2) has an off-centre anchor and shifts the mask by a pixel
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cross)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    def detect(self, frame: np.ndarray) -> Detection | None:
        if self.target is None:
            return None
        h, w = frame.shape[:2]
        scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        m = self.mask(small)
        self.last_mask = m
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        gate = None
        if self.last is not None and self.lost < 20:
            gate = 70.0 * (1.0 + 0.6 * self.lost) * scale                  # grows while the marker is missing
        max_area = self.max_area_frac * small.shape[0] * small.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area_px * scale * scale <= area <= max_area):
                continue
            per = cv2.arcLength(c, True)
            circ = float(4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
            if circ < 0.35:
                continue
            mo = cv2.moments(c)
            if mo["m00"] <= 0:
                continue
            cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
            if gate is not None and np.hypot(cx - self.last[0] * scale, cy - self.last[1] * scale) > gate:
                continue
            score = area * circ
            if score > best_score:
                best, best_score = (cx, cy, area, circ), score
        if best is None:
            self.lost += 1
            return None
        cx, cy, area, circ = best
        det = Detection(cx / scale, cy / scale, area / (scale * scale), circ)
        self.last, self.lost = (det.u, det.v), 0
        if self.adapt:
            self._adapt(small, m, (int(round(cx)), int(round(cy))))
        return det

    def _adapt(self, small: np.ndarray, mask: np.ndarray, c: tuple[int, int], rate: float = 0.04) -> None:
        r = 3
        x0, y0 = max(c[0] - r, 0), max(c[1] - r, 0)
        patch = cv2.cvtColor(small[y0:c[1] + r + 1, x0:c[0] + r + 1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        sel = mask[y0:c[1] + r + 1, x0:c[0] + r + 1].reshape(-1) > 0
        if sel.sum() < 4:
            return
        hsv = np.median(patch[sel], axis=0)
        dh = ((hsv[0] - self.target[0] + 90.0) % 180.0) - 90.0
        self.target[0] = (self.target[0] + rate * dh) % 180.0
        self.target[1:] += rate * (hsv[1:] - self.target[1:])


class PositionFilter:
    """Constant-velocity Kalman filter in room coordinates with outlier gating."""

    def __init__(self, accel_sigma: float = 0.9, meas_sigma: float = 0.07, gate: float = 4.5):
        self.qa, self.r, self.gate = accel_sigma, meas_sigma, gate
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None
        self.rejected = 0
        self.jumps = 0                                              # persistent jumps accepted (track re-seeded)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3])) if self.x is not None else 0.0

    def update(self, t: float, z: np.ndarray | None) -> np.ndarray | None:
        if self.x is None:
            if z is None:
                return None
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.r ** 2, self.r ** 2, 1.0, 1.0])
            self.t = t
            return self.x
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.qa ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if z is None:
            return self.x
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = Hm @ self.P @ Hm.T + np.eye(2) * self.r ** 2
        innov = np.asarray(z) - Hm @ self.x
        m2 = float(innov @ np.linalg.solve(S, innov))
        if m2 > self.gate ** 2:                                     # a jump no walker can make: a false detection...
            if self.rejected < 4:
                self.rejected += 1
                return self.x
            # ...unless it persists: the marker really is somewhere else now.  Start the track afresh there rather than
            # letting a Kalman update drag a huge fake velocity through the state.
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.r ** 2, self.r ** 2, 1.0, 1.0])
            self.rejected = 0
            self.jumps += 1
            return self.x
        self.rejected = 0
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ Hm) @ self.P
        return self.x


@dataclass
class TrackState:
    t: float
    valid: bool
    xy: tuple[float, float] | None
    speed: float
    px: tuple[float, float] | None
    detected: bool
    lost_s: float


class MarkerPipeline:
    """Frame -> blob -> room coordinates at ``marker_height`` -> filtered position and speed."""

    def __init__(self, tracker: ColorMarkerTracker | None = None, dims: RoomDims | None = None,
                 marker_height: float = 1.2, margin: float = 0.7, valid_hold_s: float = 0.5):
        self.tracker = tracker or ColorMarkerTracker()
        self.filter = PositionFilter()
        self.cam: RoomCamera | None = None
        self.dims = dims
        self.z = marker_height
        self.margin = margin
        self.valid_hold_s = valid_hold_s
        self.trail: deque = deque(maxlen=600)         # (t, x, y); a NaN row separates unconnected pieces of path
        self._last_meas_t: float | None = None
        self._trail_break = True
        self._jumps = 0
        self.state = TrackState(0.0, False, None, 0.0, None, False, 1e9)

    def set_camera(self, cam: RoomCamera | None) -> None:
        self.cam = cam
        self.filter.reset()
        self._last_meas_t = None
        self._trail_break = True
        self._jumps = 0

    def process(self, frame: np.ndarray, t: float) -> TrackState:
        det = self.tracker.detect(frame)
        meas = None
        px = None
        if det is not None and self.cam is not None:
            px = (det.u, det.v)
            xy = self.cam.backproject(np.array([[det.u, det.v]]), self.z)[0]
            if np.all(np.isfinite(xy)) and (self.dims is None or (
                    -self.margin <= xy[0] <= self.dims.lx + self.margin and -self.margin <= xy[1] <= self.dims.ly + self.margin)):
                meas = xy
        x = self.filter.update(t, meas)
        if meas is not None:
            self._last_meas_t = t
        lost = (t - self._last_meas_t) if self._last_meas_t is not None else 1e9
        valid = x is not None and lost <= self.valid_hold_s
        if lost > 2.0:
            self.filter.reset()                         # lost for good: re-acquire cleanly
            x, valid = None, False
        xy_out = (float(x[0]), float(x[1])) if x is not None else None
        self.state = TrackState(t, bool(valid), xy_out, self.filter.speed, px, det is not None, float(lost))
        if not valid or self.filter.jumps != self._jumps:
            self._trail_break = True                    # lost, or re-seeded elsewhere: do not draw a line across the gap
            self._jumps = self.filter.jumps
        if valid and xy_out is not None:
            if self._trail_break and self.trail:
                self.trail.append((t, np.nan, np.nan))
            self._trail_break = False
            self.trail.append((t, *xy_out))
        return self.state
