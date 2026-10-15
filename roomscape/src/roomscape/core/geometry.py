"""Camera geometry for mapping a room from video.

Room coordinates: x along the length, y along the width, z up (metres), origin at a floor corner.
The user pins >= 4 points on the *floor* (known room coordinates, clicked in the image); a pinhole
camera with a guessed field of view then gives the full pose.  With the pose we can

* draw a map lying on any horizontal plane (e.g. at ear height) into the video, and
* convert the pixel position of the microphone marker back to room coordinates at the marker's height
  (a plain floor homography would be off by metres for something held at 1 m).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fov_h_deg: float = 65.0

    @property
    def focal(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.fov_h_deg) / 2.0)

    @property
    def K(self) -> np.ndarray:
        f = self.focal
        return np.array([[f, 0.0, self.width / 2.0], [0.0, f, self.height / 2.0], [0.0, 0.0, 1.0]])

    def with_fov(self, fov_h_deg: float) -> "Intrinsics":
        return Intrinsics(self.width, self.height, float(fov_h_deg))

    @staticmethod
    def fov_from_focal(width: int, focal: float) -> float:
        return float(np.degrees(2.0 * np.arctan((width / 2.0) / focal)))


@dataclass
class RoomCamera:
    K: np.ndarray
    rvec: np.ndarray          # (3,) Rodrigues, world -> camera
    tvec: np.ndarray          # (3,)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))[0]

    @property
    def center(self) -> np.ndarray:
        """Camera position in room coordinates."""
        return -self.R.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)

    @classmethod
    def look_at(cls, K: np.ndarray, eye, target, up=(0.0, 0.0, 1.0)) -> "RoomCamera":
        eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack((right, down, fwd))                      # rows = camera axes in world coordinates
        rvec = cv2.Rodrigues(R)[0].reshape(3)
        return cls(np.asarray(K, dtype=np.float64), rvec, (-R @ eye).reshape(3))

    # ------------------------------------------------------------------ projection
    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return np.atleast_2d(pts) @ self.R.T + np.asarray(self.tvec).reshape(1, 3)

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depth (N,) of room points (N, 3); pixels are NaN for points behind the camera."""
        pc = self.to_camera(np.asarray(pts, dtype=np.float64))
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (pc[:, :2] / z[:, None]) * np.array([self.K[0, 0], self.K[1, 1]]) + self.K[:2, 2]
        uv[z <= 1e-6] = np.nan
        return uv, z

    def backproject(self, px: np.ndarray, z_plane: float) -> np.ndarray:
        """Room (x, y) where the pixel rays meet the horizontal plane ``z = z_plane`` (NaN if they miss it in front)."""
        px = np.atleast_2d(np.asarray(px, dtype=np.float64))
        rays_cam = np.column_stack(((px[:, 0] - self.K[0, 2]) / self.K[0, 0], (px[:, 1] - self.K[1, 2]) / self.K[1, 1],
                                    np.ones(px.shape[0])))
        d = rays_cam @ self.R                                 # R^T applied to each ray (row vectors)
        c = self.center
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z_plane - c[2]) / d[:, 2]
        pts = c[None, :] + s[:, None] * d
        out = pts[:, :2].copy()
        out[~(s > 0)] = np.nan
        return out

    def plane_homography(self, z_plane: float) -> np.ndarray:
        """3x3 H with pixel ~ H @ (x, y, 1) for points on the plane ``z = z_plane``."""
        R, t = self.R, np.asarray(self.tvec, dtype=np.float64).reshape(3)
        return self.K @ np.column_stack((R[:, 0], R[:, 1], z_plane * R[:, 2] + t))

    def visible_mask(self, width: int, height: int, z_plane: float, step: int = 1) -> np.ndarray:
        """Pixels whose ray meets the plane in front of the camera (below the horizon for a floor-like plane)."""
        u = np.arange(0, width, step) + 0.5 * (step - 1)
        v = np.arange(0, height, step) + 0.5 * (step - 1)
        # d_z of the ray through (u, v) is linear in (u, v): d = R^T K^-1 (u, v, 1)
        row = self.R[:, 2] @ np.linalg.inv(self.K)             # (a, b, c): d_z = a u + b v + c
        dz = row[0] * u[None, :] + row[1] * v[:, None] + row[2]
        s_sign = np.sign(z_plane - self.center[2])
        return dz * s_sign > 1e-9

    def reprojection_rms(self, world_pts: np.ndarray, img_pts: np.ndarray) -> float:
        uv, _ = self.project(world_pts)
        d = np.sum((uv - np.asarray(img_pts)) ** 2, axis=1)
        return float(np.sqrt(np.mean(d[np.isfinite(d)]))) if np.any(np.isfinite(d)) else float("inf")


def _check_spread(world_xy: np.ndarray) -> None:
    pts = world_xy - world_xy.mean(axis=0)
    sv = np.linalg.svd(pts, compute_uv=False)
    if sv[1] < 0.15 * sv[0] or sv[1] < 0.3:
        raise GeometryError("The pinned points lie nearly on a line; spread them over the floor (at least 4, not in a row).")


def fit_camera(img_pts, world_xy, intr: Intrinsics) -> tuple[RoomCamera, float]:
    """Camera pose from >= 4 floor points (pixels <-> room x,y at z = 0).  Returns (camera, reprojection RMS px)."""
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    wxy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    if img.shape[0] < 4 or img.shape[0] != wxy.shape[0]:
        raise GeometryError("Need at least 4 floor points with matching pixel and room coordinates.")
    _check_spread(wxy)
    obj = np.column_stack((wxy, np.zeros(len(wxy))))
    K = intr.K
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rv, tv in zip(rvecs, tvecs):
        cam = RoomCamera(K, np.asarray(rv, dtype=np.float64).reshape(3), np.asarray(tv, dtype=np.float64).reshape(3))
        _, depth = cam.project(obj)
        if cam.center[2] <= 0 or np.any(depth <= 0):       # camera must be above the floor, looking at the points
            continue
        rms = cam.reprojection_rms(obj, img)
        if best is None or rms < best[1]:
            best = (cam, rms)
    if best is None:
        raise GeometryError("No consistent camera pose: check the pinned points (order and positions).")
    cam, _ = best
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, cam.rvec.reshape(3, 1).copy(), cam.tvec.reshape(3, 1).copy())
    cam = RoomCamera(K, np.asarray(rv, dtype=np.float64).reshape(3), np.asarray(tv, dtype=np.float64).reshape(3))
    return cam, cam.reprojection_rms(obj, img)


def estimate_focal(img_pts, world_xy, width: int, height: int) -> float | None:
    """Focal length (px) from the floor homography (one-view planar self-calibration), or None if ill-conditioned.

    Needs a clearly oblique view (strong perspective); a camera looking straight down gives no information.
    """
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    wxy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
    if img.shape[0] < 4:
        return None
    H, _ = cv2.findHomography(wxy, img, 0)
    if H is None:
        return None
    T = np.array([[1, 0, -width / 2.0], [0, 1, -height / 2.0], [0, 0, 1]])
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    den1 = h1[2] * h2[2]
    den2 = h2[2] ** 2 - h1[2] ** 2
    if abs(den1) < 1e-12 or abs(den2) < 1e-12:
        return None
    f1 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den1
    f2 = (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den2
    if f1 <= 0 or f2 <= 0 or not (0.6 < f1 / f2 < 1.0 / 0.6):
        return None
    f = float(np.sqrt(np.sqrt(f1 * f2) * 1.0))
    fov = Intrinsics.fov_from_focal(width, f)
    return f if 20.0 <= fov <= 140.0 else None


def segment_in_front(cam: RoomCamera, p0, p1, near: float = 0.05):
    """Clip the 3D segment p0-p1 to the camera's near plane and project it; None if wholly behind."""
    a, b = cam.to_camera(np.asarray(p0, dtype=float))[0], cam.to_camera(np.asarray(p1, dtype=float))[0]
    if a[2] < near and b[2] < near:
        return None
    if a[2] < near:
        a = a + (b - a) * ((near - a[2]) / (b[2] - a[2]))
    elif b[2] < near:
        b = b + (a - b) * ((near - b[2]) / (a[2] - b[2]))
    f = np.array([cam.K[0, 0], cam.K[1, 1]])
    c = cam.K[:2, 2]
    return a[:2] / a[2] * f + c, b[:2] / b[2] * f + c


# --------------------------------------------------------------------------- #
# Pinning the field of view down (the one thing a single view cannot tell us)
# --------------------------------------------------------------------------- #
def _scan_fov(score, fov_range=(30.0, 130.0)) -> tuple[float, float] | None:
    """Minimise ``score(fov)`` (None = infeasible) with a coarse then a fine scan."""
    best = None
    for fov in np.arange(fov_range[0], fov_range[1] + 0.01, 1.0):
        v = score(float(fov))
        if v is not None and (best is None or v < best[1]):
            best = (float(fov), v)
    if best is None:
        return None
    for fov in np.arange(best[0] - 1.0, best[0] + 1.0001, 0.1):
        v = score(float(fov))
        if v is not None and v < best[1]:
            best = (float(fov), v)
    return best


def fov_from_camera_height(img_pts, world_xy, width: int, height: int, cam_height: float,
                           fov_range=(30.0, 130.0)) -> float | None:
    """FOV for which the fitted camera sits ``cam_height`` metres above the floor (tape-measure the camera)."""
    def score(fov):
        try:
            cam, _ = fit_camera(img_pts, world_xy, Intrinsics(width, height, fov))
        except GeometryError:
            return None
        return abs(float(cam.center[2]) - cam_height)

    best = _scan_fov(score, fov_range)
    return None if best is None or best[1] > 0.25 * cam_height else best[0]


def fov_from_raised_points(img_pts, world_xy, width: int, height: int, raised, fov_range=(30.0, 130.0)):
    """FOV that best maps raised markers back to their known positions.

    ``raised``: list of (pixel (2,), room xyz (3,)) - e.g. the marker held at mic height exactly above a pinned
    floor point.  Returns (fov_deg, mean position error in metres) or None.
    """
    pix = np.array([r[0] for r in raised], dtype=np.float64).reshape(-1, 2)
    xyz = np.array([r[1] for r in raised], dtype=np.float64).reshape(-1, 3)

    def score(fov):
        try:
            cam, _ = fit_camera(img_pts, world_xy, Intrinsics(width, height, fov))
        except GeometryError:
            return None
        err = [np.linalg.norm(cam.backproject(p[None, :], q[2])[0] - q[:2]) for p, q in zip(pix, xyz)]
        e = float(np.mean(err))
        return e if np.isfinite(e) else None

    return _scan_fov(score, fov_range)
