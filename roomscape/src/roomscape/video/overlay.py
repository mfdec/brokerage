"""Drawing a map onto the camera image: perspective-correct heat-map, room wireframe, markers."""

from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg

from roomscape.core.geometry import RoomCamera, segment_in_front
from roomscape.core.modes import RoomDims

_LUTS: dict[str, np.ndarray] = {}


def lut_bgr(name: str = "CET-D1") -> np.ndarray:
    """256x3 uint8 BGR lookup table of a pyqtgraph colormap."""
    if name not in _LUTS:
        rgb = pg.colormap.get(name).getLookupTable(0.0, 1.0, 256)[:, :3]
        _LUTS[name] = np.ascontiguousarray(rgb[:, ::-1]).astype(np.uint8)
    return _LUTS[name]


def colorize(field_db: np.ndarray, lo: float, hi: float, cmap: str = "CET-D1", alpha: np.ndarray | float = 1.0) -> np.ndarray:
    """(ny, nx) dB values -> BGRA uint8 texture.  NaN cells are fully transparent.

    ``alpha`` is a scalar or (ny, nx) array in 0..1 (e.g. data confidence for a measured map).
    """
    f = np.asarray(field_db, dtype=np.float64)
    idx = np.clip(((f - lo) / (hi - lo) * 255.0), 0, 255)
    idx = np.nan_to_num(idx, nan=0.0).astype(np.uint8)
    out = np.empty(f.shape + (4,), dtype=np.uint8)
    out[..., :3] = lut_bgr(cmap)[idx]
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float64), f.shape) * np.isfinite(f)
    out[..., 3] = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return out


def draw_map(frame: np.ndarray, cam: RoomCamera, dims: RoomDims, texture_bgra: np.ndarray, z_plane: float,
             opacity: float = 0.6, mask_cache: dict | None = None) -> np.ndarray:
    """Blend a room-aligned BGRA texture (rows = y, cols = x) onto ``frame`` as a plane at height ``z_plane``."""
    h, w = frame.shape[:2]
    ny, nx = texture_bgra.shape[:2]
    S = np.array([[dims.lx / nx, 0, 0.5 * dims.lx / nx], [0, dims.ly / ny, 0.5 * dims.ly / ny], [0, 0, 1.0]])
    Hm = cam.plane_homography(z_plane) @ S
    warped = cv2.warpPerspective(texture_bgra, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # rays above the plane's horizon would hit it *behind* the camera and show as a mirror image: mask them out
    key = (w, h, round(z_plane, 3), tuple(np.round(cam.rvec, 6)), tuple(np.round(cam.tvec, 6)))
    if mask_cache is not None and mask_cache.get("key") == key:
        vis = mask_cache["mask"]
    else:
        vis = cam.visible_mask(w, h, z_plane, step=2)
        vis = cv2.resize(vis.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask_cache is not None:
            mask_cache["key"], mask_cache["mask"] = key, vis
    alpha = warped[..., 3]
    rows, cols = np.any(alpha > 0, axis=1), np.any(alpha > 0, axis=0)
    if not rows.any():
        return frame
    y0, y1 = np.nonzero(rows)[0][[0, -1]]
    x0, x1 = np.nonzero(cols)[0][[0, -1]]
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))                       # blend only where the map actually is
    a = (alpha[sl].astype(np.float32) / 255.0) * float(opacity) * vis[sl]
    out = frame.copy()
    out[sl] = (frame[sl].astype(np.float32) * (1.0 - a[..., None]) + warped[sl][..., :3].astype(np.float32) * a[..., None]).astype(np.uint8)
    return out


def draw_wireframe(frame: np.ndarray, cam: RoomCamera, dims: RoomDims, color=(200, 200, 200), thickness: int = 1) -> None:
    """Floor and ceiling outline plus the four vertical edges, so the user can judge alignment with the real room."""
    L, W, Hh = dims.lx, dims.ly, dims.lz
    c = [(0, 0), (L, 0), (L, W), (0, W)]
    edges = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        edges.append(((a[0], a[1], 0.0), (b[0], b[1], 0.0)))
        edges.append(((a[0], a[1], Hh), (b[0], b[1], Hh)))
        edges.append(((a[0], a[1], 0.0), (a[0], a[1], Hh)))
    for p0, p1 in edges:
        seg = segment_in_front(cam, p0, p1)
        if seg is None:
            continue
        pt = lambda p: (int(np.clip(p[0], -30000, 30000)), int(np.clip(p[1], -30000, 30000)))      # noqa: E731
        cv2.line(frame, pt(seg[0]), pt(seg[1]), color, thickness, cv2.LINE_AA)


def draw_marker(frame: np.ndarray, cam: RoomCamera, p3, color, label: str = "", radius: int = 9) -> tuple[int, int] | None:
    uv, depth = cam.project(np.asarray(p3, dtype=float)[None, :])
    if not np.all(np.isfinite(uv)):
        return None
    pt = (int(uv[0, 0]), int(uv[0, 1]))
    cv2.circle(frame, pt, radius, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, pt, radius, color, 2, cv2.LINE_AA)
    if label:
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = pt[0] + radius + 3
        if x + tw > frame.shape[1] - 2:                          # no room on the right: put the label on the left of the marker
            x = max(pt[0] - radius - 3 - tw, 2)
        cv2.putText(frame, label, (x, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return pt


def draw_trail(frame: np.ndarray, cam: RoomCamera, xy_path: np.ndarray, z: float, color=(255, 255, 255)) -> None:
    """Polyline of the walked path; rows with NaN coordinates split it into separate pieces."""
    xy = np.asarray(xy_path, dtype=float)
    if len(xy) < 2:
        return
    pieces = np.split(xy, np.nonzero(~np.all(np.isfinite(xy), axis=1))[0])
    for piece in pieces:
        piece = piece[np.all(np.isfinite(piece), axis=1)]
        if len(piece) < 2:
            continue
        uv, _ = cam.project(np.column_stack((piece, np.full(len(piece), z))))
        uv = uv[np.all(np.isfinite(uv), axis=1)].astype(np.int32)
        if len(uv) >= 2:
            cv2.polylines(frame, [uv.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
