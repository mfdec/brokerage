"""Synthetic camera: renders the room as seen by a virtual webcam, with a coloured marker ball on the mic.

Used for hardware-free demos and for testing tracking/mapping against ground truth.  The scene is drawn from
the *true* camera pose with straightforward polygon projection, independent of the estimation code.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from roomscope.core.geometry import Intrinsics, RoomCamera
from roomscope.core.modes import RoomDims
from roomscope.video.camera import CameraSource, Frame

MARKER_BGR = (190, 30, 255)          # hot pink: a hue that does not occur anywhere else in the scene
DEFAULT_EYE, DEFAULT_TARGET, DEFAULT_FOV = (5.0, 2.0, 2.4), (1.0, 2.0, -0.5), 70.0


def default_camera(width: int = 960, height: int = 540, fov: float = DEFAULT_FOV) -> RoomCamera:
    return RoomCamera.look_at(Intrinsics(width, height, fov).K, DEFAULT_EYE, DEFAULT_TARGET)


def _clip_near(pc: np.ndarray, near: float = 0.1) -> np.ndarray:
    out = []
    for i in range(len(pc)):
        a, b = pc[i], pc[(i + 1) % len(pc)]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            out.append(a + (near - a[2]) / (b[2] - a[2]) * (b - a))
    return np.array(out)


class SimScene:
    def __init__(self, dims: RoomDims, cam: RoomCamera, size: tuple[int, int] = (960, 540),
                 speaker=(0.4, 1.3, 1.1), seed: int = 0, distractors: bool = True, noise: float = 2.5):
        self.dims, self.cam, self.size = dims, cam, size
        self.speaker = speaker
        self.distractors = distractors
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._static: np.ndarray | None = None

    # ------------------------------------------------------------------ drawing helpers
    def _uv(self, pc: np.ndarray) -> np.ndarray:
        f = np.array([self.cam.K[0, 0], self.cam.K[1, 1]])
        return pc[:, :2] / pc[:, 2:3] * f + self.cam.K[:2, 2]

    def _poly(self, img, pts3, color, edge=None) -> None:
        pc = _clip_near(self.cam.to_camera(np.asarray(pts3, dtype=float)))
        if len(pc) < 3:
            return
        uv = np.round(self._uv(pc)).astype(np.int32)
        cv2.fillPoly(img, [uv.reshape(-1, 1, 2)], color, cv2.LINE_AA)
        if edge is not None:
            cv2.polylines(img, [uv.reshape(-1, 1, 2)], True, edge, 1, cv2.LINE_AA)

    def _line(self, img, p0, p1, color, th=1) -> None:
        pc = self.cam.to_camera(np.array([p0, p1], dtype=float))
        if pc[0, 2] < 0.1 and pc[1, 2] < 0.1:
            return
        for i, j in ((0, 1), (1, 0)):
            if pc[i, 2] < 0.1:
                pc[i] = pc[i] + (0.1 - pc[i, 2]) / (pc[j, 2] - pc[i, 2]) * (pc[j] - pc[i])
        uv = np.round(self._uv(pc)).astype(int)
        cv2.line(img, tuple(uv[0]), tuple(uv[1]), color, th, cv2.LINE_AA)

    def _ball(self, img, p3, radius_m: float, color, highlight=True) -> None:
        pc = self.cam.to_camera(np.asarray(p3, dtype=float)[None, :])[0]
        if pc[2] < 0.2:
            return
        uv = self._uv(pc[None, :])[0]
        r = max(2, int(round(radius_m * self.cam.K[0, 0] / pc[2])))
        c = (int(round(uv[0])), int(round(uv[1])))
        cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
        if highlight and r >= 4:
            light = tuple(int(min(255, v + 70)) for v in color)
            cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(1, r // 3), light, -1, cv2.LINE_AA)

    def _static_layer(self) -> np.ndarray:
        w, h = self.size
        img = np.zeros((h, w, 3), np.uint8)
        img[:] = (48, 44, 40)
        L, W, Hh = self.dims.lx, self.dims.ly, self.dims.lz
        c = [(0, 0), (L, 0), (L, W), (0, W)]
        # walls (far to near by depth of their centre), floor, grid
        walls = []
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            poly = [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], Hh), (a[0], a[1], Hh)]
            centre = np.mean(poly, axis=0)
            walls.append((self.cam.to_camera(centre[None, :])[0, 2], poly, i))
        shade = [(88, 84, 80), (78, 76, 74), (96, 92, 88), (70, 68, 66)]
        for _, poly, i in sorted(walls, key=lambda t: -t[0]):
            self._poly(img, poly, shade[i], edge=(120, 120, 120))
        self._poly(img, [(x, y, 0) for x, y in c], (88, 118, 150), edge=(140, 160, 180))
        for x in np.arange(0.5, L, 0.5):
            self._line(img, (x, 0, 0), (x, W, 0), (104, 134, 166))
        for y in np.arange(0.5, W, 0.5):
            self._line(img, (0, y, 0), (L, y, 0), (104, 134, 166))
        sx, sy, sz = self.speaker                                    # speaker cabinet
        self._poly(img, [(sx, sy - 0.15, 0), (sx, sy + 0.15, 0), (sx, sy + 0.15, sz + 0.3), (sx, sy - 0.15, sz + 0.3)], (60, 60, 64), edge=(150, 150, 150))
        self._poly(img, [(sx + 0.25, sy - 0.15, 0), (sx + 0.25, sy + 0.15, 0), (sx + 0.25, sy + 0.15, sz + 0.3), (sx + 0.25, sy - 0.15, sz + 0.3)], (84, 84, 90), edge=(150, 150, 150))
        if self.distractors:                                         # things the tracker must NOT lock onto
            self._ball(img, (L - 0.5, 0.4, 0.45), 0.28, (60, 140, 50))            # green plant
            self._ball(img, (L * 0.6, W - 0.35, 1.6), 0.10, (30, 130, 250))       # orange lamp
            self._ball(img, (L * 0.3, 0.35, 0.3), 0.12, (170, 90, 40))            # blue-ish bag
        return img

    # ------------------------------------------------------------------ public
    def render(self, marker_xyz=None, with_noise: bool = True) -> np.ndarray:
        if self._static is None:
            self._static = self._static_layer()
        img = self._static.copy()
        if marker_xyz is not None:
            x, y, z = marker_xyz
            self._line(img, (x, y, 0.0), (x, y, z - 0.04), (110, 110, 112), 2)      # mic stand
            self._ball(img, (x, y, z), 0.035, MARKER_BGR)
        if with_noise and self.noise > 0:
            img = cv2.GaussianBlur(img, (3, 3), 0.6)
            n = self._rng.normal(0, self.noise, img.shape)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return img


class SimCamera(CameraSource):
    """A ``CameraSource`` whose frames come from a ``SimScene``; the marker sits on the simulated microphone."""

    is_simulated = True
    name = "Simulated camera"

    def __init__(self, scene: SimScene, position_fn, fps: float = 30.0):
        self.scene = scene
        self.position_fn = position_fn                  # () -> (x, y, z) of the marker, or None
        self.period = 1.0 / fps
        self._frame: Frame | None = None
        self._count = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.fps = fps
        self.size = scene.size
        self.error = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sim-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def grab_now(self) -> Frame:
        """Render one frame synchronously (offline tests, virtual time)."""
        self._count += 1
        f = Frame(self.scene.render(self.position_fn()), time.monotonic(), self._count)
        with self._lock:
            self._frame = f
        return f

    def _run(self) -> None:
        t_next = time.monotonic()
        while not self._stop.is_set():
            self.grab_now()
            t_next += self.period
            self._stop.wait(max(0.0, t_next - time.monotonic()))
