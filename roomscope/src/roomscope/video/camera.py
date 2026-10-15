"""Camera capture: USB/built-in webcams, DroidCam's virtual webcam, and network streams (MJPEG / RTSP).

Everything goes through OpenCV's VideoCapture in a background thread that always keeps only the *newest*
frame, so a slow GUI never builds up latency.  Network streams reconnect automatically.

Typical sources
* USB webcam / laptop camera / DroidCam client ("DroidCam Source 3" virtual webcam): ``CameraSpec.device(index)``
* DroidCam over Wi-Fi without the PC client:    http://<phone-ip>:4747/video
* Android "IP Webcam" app:                      http://<phone-ip>:8080/video
* RTSP camera:                                  rtsp://user:pass@<ip>:554/stream1
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

URL_PRESETS = {
    "DroidCam (Wi-Fi)": "http://PHONE-IP:4747/video",
    "IP Webcam app (Android)": "http://PHONE-IP:8080/video",
    "RTSP camera": "rtsp://USER:PASSWORD@CAMERA-IP:554/stream1",
}
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSpec:
    index: int | None = None
    url: str | None = None
    width: int = 1280
    height: int = 720

    @classmethod
    def device(cls, index: int, width: int = 1280, height: int = 720) -> "CameraSpec":
        return cls(index=index, width=width, height=height)

    @classmethod
    def stream(cls, url: str) -> "CameraSpec":
        return cls(url=url.strip())

    @property
    def label(self) -> str:
        return self.url if self.url else f"Camera {self.index}"


def _backends() -> list[int]:
    """Preferred OpenCV backends for local devices (DirectShow opens fastest on Windows)."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def open_capture(spec: CameraSpec, factory: Callable = cv2.VideoCapture):
    """Open ``spec`` and return a cv2.VideoCapture-like object that delivers at least one frame."""
    if spec.url:
        if not spec.url.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
            raise CameraError("Stream URLs start with http:// or rtsp:// (for DroidCam: http://<phone-ip>:4747/video)")
        attempts = [(spec.url, cv2.CAP_FFMPEG), (spec.url, cv2.CAP_ANY)]
    else:
        attempts = [(spec.index, b) for b in _backends()]
    last = "no backend could open it"
    for target, backend in attempts:
        cap = factory(target, backend)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        if spec.index is not None:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # unlocks HD at full fps on most USB cams
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            except Exception:
                pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                                 # keep latency low where supported
        except Exception:
            pass
        ok, frame = cap.read()
        for _ in range(10):                                                     # cameras need a few frames to warm up
            if ok and frame is not None:
                break
            time.sleep(0.1)
            ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        last = "opened but delivered no picture"
        cap.release()
    raise CameraError(f"Could not open {spec.label}: {last}. Is another program (e.g. the DroidCam client) using it?")


class Frame:
    __slots__ = ("image", "t", "index")

    def __init__(self, image: np.ndarray, t: float, index: int):
        self.image, self.t, self.index = image, t, index


class CameraSource:
    """Common interface for the GUI/tracker: ``latest()`` returns the newest frame (or None)."""

    is_simulated = False
    name = "camera"

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def latest(self) -> Frame | None:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def running(self) -> bool:  # pragma: no cover - interface
        return False


class OpenCvCamera(CameraSource):
    def __init__(self, spec: CameraSpec, factory: Callable = cv2.VideoCapture, reconnect_s: float = 2.0, fail_limit: int = 30):
        self.spec = spec
        self._fail_limit = fail_limit
        self.name = spec.label
        self._factory = factory
        self._reconnect_s = reconnect_s
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._count = 0
        self.fps = 0.0
        self.size = (0, 0)
        self.error: str | None = None
        self.reconnects = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._cap = open_capture(self.spec, self._factory)                       # raises CameraError
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        with self._lock:
            self._frame = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def _run(self) -> None:
        t_fps, n_fps, fails = time.monotonic(), 0, 0
        while not self._stop.is_set():
            cap = self._cap
            ok, img = cap.read() if cap is not None else (False, None)
            if ok and img is not None:
                fails = 0
                self._count += 1
                now = time.monotonic()
                with self._lock:
                    self._frame = Frame(img, now, self._count)
                self.size = (img.shape[1], img.shape[0])
                n_fps += 1
                if now - t_fps >= 0.5:
                    self.fps = n_fps / (now - t_fps)
                    t_fps, n_fps = now, 0
                continue
            fails += 1
            if fails < self._fail_limit:
                time.sleep(0.02)
                continue
            # the stream died: release and try to reopen (Wi-Fi hiccup, phone app restarted, ...)
            self.error = "connection lost, reconnecting..."
            if cap is not None:
                cap.release()
            self._cap = None
            while not self._stop.is_set():
                time.sleep(self._reconnect_s)
                try:
                    self._cap = open_capture(self.spec, self._factory)
                    self.reconnects += 1
                    self.error = None
                    fails = 0
                    break
                except CameraError as e:
                    self.error = str(e)


def probe_cameras(max_index: int = 6, factory: Callable = cv2.VideoCapture) -> list[int]:
    """Indices of local cameras that deliver a frame (slow: each missing index can take about a second on Windows)."""
    found = []
    for i in range(max_index):
        try:
            cap = open_capture(CameraSpec.device(i, 640, 480), factory)
        except CameraError:
            continue
        cap.release()
        found.append(i)
    return found


def camera_names() -> list[str]:
    """DirectShow device names in OpenCV's index order (Windows, needs the optional ``pygrabber``)."""
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph  # noqa: PLC0415

        return list(FilterGraph().get_input_devices())
    except Exception:
        return []
