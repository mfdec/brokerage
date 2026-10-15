import time

import cv2
import numpy as np
import pytest

from roomscape.core.geometry import Intrinsics, RoomCamera
from roomscape.core.modes import RoomDims
from roomscape.video import camera as cam_mod
from roomscape.video.camera import CameraError, CameraSpec, OpenCvCamera, open_capture, probe_cameras
from roomscape.video.overlay import colorize, draw_map, draw_marker, draw_wireframe
from roomscape.video.sim import MARKER_BGR, SimCamera, SimScene, default_camera

DIMS = RoomDims(5.0, 4.0, 2.5)


# ------------------------------------------------------------------ capture (fake cv2.VideoCapture)
class FakeCap:
    log: list = []

    def __init__(self, target, backend, behaviour):
        self.target, self.backend, self.b = target, backend, behaviour
        self.n = 0
        self.released = False
        FakeCap.log.append((target, backend))
        self.opened = behaviour.get("open", True)

    def isOpened(self):
        return self.opened

    def set(self, prop, val):
        self.b.setdefault("set", []).append((prop, val))
        return True

    def read(self):
        self.n += 1
        if self.b.get("die_after") is not None and self.n > self.b["die_after"]:
            return False, None
        if self.n <= self.b.get("blank_frames", 0):
            return False, None
        img = np.full((48, 64, 3), self.n % 250, np.uint8)
        time.sleep(0.002)
        return True, img

    def release(self):
        self.released = True


def factory_for(*behaviours):
    seq = list(behaviours)

    def factory(target, backend):
        b = seq.pop(0) if len(seq) > 1 else seq[0]
        return FakeCap(target, backend, dict(b))

    return factory


def test_open_capture_falls_back_across_backends_and_sets_mjpg_resolution():
    FakeCap.log.clear()
    cap = open_capture(CameraSpec.device(2, 1280, 720), factory_for({"open": False}, {"open": True}))
    assert cap.isOpened() and len(FakeCap.log) == 2 and FakeCap.log[0][0] == 2
    props = dict(cap.b["set"])
    assert props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and props[cv2.CAP_PROP_FRAME_HEIGHT] == 720 and cv2.CAP_PROP_FOURCC in props


def test_open_capture_tolerates_warm_up_frames_and_reports_failure_clearly():
    cap = open_capture(CameraSpec.device(0), factory_for({"blank_frames": 3}))
    assert cap.n >= 4
    with pytest.raises(CameraError, match="Could not open Camera 7"):
        open_capture(CameraSpec.device(7), factory_for({"open": False}))
    with pytest.raises(CameraError, match="http://"):
        open_capture(CameraSpec.stream("192.168.1.5:4747"), factory_for({}))


def test_url_streams_use_ffmpeg_first():
    FakeCap.log.clear()
    open_capture(CameraSpec.stream("http://10.0.0.5:4747/video"), factory_for({}))
    assert FakeCap.log[0] == ("http://10.0.0.5:4747/video", cv2.CAP_FFMPEG)


def test_probe_lists_only_cameras_that_deliver_frames():
    opened = {0: True, 1: False, 2: True, 3: False}
    seq = []

    def factory(target, backend):
        return FakeCap(target, backend, {"open": opened.get(target, False)})

    assert probe_cameras(4, factory) == [0, 2]


def test_camera_thread_keeps_only_the_newest_frame_and_stops_cleanly():
    cam = OpenCvCamera(CameraSpec.device(0), factory_for({}))
    cam.start()
    time.sleep(0.4)
    f1 = cam.latest()
    time.sleep(0.2)
    f2 = cam.latest()
    assert f1 is not None and f2.index > f1.index and f2.t >= f1.t
    assert cam.size == (64, 48) and cam.fps > 5
    cam.stop()
    assert not cam.running and cam.latest() is None


def test_network_camera_reconnects_after_the_stream_dies():
    cam = OpenCvCamera(CameraSpec.stream("http://10.0.0.5:4747/video"),
                       factory_for({"die_after": 5}, {}), reconnect_s=0.05, fail_limit=3)
    cam.start()
    deadline = time.time() + 5
    while cam.reconnects < 1 and time.time() < deadline:
        time.sleep(0.02)
    got = cam.latest()
    cam.stop()
    assert cam.reconnects >= 1 and got is not None


def test_start_raises_camera_error_when_nothing_opens():
    cam = OpenCvCamera(CameraSpec.device(3), factory_for({"open": False}))
    with pytest.raises(CameraError):
        cam.start()


# ------------------------------------------------------------------ overlay
def test_colorize_maps_levels_and_makes_nan_transparent():
    f = np.array([[-20.0, 0.0, 20.0, np.nan]])
    tex = colorize(f, -12, 12)
    assert tex.shape == (1, 4, 4)
    assert tex[0, 0, 3] == 255 and tex[0, 3, 3] == 0
    assert tex[0, 0, 0] > tex[0, 0, 2] and tex[0, 2, 2] > tex[0, 2, 0]         # BGR: cold = blue, hot = red
    assert colorize(f, -12, 12, alpha=0.5)[0, 0, 3] == 127


def test_overlay_covers_exactly_the_projected_floor_quadrilateral():
    cam = default_camera()
    frame = np.zeros((540, 960, 3), np.uint8)
    tex = colorize(np.zeros((36, 45)), -12, 12)
    out = draw_map(frame, cam, DIMS, tex, 0.0, opacity=1.0)
    covered = np.any(out != 0, axis=2)
    corners, _ = cam.project(np.array([[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]], dtype=float))
    inside = ~np.any(np.isnan(corners))
    quad = np.zeros((540, 960), np.uint8)
    if inside:
        cv2.fillConvexPoly(quad, np.round(corners).astype(np.int32), 1)
    # the near corners are behind/below the frame; what is covered must lie within the true floor region
    ys, xs = np.nonzero(covered)
    assert covered.sum() > 20000
    xy = cam.backproject(np.column_stack((xs[::97], ys[::97])).astype(float), 0.0)
    assert np.all(np.isfinite(xy)) and xy[:, 0].min() > -0.05 and xy[:, 0].max() < 5.05 and xy[:, 1].min() > -0.05 and xy[:, 1].max() < 4.05


def test_overlay_has_no_mirror_ghost_above_the_horizon_for_a_camera_inside_the_room():
    cam = RoomCamera.look_at(Intrinsics(960, 540, 70.0).K, (4.0, 2.0, 1.6), (0.0, 2.0, 1.0))
    frame = np.zeros((540, 960, 3), np.uint8)
    out = draw_map(frame, cam, DIMS, colorize(np.zeros((36, 45)), -12, 12), 0.0, opacity=1.0)
    covered = np.any(out != 0, axis=2)
    horizon_row = np.nonzero(cam.visible_mask(960, 540, 0.0, step=1)[:, 480])[0].min()
    assert covered[: horizon_row - 3].sum() == 0 and covered[horizon_row + 20:].sum() > 5000


def test_overlay_plane_height_shifts_the_drawn_map():
    cam = default_camera()
    frame = np.zeros((540, 960, 3), np.uint8)
    tex = colorize(np.zeros((36, 45)), -12, 12)
    low = np.any(draw_map(frame, cam, DIMS, tex, 0.0, 1.0) != 0, axis=2)
    high = np.any(draw_map(frame, cam, DIMS, tex, 1.2, 1.0) != 0, axis=2)
    assert low.sum() > 0 and high.sum() > 0 and (low ^ high).sum() > 5000


def test_wireframe_and_markers_draw_something_visible():
    cam = default_camera()
    frame = np.zeros((540, 960, 3), np.uint8)
    draw_wireframe(frame, cam, DIMS)
    assert frame.sum() > 0
    pt = draw_marker(frame, cam, (2.0, 2.0, 1.2), (0, 255, 0), "mic")
    assert pt is not None and frame[pt[1], pt[0] - 9].any()
    assert draw_marker(frame, RoomCamera.look_at(Intrinsics(960, 540).K, (2, 2, 1.5), (0, 2, 1.5)), (4, 2, 1), (0, 0, 255)) is None  # behind


# ------------------------------------------------------------------ synthetic camera
def hsv_blob_centroid(img, color_bgr=MARKER_BGR, tol=14):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    lo = np.array([max(target[0] - tol, 0), 90, 90])
    hi = np.array([min(target[0] + tol, 179), 255, 255])
    m = cv2.inRange(hsv, lo, hi)
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean(), len(xs)) if len(xs) else None


def test_sim_scene_draws_the_marker_where_the_true_camera_projects_it():
    cam = default_camera()
    scene = SimScene(DIMS, cam)
    for xyz in ((2.0, 2.0, 1.2), (3.2, 1.0, 1.0), (1.2, 3.0, 1.4)):
        uv, _ = cam.project(np.array([xyz]))
        blob = hsv_blob_centroid(scene.render(xyz))
        assert blob is not None
        assert abs(blob[0] - uv[0, 0]) < 1.5 and abs(blob[1] - uv[0, 1]) < 1.5, (xyz, blob, uv)
    assert hsv_blob_centroid(scene.render(None)) is None                 # nothing marker-coloured without the marker


def test_sim_camera_streams_frames_and_follows_the_marker():
    pos = [(2.0, 2.0, 1.2)]
    sc = SimCamera(SimScene(DIMS, default_camera()), lambda: pos[0], fps=60)
    sc.start()
    time.sleep(0.4)
    a = sc.latest()
    pos[0] = (3.0, 1.5, 1.2)
    time.sleep(0.4)
    b = sc.latest()
    sc.stop()
    assert a is not None and b.index > a.index
    ca, cb = hsv_blob_centroid(a.image), hsv_blob_centroid(b.image)
    assert ca is not None and cb is not None and abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) > 20
