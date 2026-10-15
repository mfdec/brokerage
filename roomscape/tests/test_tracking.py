import cv2
import numpy as np
import pytest

from roomscape.core.geometry import Intrinsics, fit_camera
from roomscape.core.modes import RoomDims
from roomscape.video.sim import MARKER_BGR, SimScene, default_camera
from roomscape.video.tracking import ColorMarkerTracker, MarkerPipeline, PositionFilter

DIMS = RoomDims(5.0, 4.0, 2.5)
FRONT = np.array([[0.8, 0.9], [3.2, 0.9], [3.2, 3.1], [0.8, 3.1]])


@pytest.fixture(scope="module")
def scene():
    return SimScene(DIMS, default_camera(), noise=2.5)


def tracker():
    t = ColorMarkerTracker()
    t.set_target_bgr(MARKER_BGR)
    return t


def disc_frame(center, color_bgr, r=9, size=(640, 480), bg=(40, 44, 48)):
    img = np.full((size[1], size[0], 3), bg, np.uint8)
    cv2.circle(img, (int(center[0]), int(center[1])), r, color_bgr, -1, cv2.LINE_AA)
    return img


def test_marker_is_found_within_a_pixel_everywhere_in_the_walking_area(scene):
    tr = tracker()
    errs = []
    for x in np.linspace(0.9, 3.1, 5):
        for y in np.linspace(1.0, 3.0, 5):
            xyz = (x, y, 1.2)
            uv, _ = scene.cam.project(np.array([xyz]))
            tr.last = None
            det = tr.detect(scene.render(xyz))
            assert det is not None, xyz
            errs.append(np.hypot(det.u - uv[0, 0], det.v - uv[0, 1]))
    assert np.max(errs) < 1.5 and np.mean(errs) < 0.8


def test_click_to_pick_the_colour_and_never_lock_onto_distractors(scene):
    xyz = (2.2, 2.0, 1.2)
    frame = scene.render(xyz)
    uv, _ = scene.cam.project(np.array([xyz]))
    tr = ColorMarkerTracker()
    hsv = tr.set_target_from_click(frame, uv[0, 0] + 1.0, uv[0, 1] - 1.0)
    target_hue = cv2.cvtColor(np.uint8([[MARKER_BGR]]), cv2.COLOR_BGR2HSV)[0, 0, 0]
    assert abs(hsv[0] - target_hue) < 4
    assert tr.detect(frame) is not None
    tr2 = ColorMarkerTracker()
    tr2.set_target_from_click(frame, uv[0, 0], uv[0, 1])
    assert tr2.detect(scene.render(None)) is None             # orange lamp, green plant, blue bag: no false lock


def test_red_hue_wraps_around_zero():
    for bgr in ((20, 20, 230), (60, 20, 230)):                # red slightly toward orange / toward magenta
        frame = disc_frame((300, 200), bgr)
        tr = ColorMarkerTracker()
        tr.set_target_from_click(frame, 300, 200)
        det = tr.detect(frame)
        assert det is not None and abs(det.u - 300) < 1 and abs(det.v - 200) < 1


def test_gate_keeps_the_track_on_the_blob_near_the_last_position():
    tr = ColorMarkerTracker()
    f1 = disc_frame((100, 240), MARKER_BGR, r=8)
    tr.set_target_from_click(f1, 100, 240)
    assert tr.detect(f1) is not None
    both = disc_frame((100, 240), MARKER_BGR, r=8)
    cv2.circle(both, (520, 240), 14, MARKER_BGR, -1, cv2.LINE_AA)            # a bigger, farther lookalike
    det = tr.detect(both)
    assert det is not None and abs(det.u - 100) < 2


def test_dimmer_lighting_still_tracks_but_a_black_room_does_not(scene):
    xyz = (2.0, 2.0, 1.2)
    frame = scene.render(xyz, with_noise=False)
    tr = tracker()
    assert tr.detect((frame * 0.6).astype(np.uint8)) is not None
    tr = tracker()
    assert tr.detect((frame * 0.1).astype(np.uint8)) is None


def test_loses_and_reacquires_after_an_occlusion(scene):
    tr = tracker()
    a, b = (1.5, 1.5, 1.2), (1.6, 1.55, 1.2)
    assert tr.detect(scene.render(a)) is not None
    for _ in range(8):
        assert tr.detect(scene.render(None)) is None
    assert tr.lost == 8
    assert tr.detect(scene.render(b)) is not None and tr.lost == 0


def test_adaptive_colour_follows_slow_drift_that_a_fixed_target_loses():
    def drifted(h_shift):
        hsv = cv2.cvtColor(np.uint8([[MARKER_BGR]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        hsv[0] = (hsv[0] + h_shift) % 180
        return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])

    fixed, adaptive = ColorMarkerTracker(adapt=False), ColorMarkerTracker(adapt=True)
    for t in (fixed, adaptive):
        t.set_target_bgr(MARKER_BGR)
    found = {"fixed": 0, "adaptive": 0}
    for k in range(120):
        frame = disc_frame((300 + k, 200), drifted(k * 0.2), r=9)              # hue drifts 24 units over the walk
        found["fixed"] += fixed.detect(frame) is not None
        found["adaptive"] += adaptive.detect(frame) is not None
    assert found["adaptive"] > 110 and found["fixed"] < 80


# ------------------------------------------------------------------ Kalman filter
def test_kalman_filter_smooths_noise_and_estimates_speed():
    rng = np.random.default_rng(0)
    kf = PositionFilter(accel_sigma=0.8, meas_sigma=0.06)
    errs, speeds = [], []
    for k in range(300):
        t = k / 30
        truth = np.array([1.0 + 0.5 * t, 2.0 + 0.2 * np.sin(t)])
        est = kf.update(t, truth + rng.normal(0, 0.06, 2))
        if k > 30:
            errs.append(np.linalg.norm(est[:2] - truth))
            speeds.append(kf.speed)
    assert np.sqrt(np.mean(np.square(errs))) < 0.05                               # better than the raw 0.085 m (2-D) noise
    assert np.mean(speeds) == pytest.approx(0.53, abs=0.12)


def test_kalman_filter_rejects_false_detections_but_follows_a_real_jump():
    kf = PositionFilter()
    for k in range(40):
        kf.update(k / 30, np.array([2.0, 2.0]))
    for k in range(40, 43):
        est = kf.update(k / 30, np.array([4.5, 0.2]))                             # 3 frames of a false blob far away
    assert np.linalg.norm(est[:2] - [2.0, 2.0]) < 0.3
    for k in range(43, 60):                                                       # ...but if it stays there, it is real
        est = kf.update(k / 30, np.array([4.5, 0.2]))
    assert np.linalg.norm(est[:2] - [4.5, 0.2]) < 0.3


def test_kalman_coasts_through_missing_measurements():
    kf = PositionFilter()
    for k in range(60):
        kf.update(k / 30, np.array([1.0 + 0.6 * k / 30, 2.0]))
    est = None
    for k in range(60, 75):
        est = kf.update(k / 30, None)                                              # half a second of occlusion
    assert est[0] == pytest.approx(1.0 + 0.6 * 74 / 30, abs=0.15)


# ------------------------------------------------------------------ pipeline
def calibrated(scene, fov):
    uv = scene.cam.project(np.column_stack((FRONT, np.zeros(4))))[0]
    cam, _ = fit_camera(uv, FRONT, Intrinsics(960, 540, fov))
    return cam


def run_positions(scene, cam, positions, z=1.2):
    pipe = MarkerPipeline(tracker(), DIMS, z)
    pipe.set_camera(cam)
    out = []
    for k, (x, y) in enumerate(positions):
        st = pipe.process(scene.render((x, y, z)), k / 30)
        out.append((st, (x, y)))
    return pipe, out


def test_pipeline_maps_the_marker_to_room_coordinates_at_mic_height(scene):
    # a walk at ~0.6 m/s along a gentle curve (30 fps)
    pos = [(1.0 + 0.02 * k, 2.0 + 0.6 * np.sin(0.02 * k * 1.4)) for k in range(90)]
    _, out = run_positions(scene, calibrated(scene, 70.0), pos)
    err = [np.hypot(st.xy[0] - p[0], st.xy[1] - p[1]) for st, p in out[10:]]
    assert all(st.valid for st, _ in out[10:]) and np.sqrt(np.mean(np.square(err))) < 0.04
    assert np.mean([st.speed for st, _ in out[30:]]) == pytest.approx(0.6, abs=0.15)


def test_a_wrong_fov_shows_up_as_a_position_offset(scene):
    pos = [(2.0 + 0.01 * k, 2.0) for k in range(30)]
    _, good = run_positions(scene, calibrated(scene, 70.0), pos)
    _, bad = run_positions(scene, calibrated(scene, 60.0), pos)
    eg = np.hypot(good[-1][0].xy[0] - good[-1][1][0], good[-1][0].xy[1] - good[-1][1][1])
    eb = np.hypot(bad[-1][0].xy[0] - bad[-1][1][0], bad[-1][0].xy[1] - bad[-1][1][1])
    assert eg < 0.04 and eb > eg + 0.01


def test_validity_drops_when_the_marker_is_lost_and_the_filter_resets(scene):
    pipe = MarkerPipeline(tracker(), DIMS, 1.2)
    pipe.set_camera(calibrated(scene, 70.0))
    for k in range(20):
        st = pipe.process(scene.render((2.0, 2.0, 1.2)), k / 30)
    assert st.valid and st.detected and st.speed < 0.2
    for k in range(20, 40):                                                  # 0.67 s hidden
        st = pipe.process(scene.render(None), k / 30)
    assert not st.valid and st.lost_s > 0.5
    for k in range(40, 90):                                                  # hidden for good
        st = pipe.process(scene.render(None), k / 30)
    assert st.xy is None and not st.valid


def test_detections_outside_the_room_are_ignored(scene):
    pipe = MarkerPipeline(tracker(), RoomDims(2.0, 2.0, 2.5), 1.2, margin=0.2)   # a smaller 'room' than the scene
    pipe.set_camera(calibrated(scene, 70.0))
    st = pipe.process(scene.render((3.0, 3.0, 1.2)), 0.0)
    assert st.detected and not st.valid


def test_a_persistent_jump_reseeds_the_filter_without_a_fake_velocity():
    kf = PositionFilter()
    for k in range(40):
        kf.update(k / 30, np.array([3.5, 2.0]))
    for k in range(40, 60):                                                  # marker is suddenly (and stays) elsewhere
        est = kf.update(k / 30, np.array([0.8, 0.9]))
    assert kf.jumps == 1
    assert np.linalg.norm(est[:2] - [0.8, 0.9]) < 0.05 and kf.speed < 0.2    # no overshoot / no phantom speed


def test_trail_is_broken_at_jumps_and_after_losing_the_marker(scene):
    pipe = MarkerPipeline(tracker(), DIMS, 1.2)
    pipe.set_camera(calibrated(scene, 70.0))
    k = 0
    for _ in range(15):
        pipe.process(scene.render((2.0, 2.0, 1.2)), k / 30)
        k += 1
    for _ in range(15):                                                      # teleport
        pipe.process(scene.render((3.0, 3.0, 1.2)), k / 30)
        k += 1
    for _ in range(80):                                                      # hidden for long enough to lose the track
        pipe.process(scene.render(None), k / 30)
        k += 1
    for _ in range(15):
        pipe.process(scene.render((1.0, 1.5, 1.2)), k / 30)
        k += 1
    arr = np.array(list(pipe.trail))
    breaks = np.nonzero(~np.isfinite(arr[:, 1]))[0]
    assert len(breaks) >= 2
    pieces = np.split(arr[:, 1:], breaks)
    for piece in pieces:
        piece = piece[np.isfinite(piece[:, 0])]
        if len(piece) > 1:
            assert np.max(np.hypot(*np.diff(piece, axis=0).T)) < 0.5         # nothing is ever drawn across a gap
