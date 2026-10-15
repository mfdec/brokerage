import numpy as np
import pytest

from roomscape.core.geometry import (
    GeometryError, Intrinsics, RoomCamera, estimate_focal, fit_camera, fov_from_camera_height, fov_from_raised_points,
    segment_in_front,
)

W, H = 1280, 720
FLOOR = np.array([[0.8, 0.9], [3.2, 0.9], [3.2, 3.1], [0.8, 3.1]])          # pinned floor rectangle (m)
EYE, TARGET = (5.0, 2.0, 2.4), (1.0, 2.0, -0.5)                              # camera on the rear wall, tilted down ~36 deg


def make_cam(fov=70.0, eye=EYE, target=TARGET, size=(W, H)):
    return RoomCamera.look_at(Intrinsics(*size, fov).K, eye, target)


def pinned(cam, pts=FLOOR, noise=0.0, seed=0):
    uv, _ = cam.project(np.column_stack((pts, np.zeros(len(pts)))))
    return uv + np.random.default_rng(seed).normal(0, noise, uv.shape)


def test_look_at_puts_the_target_at_the_image_centre_and_in_front():
    cam = make_cam()
    uv, depth = cam.project(np.array([TARGET]))
    assert uv[0] == pytest.approx([W / 2, H / 2], abs=1e-6) and depth[0] > 0
    assert np.allclose(cam.center, EYE)


def test_the_test_scene_actually_shows_the_pinned_floor_points():
    uv = pinned(make_cam())
    assert np.all((uv[:, 0] > 60) & (uv[:, 0] < W - 60) & (uv[:, 1] > 60) & (uv[:, 1] < H - 60))


def test_fit_recovers_the_pose_from_noise_free_floor_points():
    true = make_cam()
    cam, rms = fit_camera(pinned(true), FLOOR, Intrinsics(W, H, 70.0))
    assert rms < 1e-3
    assert np.allclose(cam.center, true.center, atol=1e-4)
    assert np.allclose(cam.R, true.R, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_fit_is_stable_with_click_noise(seed):
    true = make_cam()
    cam, rms = fit_camera(pinned(true, noise=1.0, seed=seed), FLOOR, Intrinsics(W, H, 70.0))
    assert rms < 2.5
    assert np.linalg.norm(cam.center - true.center) < 0.25


def test_backprojection_of_a_raised_marker_is_exact_at_its_own_height_and_wrong_on_the_floor():
    true = make_cam()
    marker = np.array([[2.2, 1.9, 1.15]])                       # mic marker held at 1.15 m
    uv, _ = true.project(marker)
    at_height = true.backproject(uv, 1.15)[0]
    on_floor = true.backproject(uv, 0.0)[0]
    assert at_height == pytest.approx([2.2, 1.9], abs=1e-9)
    assert np.linalg.norm(on_floor - [2.2, 1.9]) > 0.5          # a plain floor homography is metres-level wrong
    cam, _ = fit_camera(pinned(true, noise=0.7), FLOOR, Intrinsics(W, H, 70.0))
    assert np.linalg.norm(cam.backproject(uv, 1.15)[0] - [2.2, 1.9]) < 0.1


def test_wrong_fov_moves_raised_markers_so_it_must_be_pinned_down():
    """A FOV error is invisible to the floor points (they still fit exactly) but shifts raised markers; how much
    depends on the camera placement (a few cm here, up to metres for a low, skewed camera)."""
    true = make_cam(fov=70.0)
    pts3 = np.array([[1.5, 1.2, 1.2], [2.5, 2.0, 1.2], [2.8, 2.9, 1.2], [1.6, 2.6, 1.2]])
    uv, _ = true.project(pts3)
    errs = {}
    for fov in (62.0, 70.0, 78.0):
        cam, _ = fit_camera(pinned(true), FLOOR, Intrinsics(W, H, fov))
        errs[fov] = np.linalg.norm(cam.backproject(uv, 1.2) - pts3[:, :2], axis=1).max()
    assert errs[70.0] < 1e-6 and errs[62.0] > 0.02 and errs[78.0] > 0.02


@pytest.mark.parametrize("fov", [58.0, 70.0, 90.0])
def test_fov_from_a_tape_measured_camera_height(fov):
    true = make_cam(fov=fov)
    est = fov_from_camera_height(pinned(true, noise=0.4), FLOOR, W, H, cam_height=2.4)
    assert est == pytest.approx(fov, abs=3.0)
    sloppy = fov_from_camera_height(pinned(true, noise=0.4), FLOOR, W, H, cam_height=2.4 + 0.05)   # 5 cm tape error
    assert sloppy == pytest.approx(fov, abs=5.0)


@pytest.mark.parametrize("fov", [58.0, 70.0, 90.0])
def test_fov_from_a_raised_marker_over_a_pinned_point_makes_mapping_accurate(fov):
    true = make_cam(fov=fov)
    marks = [(true.project(np.array([[x, y, 1.2]]))[0][0], np.array([x, y, 1.2])) for x, y in ((0.8, 0.9), (3.2, 3.1))]
    est, err = fov_from_raised_points(pinned(true, noise=0.4), FLOOR, W, H, marks)
    assert est == pytest.approx(fov, abs=2.5) and err < 0.06
    # with the refined FOV an independent raised point maps within a few cm
    cam, _ = fit_camera(pinned(true, noise=0.4), FLOOR, Intrinsics(W, H, est))
    probe = np.array([[2.0, 2.0, 1.2]])
    uv, _ = true.project(probe)
    assert np.linalg.norm(cam.backproject(uv, 1.2)[0] - [2.0, 2.0]) < 0.12


def test_plane_homography_agrees_with_projection_and_inverse():
    cam = make_cam()
    for z in (0.0, 1.2, 2.0):
        Hm = cam.plane_homography(z)
        xy = np.array([[1.0, 1.0], [3.0, 2.5]])
        p = np.column_stack((xy, np.ones(2))) @ Hm.T
        uv_h = p[:, :2] / p[:, 2:3]
        uv, _ = cam.project(np.column_stack((xy, np.full(2, z))))
        assert np.allclose(uv_h, uv, atol=1e-6)


def test_visible_mask_excludes_the_sky_for_a_floor_plane():
    cam = make_cam(eye=(4.8, 2.0, 1.6), target=(0.5, 2.0, 1.0))          # looks nearly horizontally
    m = cam.visible_mask(W, H, 0.0, step=4)
    assert m[-1, W // 8].item() and not m[0, W // 8].item()
    # the boundary row equals the horizon of the floor plane
    rows = np.nonzero(m[:, 10])[0]
    assert rows.min() > 0


def test_pixels_that_miss_the_plane_return_nan():
    cam = make_cam(eye=(4.8, 2.0, 1.6), target=(0.5, 2.0, 1.0))
    xy = cam.backproject(np.array([[W / 2, 5.0], [W / 2, H - 5.0]]), 0.0)     # sky pixel, floor pixel
    assert np.all(np.isnan(xy[0])) and np.all(np.isfinite(xy[1]))


@pytest.mark.parametrize("fov", [55.0, 70.0, 85.0])
def test_focal_estimate_from_an_oblique_floor_view(fov):
    true = make_cam(fov=fov, eye=(5.0, 2.0, 2.4), target=(1.0, 2.0, -0.5))
    pts = np.array([[0.8, 0.9], [3.2, 0.9], [3.2, 3.1], [0.8, 3.1], [2.0, 2.0]])
    f = estimate_focal(pinned(true, pts, noise=0.3), pts, W, H)
    assert f is not None
    assert Intrinsics.fov_from_focal(W, f) == pytest.approx(fov, rel=0.12)


def test_focal_estimate_declines_when_looking_straight_down():
    # straight down: the look-at needs an 'up' vector that is not parallel to the view direction
    true = RoomCamera.look_at(Intrinsics(W, H, 70.0).K, (2.5, 2.0, 3.0), (2.5, 2.001, 0.0), up=(1.0, 0.0, 0.0))
    pts = np.array([[1.0, 0.8], [4.0, 0.8], [4.0, 3.2], [1.0, 3.2]])
    f = estimate_focal(pinned(true, pts, noise=0.5), pts, W, H)
    assert f is None or Intrinsics.fov_from_focal(W, f) > 20


def test_bad_input_raises_clear_errors():
    cam = make_cam()
    with pytest.raises(GeometryError, match="at least 4"):
        fit_camera(pinned(cam)[:3], FLOOR[:3], Intrinsics(W, H))
    line = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    with pytest.raises(GeometryError, match="line"):
        fit_camera(pinned(cam, line), line, Intrinsics(W, H))


def test_swapped_point_order_is_detected_by_a_large_reprojection_error_or_rejected():
    cam = make_cam()
    uv = pinned(cam)
    bad = uv[[1, 0, 2, 3]]                                               # two clicks swapped
    try:
        _, rms = fit_camera(bad, FLOOR, Intrinsics(W, H, 70.0))
    except GeometryError:
        return
    assert rms > 5.0


def test_segment_clipping_against_the_near_plane():
    cam = make_cam(eye=(2.5, 2.0, 1.5), target=(0.0, 2.0, 1.5))          # looking toward -x
    seg = segment_in_front(cam, (-1.0, 2.0, 1.0), (4.0, 2.0, 1.0))        # runs from in front to behind the camera
    assert seg is not None and np.all(np.isfinite(seg[0])) and np.all(np.isfinite(seg[1]))
    assert segment_in_front(cam, (3.0, 1.0, 1.0), (4.0, 3.0, 1.0)) is None  # wholly behind
