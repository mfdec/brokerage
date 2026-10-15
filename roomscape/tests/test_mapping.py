"""Walk-the-room mapping, verified against ground truth in virtual time (no hardware, no real-time waiting)."""

import time

import numpy as np
import pytest

from roomscape.core.geometry import Intrinsics, fit_camera
from roomscape.core.mapping import MapperSettings, WalkMapper
from roomscape.core.multisine import MultisineSpec, analyze_frame, line_params
from roomscape.sim.world import LawnmowerPath, SimStimulusPlayer, SimWorld
from roomscape.video.sim import MARKER_BGR, SimScene, default_camera
from roomscape.video.tracking import ColorMarkerTracker, MarkerPipeline

FS = 48000
FRONT = np.array([[0.8, 0.9], [3.2, 0.9], [3.2, 3.1], [0.8, 3.1]])


# --------------------------------------------------------------------------- simulated stimulus
def test_simulated_room_returns_the_stimulus_with_the_analytic_response_at_the_mic():
    world = SimWorld(FS)
    spec = MultisineSpec(FS, level_dbfs=-20)
    SimStimulusPlayer(world).start(spec)
    block = np.concatenate([world.render(1024) for _ in range(120)])            # ~2.5 s
    res = analyze_frame(block[-2 * spec.n_period:], spec)
    f, a, _ = line_params(spec)
    H = world.room.frequency_response(world.mic, f) * world.gain
    expected = 20 * np.log10(a * np.abs(H) / np.sqrt(2))
    assert np.median(np.abs(res.power_db - expected)) < 0.1 and np.abs(res.power_db - expected).max() < 0.6
    assert res.snr_db > 40
    SimStimulusPlayer(world).stop()


def test_walker_moves_the_microphone_with_the_audio_clock():
    world = SimWorld(FS)
    world.walker = LawnmowerPath(0.8, 3.2, 0.9, 3.1, speed=0.5, start_delay_s=0.5)
    start = world.marker_xyz()
    for _ in range(int(3.0 * FS / 1024)):
        world.render(1024)
    x, y, z = world.marker_xyz()
    assert (x, y) != tuple(start[:2]) and z == 1.2 and world.time_s == pytest.approx(3.0, abs=0.03)
    assert x == pytest.approx(0.8 + 0.5 * (3.0 - 0.5), abs=0.1)


# --------------------------------------------------------------------------- mapper gating
def stim_audio(spec, seconds, gain_db=-30.0, noise_dbfs=-100.0, seed=0):
    f, a, th = line_params(spec)
    t = np.arange(int(seconds * spec.fs)) / spec.fs
    x = np.cos(2 * np.pi * np.outer(t, f) + th) @ (a * 10 ** (gain_db / 20))
    return x + 10 ** (noise_dbfs / 20) * np.random.default_rng(seed).standard_normal(x.size)


def feed(mapper, spec, seconds, path_fn, audio=None, valid=True, fps=30):
    audio = stim_audio(spec, seconds) if audio is None else audio
    n_frames = int(seconds * fps)
    chunk = len(audio) // n_frames
    for k in range(n_frames):
        t = (k + 1) / fps
        xy = path_fn(t)
        mapper.feed_position(t, xy, valid)
        mapper.feed_audio(audio[k * chunk:(k + 1) * chunk], t)


def test_mapper_accepts_a_slow_tracked_walk_and_rejects_bad_frames():
    spec = MultisineSpec(FS)
    dims = SimWorld(FS).room.dims
    m = WalkMapper(dims, MapperSettings(spec=spec))
    feed(m, spec, 6.0, lambda t: (1.0 + 0.3 * t, 2.0))                           # 0.3 m/s
    assert m.counts["accepted"] >= 8 and m.field.stats.n_added == m.counts["accepted"]

    fast = WalkMapper(dims, MapperSettings(spec=spec))
    feed(fast, spec, 6.0, lambda t: (0.5 + 1.2 * t, 2.0))                        # 1.2 m/s: too fast
    assert fast.counts["speed"] >= 6 and fast.counts["accepted"] == 0

    lost = WalkMapper(dims, MapperSettings(spec=spec))
    feed(lost, spec, 6.0, lambda t: None, valid=False)
    assert lost.counts["no_position"] >= 8 and lost.counts["accepted"] == 0

    quiet = WalkMapper(dims, MapperSettings(spec=spec))
    feed(quiet, spec, 6.0, lambda t: (2.0, 2.0), audio=1e-5 * np.random.default_rng(0).standard_normal(6 * FS))
    assert quiet.counts["snr"] >= 8 and quiet.counts["accepted"] == 0

    hot = WalkMapper(dims, MapperSettings(spec=spec))
    feed(hot, spec, 6.0, lambda t: (2.0, 2.0), audio=np.clip(stim_audio(spec, 6.0, gain_db=30.0), -1, 1))
    assert hot.counts["clipped"] >= 8 and hot.counts["accepted"] == 0


def test_mapper_survives_any_audio_chunking_and_reports_stationarity():
    spec = MultisineSpec(FS)
    dims = SimWorld(FS).room.dims
    audio = stim_audio(spec, 8.0)
    results = []
    for chunk in (480, 1000, 4800, 24000):
        m = WalkMapper(dims, MapperSettings(spec=spec))
        for k in range(0, len(audio), chunk):
            t = (k + chunk) / FS
            m.feed_position(t, (2.0, 2.0), True)
            m.feed_audio(audio[k:k + chunk], t)
        results.append(m.counts["accepted"])
        assert m.stationary_for(2.0)
    assert len(set(results)) == 1 and results[0] >= 12                            # identical frames whatever the chunk size


def test_camera_delay_shifts_positions_back_in_time():
    spec = MultisineSpec(FS)
    m = WalkMapper(SimWorld(FS).room.dims, MapperSettings(spec=spec, camera_delay_s=0.3))
    m.feed_position(1.0, (1.0, 1.0), True)
    assert m._pos[-1][0] == pytest.approx(0.7)


# --------------------------------------------------------------------------- the headline test
def simulate_walk(fov_error_deg=0.0, speed=0.5, seed=1, fps=20):
    world = SimWorld(FS, seed=seed)
    world.demo_on = False
    spec = MultisineSpec(FS, 0.5, 20.0, 300.0, True, -20.0)
    world.set_stimulus(spec)
    world.walker = LawnmowerPath(0.8, 3.2, 0.9, 3.1, z=1.2, speed=speed, row_spacing=0.55, pause_s=0.3, start_delay_s=1.0)
    scene = SimScene(world.room.dims, default_camera(), (960, 540), speaker=tuple(world.room.src), seed=seed)
    uv = scene.cam.project(np.column_stack((FRONT, np.zeros(4))))[0]
    cam, _ = fit_camera(uv, FRONT, Intrinsics(960, 540, 70.0 + fov_error_deg))
    tracker = ColorMarkerTracker()
    tracker.set_target_bgr(MARKER_BGR)
    pipe = MarkerPipeline(tracker, world.room.dims, 1.2)
    pipe.set_camera(cam)
    mapper = WalkMapper(world.room.dims, MapperSettings(spec=spec))
    truth, t = [], 0.0
    n = int(FS / fps)
    while t < world.walker.duration + 1.0:
        block = world.render(n)
        t += n / FS
        st = pipe.process(scene.render(world.marker_xyz()), t)
        mapper.feed_position(t, st.xy, st.valid)
        mapper.feed_audio(block, t)
        truth.append((t, st.xy, tuple(world.mic[:2])))
    return world, mapper, pipe, truth, spec


@pytest.fixture(scope="module")
def walk():
    t0 = time.time()
    out = simulate_walk()
    out[1].wall = time.time() - t0
    return out


def truth_map(world, mapper, freq):
    """Ground-truth dB map (relative to its median) on the mapper's cells, from the analytic modal room."""
    gx, gy = np.meshgrid(mapper.field.xs, mapper.field.ys)
    pts = np.stack((gx.ravel(), gy.ravel(), np.full(gx.size, 1.2)), axis=1)
    H = world.room.frequency_response_many(pts, np.array([freq]))[:, 0]
    return 20 * np.log10(np.abs(H)).reshape(gx.shape)


def compare(world, mapper, freq, min_conf=0.7):
    meas = mapper.field.level_db(freq)
    conf = mapper.field.confidence() > min_conf
    ok = conf & np.isfinite(meas)
    tr = truth_map(world, mapper, freq)
    m, t = meas[ok] - np.median(meas[ok]), tr[ok] - np.median(tr[ok])
    return float(np.corrcoef(m, t)[0, 1]), float(np.sqrt(np.mean((m - t) ** 2))), int(ok.sum())


def test_tracking_error_during_the_walk_is_a_few_centimetres(walk):
    world, mapper, pipe, truth, spec = walk
    err = [np.hypot(est[0] - true[0], est[1] - true[1]) for _t, est, true in truth if est is not None]
    assert len(err) > 0.9 * len(truth) and np.sqrt(np.mean(np.square(err[20:]))) < 0.06


def test_walk_produces_frames_and_covers_the_walked_area(walk):
    world, mapper, pipe, truth, spec = walk
    assert mapper.counts["accepted"] > 40 and mapper.accepted_fraction > 0.7
    inside = (mapper.field.xs[None, :] > 0.9) & (mapper.field.xs[None, :] < 3.1) & (mapper.field.ys[:, None] > 1.0) & (mapper.field.ys[:, None] < 3.0)
    assert np.mean(mapper.field.confidence()[inside] > 0.7) > 0.85


@pytest.mark.parametrize("freq,min_r", [(45.0, 0.9), (70.0, 0.9), (100.0, 0.9), (130.0, 0.8)])
def test_measured_map_matches_the_modal_ground_truth(walk, freq, min_r):
    """Spatial detail shrinks with frequency (half a wavelength is 1.3 m at 130 Hz), so the footprint of each
    measurement (0.25 m) and the motion during a 1 s frame cost correlation as frequency rises."""
    world, mapper, pipe, truth, spec = walk
    r, rmse, n = compare(world, mapper, freq)
    assert n > 300
    assert r > min_r and rmse < 3.5, (freq, r, rmse)


def test_a_wrong_fov_degrades_the_map_gracefully():
    world, mapper, pipe, truth, spec = simulate_walk(fov_error_deg=-8.0, fps=15)
    errs = [np.hypot(e[0] - tr[0], e[1] - tr[1]) for _t, e, tr in truth if e is not None]
    r, rmse, n = compare(world, mapper, 70.0, min_conf=0.5)
    assert np.mean(errs) > 0.01                                     # the mis-calibrated camera does shift positions
    assert r > 0.6                                                  # ...yet the standing-wave pattern is still recognisable
