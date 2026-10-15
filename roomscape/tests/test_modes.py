import numpy as np
import pytest

from roomscape.core.modes import (
    Mode, RoomDims, bonello_check, coincident_modes, count_modes_below, critical_distance, damping_from_t60,
    forced_response, largest_gaps, mode_norm, mode_shape, modal_density, room_modes, schroeder_frequency,
    third_octave_counts,
)

C = 343.0
ROOM = RoomDims(5.0, 4.0, 2.5)


def test_axial_mode_frequencies_follow_c_over_2L():
    modes = {m.n: m.freq for m in room_modes(ROOM, 120, C)}
    assert modes[(1, 0, 0)] == pytest.approx(C / (2 * 5.0))
    assert modes[(0, 1, 0)] == pytest.approx(C / (2 * 4.0))
    assert modes[(0, 0, 1)] == pytest.approx(C / (2 * 2.5))
    assert modes[(2, 0, 0)] == pytest.approx(C / 5.0)
    assert modes[(1, 1, 0)] == pytest.approx(0.5 * C * np.hypot(1 / 5.0, 1 / 4.0))
    assert modes[(1, 1, 1)] == pytest.approx(0.5 * C * np.sqrt((1 / 5) ** 2 + (1 / 4) ** 2 + (1 / 2.5) ** 2))


def test_classification_and_labels():
    ms = {m.n: m for m in room_modes(ROOM, 150, C)}
    assert ms[(2, 0, 0)].kind == "axial" and ms[(2, 0, 0)].label == "2L"
    assert ms[(1, 1, 0)].kind == "tangential" and ms[(1, 1, 0)].label == "1L1W"
    assert ms[(1, 1, 1)].kind == "oblique" and ms[(1, 1, 1)].label == "1L1W1H"
    assert (0, 0, 0) not in ms
    f = [m.freq for m in room_modes(ROOM, 200, C)]
    assert f == sorted(f) and f[-1] <= 200


def test_mode_count_matches_weyl_law():
    for f_max in (150, 250, 400):
        n = len(room_modes(ROOM, f_max, C))
        assert n == pytest.approx(count_modes_below(f_max, ROOM, C), rel=0.12)
    assert modal_density(300, ROOM, C) > modal_density(100, ROOM, C)


def test_mode_shapes_and_norm():
    n = np.array([[1, 0, 0], [1, 1, 1], [0, 2, 0]], dtype=float)
    corners = np.array([[0, 0, 0], [5, 4, 2.5]])
    psi = mode_shape(ROOM, n, corners)
    assert np.allclose(np.abs(psi), 1.0)                                  # antinodes in every corner
    assert mode_shape(ROOM, n, np.array([[2.5, 2.0, 1.25]]))[0, 0] == pytest.approx(0.0, abs=1e-12)   # (1,0,0) node
    assert mode_norm(n).tolist() == [0.5, 0.125, 0.5]
    # mean-square over the volume equals Lambda
    g = np.stack(np.meshgrid(*[np.linspace(0.0, L, 61) for L in ROOM.as_array()], indexing="ij"), -1).reshape(-1, 3)
    msq = (mode_shape(ROOM, n, g) ** 2).mean(axis=0)
    assert np.allclose(msq, mode_norm(n), atol=0.02)


def test_forced_response_is_reciprocal():
    modes = room_modes(ROOM, 250, C)
    a, b = np.array([0.4, 1.3, 1.1]), np.array([3.5, 2.0, 1.2])
    f = np.linspace(25, 240, 50)
    hab = forced_response(ROOM, modes, a, b[None], f, 0.5, C)[0]
    hba = forced_response(ROOM, modes, b, a[None], f, 0.5, C)[0]
    assert np.allclose(hab, hba)


def test_single_mode_resonance_and_bandwidth():
    f_n = C / (2 * 5.0)
    only = [Mode((1, 0, 0), f_n)]
    src, rcv = np.array([0.0, 0.0, 0.0]), np.array([[5.0, 0.0, 0.0]])
    t60 = 1.2
    f = np.linspace(f_n - 4, f_n + 4, 4001)
    mag = np.abs(forced_response(ROOM, only, src, rcv, f, t60, C)[0])
    assert f[np.argmax(mag)] == pytest.approx(f_n, abs=0.01)
    # -3 dB bandwidth of a damped resonator = delta / pi  (delta = 6.91 / T60)
    half = mag >= mag.max() / np.sqrt(2)
    bw = f[half][-1] - f[half][0]
    assert bw == pytest.approx(float(damping_from_t60(t60)) / np.pi, rel=0.03)


def test_schroeder_and_critical_distance():
    assert schroeder_frequency(0.5, 60.0) == pytest.approx(2000 * np.sqrt(0.5 / 60.0))
    assert critical_distance(60.0, 0.5) == pytest.approx(0.057 * np.sqrt(120.0))


def test_cube_has_coincident_modes_and_fails_bonello_while_irregular_room_is_better():
    cube = room_modes(RoomDims(4.0, 4.0, 4.0), 150, C)
    irregular = room_modes(RoomDims(5.3, 3.7, 2.45), 150, C)
    assert len(coincident_modes(cube, 0.01)) >= 3
    assert len(coincident_modes(irregular, 0.01)) < len(coincident_modes(cube, 0.01))
    fc, counts = third_octave_counts(irregular, 25, 150)
    assert counts.sum() == len([m for m in irregular if 20 <= m.freq <= 200]) or counts.sum() > 0
    ok, bad = bonello_check(cube, 25, 150)
    assert not ok and bad


def test_largest_gaps_reports_widest_first():
    modes = [Mode((1, 0, 0), 30.0), Mode((2, 0, 0), 36.0), Mode((3, 0, 0), 60.0), Mode((4, 0, 0), 64.0)]
    g = largest_gaps(modes, 20, 100, top=2)
    assert g[0] == (24.0, 36.0) and g[1][0] == 6.0
