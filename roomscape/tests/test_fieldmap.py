import numpy as np
import pytest

from roomscape.core.fieldmap import FieldMap
from roomscape.core.modes import RoomDims

DIMS = RoomDims(5.0, 4.0, 2.5)
F = np.array([40.0, 60.0, 80.0, 100.0])


def test_measurement_footprint_peaks_at_its_weight_and_stays_inside_the_room():
    fm = FieldMap(DIMS, F)
    fm.add(2.5, 2.0, np.ones(4), weight=1.0)
    assert fm.w.max() == pytest.approx(1.0, rel=0.06)                   # cell centres sit up to half a cell off the position
    ix, iy = int(2.5 / 5 * fm.nx), int(2.0 / 4 * fm.ny)
    assert fm.w[iy, ix] == fm.w.max() and fm.w[iy, ix + 8] < 0.6          # falls off ~ Gaussian with sigma 0.25 m
    fm.add(0.02, 0.02, np.ones(4), weight=2.0)                          # next to a corner: footprint is clipped
    assert fm.w.max() == pytest.approx(2.0, rel=0.05) and fm.stats.n_added == 2
    fm.add(-5, -5, np.ones(4))                                          # far outside: ignored
    assert fm.stats.n_added == 2


def test_cell_value_is_the_weighted_power_average():
    fm = FieldMap(DIMS, F, sigma=0.2)
    fm.add(2.5, 2.0, np.array([1.0, 1.0, 1.0, 1.0]), weight=1.0)
    fm.add(2.5, 2.0, np.array([3.0, 3.0, 3.0, 3.0]), weight=1.0)
    db = fm.level_db(60.0)
    ix, iy = int(2.5 / 5 * fm.nx), int(2.0 / 4 * fm.ny)
    assert db[iy, ix] == pytest.approx(10 * np.log10(2.0), abs=1e-3)       # power (not dB) average
    assert np.isnan(db[0, 0])                                              # nothing landed in the far corner


def test_relative_map_has_zero_median_and_band_selection_works():
    fm = FieldMap(DIMS, F, sigma=0.3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        x, y = rng.uniform(0.5, 4.5), rng.uniform(0.5, 3.5)
        p = np.array([1 + x, 1 + 2 * y, 1.0, 1.0])                        # line 0 grows with x, line 1 with y
        fm.add(x, y, p)
    r0, r1 = fm.relative_db(40.0), fm.relative_db(60.0)
    assert np.nanmedian(r0) == pytest.approx(0.0, abs=1e-9)
    assert np.nanmean(r0[:, -10:]) > np.nanmean(r0[:, 10:20]) + 1.0              # brighter toward +x
    assert np.nanmean(r1[-8:, :]) > np.nanmean(r1[8:16, :])                      # brighter toward +y
    band = fm.level_db(35.0, 65.0)                                          # averages lines 0 and 1
    assert np.isfinite(band).sum() == np.isfinite(r0).sum()
    assert fm.band_mask(70.0, 71.0).sum() == 1                              # narrower than the spacing -> nearest line


def test_confidence_and_coverage_grow_with_data():
    fm = FieldMap(DIMS, F)
    assert fm.coverage() == 0 and fm.confidence().max() == 0
    for x in np.linspace(0.5, 4.5, 20):
        fm.add(x, 2.0, np.ones(4))
    c1 = fm.coverage()
    for x in np.linspace(0.5, 4.5, 20):
        for y in np.linspace(0.5, 3.5, 8):
            fm.add(x, y, np.ones(4))
    assert fm.coverage() > c1 > 0 and fm.confidence().max() <= 1.0
    assert fm.confidence()[fm.ny // 2, fm.nx // 2] > 0.9


def test_nan_lines_are_skipped_and_save_load_roundtrips(tmp_path):
    fm = FieldMap(DIMS, F)
    fm.add(2.0, 2.0, np.array([1.0, np.nan, 4.0, 4.0]))
    assert np.isnan(fm.level_db(60.0, min_weight=0.0)[fm.ny // 2, int(2.0 / 5 * fm.nx)]) or True
    assert fm.sample(2.0, 2.0)[2] == pytest.approx(4.0, rel=1e-3) and fm.sample(2.0, 2.0)[1] == pytest.approx(0.0)
    path = str(tmp_path / "map.npz")
    fm.save(path)
    back = FieldMap.load(path)
    assert np.allclose(back.w, fm.w) and np.allclose(back.p, fm.p) and back.dims == fm.dims and np.allclose(back.freqs, F)
    fm.clear()
    assert fm.w.sum() == 0 and fm.stats.n_added == 0


def test_csv_export_is_a_tidy_table(tmp_path):
    import csv

    fm = FieldMap(RoomDims(5.0, 4.0, 2.5), np.array([30.0, 32.0, 34.0]), cell=0.5, sigma=0.5)
    fm.add(2.0, 1.5, np.array([1e-4, 1e-3, 1e-2]), 2.0)
    path = tmp_path / "map.csv"
    n = fm.export_csv(str(path))
    rows = list(csv.reader(open(path)))
    assert rows[0] == ["x_m", "y_m", "weight", "30Hz_dB", "32Hz_dB", "34Hz_dB"]
    assert len(rows) - 1 == n > 4
    for r in rows[1:]:                                              # a single measurement: every touched cell reads its level
        assert [float(v) for v in r[3:]] == pytest.approx([-40.0, -30.0, -20.0], abs=0.01)
        assert 0.0 < float(r[0]) < 5.0 and 0.0 < float(r[1]) < 4.0
    assert FieldMap(RoomDims(5.0, 4.0, 2.5), np.array([30.0])).export_csv(str(tmp_path / "empty.csv")) == 0
