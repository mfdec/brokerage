"""``roomscope-doctor``: every verdict path, with a fake PortAudio and fake cameras."""

import sys
import time

import numpy as np
import pytest

import roomscope.diagnostics as dx
from test_audio import FakeStream, make_fake_sd


class PacedStream(FakeStream):
    """A microphone that delivers audio at real-time speed: noise of a chosen level, silence, or full scale."""

    kind = "noise"
    level_db = -70.0

    _t_next = None

    def read(self, frames):
        if self.aborted:
            raise RuntimeError("stream aborted")
        if self._t_next is None:
            self._t_next = time.monotonic()
        self._t_next += frames / self.kw["samplerate"]                      # an absolute schedule: no cumulative drift
        time.sleep(max(self._t_next - time.monotonic(), 0.0))
        ch = self.kw["channels"]
        if self.kind == "silence":
            x = np.zeros((frames, ch))
        elif self.kind == "square":
            x = np.where(np.arange(frames)[:, None] % 2 == 0, 1.0, -1.0) * np.ones((1, ch))
        else:
            x = np.random.default_rng(0).standard_normal((frames, ch)) * 10 ** (self.level_db / 20)
        return x.astype(np.float32), False


@pytest.fixture
def mic(monkeypatch):
    FakeStream.instances.clear()
    sd = make_fake_sd()
    sd.InputStream = lambda **kw: PacedStream(**kw)
    sd.get_portaudio_version = lambda: (1246720, "PortAudio V19.7.0-devel")
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    PacedStream.kind, PacedStream.level_db = "noise", -70.0
    return sd


def statuses(checks):
    return [c.status for c in checks]


def test_software_check_lists_versions_and_flags_nothing_wrong_here():
    out = dx.check_software()
    assert out[0].status == dx.INFO and "RoomScope" in out[0].title
    assert dx.FAIL not in statuses(out)
    assert any("NumPy" in c.title for c in out) and any("OpenCV" in c.title for c in out)


def test_audio_devices_are_listed_with_the_measurement_mic_and_defaults(mic):
    out = dx.check_audio_devices()
    text = "\n".join(c.format() for c in out)
    assert dx.FAIL not in statuses(out)
    assert "iMM-6C" in text and "measurement mic looks like" in text and "PortAudio V19" in text
    assert "Speakers (Realtek)" in text and "WDM" not in text                 # the unusable host API stays hidden


def test_missing_audio_backend_is_reported_with_a_remedy(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    out = dx.check_audio_devices()
    assert out[0].status == dx.FAIL and "pip install" in out[0].detail


@pytest.mark.parametrize("kind,level,expected,needle", [
    ("noise", -70.0, dx.OK, "noise floor"),
    ("silence", -70.0, dx.FAIL, "Digital silence"),
    ("square", -70.0, dx.WARN, "clipping"),
    ("noise", -20.0, dx.WARN, "Very loud"),
    ("noise", -110.0, dx.WARN, "Extremely quiet"),
])
def test_microphone_verdicts(mic, kind, level, expected, needle):
    PacedStream.kind, PacedStream.level_db = kind, level
    out = dx.check_microphone(None, 48000, 1.3)
    assert out[0].status == expected, out[0].format()
    assert needle in out[0].detail
    assert "iMM-6C" in out[0].title                                          # picked the measurement mic by default
    assert dx.WARN not in statuses(out[1:]) or kind != "noise"               # no spurious dropout / rate warnings
    assert FakeStream.instances[-1].closed


def test_microphone_that_cannot_be_opened_explains_itself(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd(fail_open=True))
    out = dx.check_microphone(None, 48000, 0.5)
    assert out[0].status == dx.FAIL and "Invalid sample rate" in out[0].detail


def test_unknown_device_name_is_a_failure_not_a_crash(mic):
    out = dx.check_microphone("no such mic", 48000, 0.5)
    assert out[0].status == dx.FAIL and "no such mic" in out[0].detail


def test_output_tone(mic):
    out = dx.check_output(None, 48000)
    assert out[0].status == dx.OK and "Speakers" in out[0].title
    st = FakeStream.instances[-1]
    played = np.concatenate(st.written)
    assert 0.0 < np.max(np.abs(played)) < 0.05                               # quiet: -30 dBFS


class FakeCam:
    fps = 29.7
    ok = True

    def __init__(self, spec):
        self.spec = spec

    def start(self):
        from roomscope.video.camera import CameraError

        if not self.ok:
            raise CameraError("Could not open: nope")

    def stop(self):
        pass

    def latest(self):
        from roomscope.video.camera import Frame

        return Frame(np.zeros((720, 1280, 3), np.uint8), 0.0, 1)


def test_camera_report_for_found_and_missing_cameras(monkeypatch):
    import roomscope.video.camera as cam

    monkeypatch.setattr(cam, "probe_cameras", lambda n=6, factory=None: [0, 2])
    monkeypatch.setattr(cam, "camera_names", lambda: ["Integrated Camera", "x", "DroidCam Source 3"])
    monkeypatch.setattr(cam, "OpenCvCamera", FakeCam)
    monkeypatch.setattr(dx.time, "sleep", lambda s: None)
    out = dx.check_cameras()
    assert [c.status for c in out] == [dx.OK, dx.OK]
    assert "DroidCam Source 3" in out[1].title and "1280x720" in out[1].detail
    monkeypatch.setattr(cam, "probe_cameras", lambda n=6, factory=None: [])
    out = dx.check_cameras()
    assert out[0].status == dx.WARN and "DroidCam" in out[0].detail


def test_url_check(monkeypatch):
    import roomscope.video.camera as cam

    monkeypatch.setattr(dx.time, "sleep", lambda s: None)
    monkeypatch.setattr(cam, "OpenCvCamera", FakeCam)
    assert dx.check_url("http://10.0.0.5:4747/video")[0].status == dx.OK
    FakeCam.ok = False
    try:
        out = dx.check_url("http://10.0.0.5:4747/video")
    finally:
        FakeCam.ok = True
    assert out[0].status == dx.FAIL and "same Wi-Fi" in out[0].detail


def test_bad_url_scheme_is_explained_without_any_network():
    out = dx.check_url("10.0.0.5:4747/video")
    assert out[0].status == dx.FAIL and "http://" in out[0].detail


def test_dsp_self_test_recovers_the_known_decay_time():
    out = dx.check_dsp()
    assert out[0].status == dx.OK, out[0].format()
    assert "0.45" in out[0].detail


def test_cli_prints_a_report_and_sets_the_exit_code(mic, capsys):
    rc = dx.main(["--skip-dsp"])
    txt = capsys.readouterr().out
    assert rc == 0 and "RoomScope doctor" in txt and "[ ok ]" in txt and "problem(s)" in txt
    PacedStream.kind = "silence"
    rc = dx.main(["--skip-dsp", "--mic", "--seconds", "1.2"])
    txt = capsys.readouterr().out
    assert rc == 1 and "[FAIL]" in txt and "Digital silence" in txt
