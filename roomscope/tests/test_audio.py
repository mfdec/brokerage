import sys
import threading
import time
import types

import numpy as np
import pytest

from roomscope.audio import devices
from roomscope.audio.engine import AudioError, HardwareInput, SimulatedInput
from roomscope.audio.ringbuffer import RingBuffer
from roomscope.sim.demo import DemoRoomSignal


# ---------------------------------------------------------------- ring buffer
def test_ringbuffer_wraparound_and_latest():
    rb = RingBuffer(10)
    rb.write(np.arange(7, dtype=np.float32))
    rb.write(np.arange(7, 14, dtype=np.float32))            # wraps
    assert rb.total == 14
    assert rb.read_latest(5).tolist() == [9, 10, 11, 12, 13]
    assert rb.read_range(8, 14).tolist() == [8, 9, 10, 11, 12, 13]


def test_ringbuffer_zero_pads_and_flags_overwrites():
    rb = RingBuffer(8)
    rb.write(np.array([1, 2, 3], dtype=np.float32))
    assert rb.read_latest(6).tolist() == [0, 0, 0, 1, 2, 3]
    rb.write(np.arange(20, dtype=np.float32))                # larger than capacity
    assert rb.total == 23
    assert rb.read_range(0, 5) is None                        # overwritten
    assert rb.read_range(20, 30) is None                      # not yet written
    assert rb.read_latest(8).tolist() == list(range(12, 20))


def test_ringbuffer_read_since_cursor():
    rb = RingBuffer(100)
    cur = 0
    rb.write(np.ones(30, dtype=np.float32))
    a, cur, over = rb.read_since(cur)
    assert a.size == 30 and cur == 30 and not over
    rb.write(np.ones(250, dtype=np.float32))
    b, cur, over = rb.read_since(cur)
    assert b.size == 100 and over


# ---------------------------------------------------------------- fake PortAudio
class FakeStream:
    instances = []
    script_in: list = []                     # [(block (n,ch) float32, overflowed), ...] returned by read()

    def __init__(self, **kw):
        self.kw = kw
        self.started = False
        self.aborted = False
        self.stopped = False
        self.closed = False
        self.written: list[np.ndarray] = []
        self._script = list(FakeStream.script_in)
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False
        self.stopped = True

    def abort(self):
        self.started = False
        self.aborted = True

    def close(self):
        self.closed = True

    def read(self, frames):
        if self.aborted:
            raise RuntimeError("stream aborted")
        if self._script:
            return self._script.pop(0)
        time.sleep(0.003)
        return np.zeros((frames, self.kw["channels"]), dtype=np.float32), False

    def write(self, data):
        self.written.append(np.array(data))
        return False


def make_fake_sd(fail_open=False):
    sd = types.SimpleNamespace()
    sd.query_hostapis = lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]
    sd.query_devices = lambda: [
        {"name": "Microphone (iMM-6C)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0},
        {"name": "Microphone (iMM-6C)", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "Microphone (iMM-6C)", "hostapi": 2, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "Speakers (Realtek)", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "Stereo Mix", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
    ]
    sd.default = types.SimpleNamespace(device=(0, 3))
    sd.check_input_settings = lambda **kw: None
    sd.WasapiSettings = lambda **kw: ("wasapi", kw)

    def stream_factory(**kw):
        if fail_open:
            raise RuntimeError("Invalid sample rate")
        return FakeStream(**kw)

    sd.InputStream = stream_factory
    sd.OutputStream = stream_factory
    return sd


@pytest.fixture
def fake_sd(monkeypatch):
    FakeStream.instances.clear()
    FakeStream.script_in = []
    sd = make_fake_sd()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def test_device_listing_hides_wdmks_and_prefers_wasapi(fake_sd):
    devs = devices.list_input_devices()
    assert devs[0].is_simulated
    real = devs[1:]
    assert all("WDM" not in d.hostapi for d in real)
    assert real[0].hostapi == "Windows WASAPI"
    assert any(d.name == "Stereo Mix" for d in real)
    assert devices.list_output_devices()[1].name.startswith("Speakers")


def test_likely_measurement_mic_and_rates(fake_sd):
    devs = devices.list_input_devices()
    mic = devices.likely_measurement_mic(devs)
    assert mic is not None and "iMM-6C" in mic.name
    rates = devices.rates_for(mic)
    assert rates[0] == int(mic.default_fs) and 48000 in rates and len(set(rates)) == len(rates)


def test_hardware_input_streams_into_ring(fake_sd):
    FakeStream.script_in = [
        (np.full((256, 1), 0.25, dtype=np.float32), False),
        (np.full((256, 1), 0.5, dtype=np.float32), True),          # PortAudio reports an input overflow
    ]
    mic = devices.likely_measurement_mic(devices.list_input_devices())
    src = HardwareInput(mic, 48000)
    src.start()
    st = FakeStream.instances[-1]
    assert st.started and st.kw["samplerate"] == 48000 and st.kw["channels"] == 1
    assert st.kw["extra_settings"] == ("wasapi", {"auto_convert": True})        # WASAPI rate conversion enabled
    assert st.kw["latency"] == "high" and "callback" not in st.kw                # blocking reads, big device buffer
    deadline = time.time() + 3
    while src.ring.total < 512 and time.time() < deadline:
        time.sleep(0.01)
    assert src.ring.read_range(0, 512).tolist() == [0.25] * 256 + [0.5] * 256
    assert src.xruns == 1
    src.stop()
    assert not src.running and st.aborted and st.closed
    assert not any(t.name == "audio-in" for t in threading.enumerate())


def test_hardware_input_reports_stream_failure(fake_sd):
    mic = devices.likely_measurement_mic(devices.list_input_devices())
    src = HardwareInput(mic, 48000)
    src.start()
    FakeStream.instances[-1].aborted = True                  # the fake raises on the next read, like a yanked USB mic
    deadline = time.time() + 3
    while src.running and time.time() < deadline:
        time.sleep(0.01)
    assert not src.running and "aborted" in (src.error or "")
    src.stop()


def test_hardware_player_channels_and_completion(fake_sd):
    from roomscope.audio.measure import HardwarePlayer

    out = [d for d in devices.list_output_devices() if not d.is_simulated][0]
    sig = np.linspace(-1, 1, 5000).astype(np.float32)
    pl = HardwarePlayer(out, "Left")
    pl.play(sig, 48000)
    deadline = time.time() + 3
    while pl.playing and time.time() < deadline:
        time.sleep(0.01)
    assert not pl.playing
    st = FakeStream.instances[-1]
    played = np.concatenate(st.written)
    assert played.shape == (5000, 2)
    assert np.allclose(played[:, 0], sig) and np.all(played[:, 1] == 0)           # left only
    assert st.stopped and st.closed and st.kw["latency"] == "high"
    pl2 = HardwarePlayer(out, "Both")
    pl2.play(sig, 48000)
    pl2.stop()                                                                     # abort path
    assert not pl2.playing


def test_hardware_open_failure_becomes_audio_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd(fail_open=True))
    mic = devices.likely_measurement_mic(devices.list_input_devices())
    with pytest.raises(AudioError, match="Invalid sample rate"):
        HardwareInput(mic, 96000).start()


def test_backend_missing_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)       # forces ImportError
    devs = devices.list_input_devices()
    assert len(devs) == 1 and devs[0].is_simulated
    assert devices.backend_error() is not None
    with pytest.raises(AudioError):
        HardwareInput(devices.DeviceInfo(1, "x", "MME", 1, 0, 48000.0), 48000).start()


# ---------------------------------------------------------------- simulated input
def test_simulated_input_pump_and_thread():
    src = SimulatedInput(DemoRoomSignal(48000), 48000)
    src.pump(2.0)
    assert src.ring.total == 96000
    x = src.ring.read_latest(48000)
    assert 0.001 < x.std() < 0.5
    src2 = SimulatedInput(DemoRoomSignal(48000), 48000)
    src2.start()
    time.sleep(0.5)
    src2.stop()
    assert 0.35 * 48000 < src2.ring.total < 0.75 * 48000          # real-time paced (not a burst)
    assert not any(t.name == "sim-audio" for t in threading.enumerate())


def test_hardware_stimulus_player_loops_whole_periods_seamlessly_and_fades(fake_sd, monkeypatch):
    from roomscope.audio.measure import HardwareStimulusPlayer
    from roomscope.core.multisine import MultisineSpec, generate

    def paced_write(self, data):                                   # a real write blocks until the device has consumed it
        self.written.append(np.array(data))
        time.sleep(0.004)
        return False

    monkeypatch.setattr(FakeStream, "write", paced_write)
    out = [d for d in devices.list_output_devices() if not d.is_simulated][0]
    spec = MultisineSpec(48000, 0.5, 20.0, 300.0, True, -20.0)
    period = generate(spec)
    pl = HardwareStimulusPlayer(out, "Left")
    pl.start(spec)
    assert pl.playing
    time.sleep(0.15)
    pl.stop()
    assert not pl.playing
    st = FakeStream.instances[-1]
    assert st.kw["samplerate"] == 48000 and st.kw["channels"] == 2 and st.kw["latency"] == "high"
    assert st.stopped and st.closed
    first, tiles, last = st.written[0], st.written[1:-1], st.written[-1]
    assert len(tiles) >= 2
    assert tiles[0].shape[0] % period.size == 0                    # the loop tile is a whole number of periods: no seam
    assert np.allclose(tiles[0][:, 0], np.tile(period, tiles[0].shape[0] // period.size), atol=1e-6)
    assert np.all(tiles[0][:, 1] == 0)                             # left channel only
    assert abs(first[0, 0]) < 1e-3 and np.max(np.abs(first[:2400, 0])) < np.max(np.abs(tiles[0][:2400, 0]))   # faded in
    assert abs(last[-1, 0]) < 0.02 * np.max(np.abs(period))        # faded out: no click when it stops
    pl2 = HardwareStimulusPlayer(out, "Both")
    pl2.start(spec)
    pl2.stop()
    assert not pl2.playing
