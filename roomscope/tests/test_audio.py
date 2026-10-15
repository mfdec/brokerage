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

    def __init__(self, **kw):
        self.kw = kw
        self.started = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


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

    def input_stream(**kw):
        if fail_open:
            raise RuntimeError("Invalid sample rate")
        return FakeStream(**kw)

    sd.InputStream = input_stream
    return sd


@pytest.fixture
def fake_sd(monkeypatch):
    FakeStream.instances.clear()
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
    mic = devices.likely_measurement_mic(devices.list_input_devices())
    src = HardwareInput(mic, 48000)
    src.start()
    st = FakeStream.instances[-1]
    assert st.started and st.kw["samplerate"] == 48000 and st.kw["channels"] == 1
    assert st.kw["extra_settings"] == ("wasapi", {"auto_convert": True})        # WASAPI rate conversion enabled
    cb = st.kw["callback"]
    cb(np.full((256, 1), 0.25, dtype=np.float32), 256, None, None)
    class Status:  # truthy -> counts as an xrun
        def __bool__(self): return True
    cb(np.full((256, 1), 0.5, dtype=np.float32), 256, None, Status())
    assert src.ring.total == 512 and src.xruns == 1
    assert src.ring.read_latest(2).tolist() == [0.5, 0.5]
    src.stop()
    assert not src.running and not st.started


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
