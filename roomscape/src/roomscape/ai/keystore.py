"""Where the Claude API key lives.

Windows: the Credential Manager (per user, not a plain file, not in QSettings).  Other systems: the ``ANTHROPIC_API_KEY`` environment
variable, or memory for the running session only.  The key is never written to the settings file, the logs or the exports.
"""

from __future__ import annotations

import os
import sys

TARGET = "RoomScape/AnthropicApiKey"
ENV_VAR = "ANTHROPIC_API_KEY"


class _WindowsBackend:
    """Generic credential in the Windows Credential Manager (advapi32), through ctypes: no extra package needed."""

    name = "Windows Credential Manager"

    def __init__(self, target: str = TARGET):
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        self.target = target
        self.ct, self.wt = ctypes, wintypes

        class CREDENTIAL(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                        ("Comment", wintypes.LPWSTR), ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
                        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
                        ("UserName", wintypes.LPWSTR)]

        self.CRED = CREDENTIAL
        adv = ctypes.WinDLL("advapi32", use_last_error=True)
        self.read = adv.CredReadW
        self.read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
        self.read.restype = wintypes.BOOL
        self.write = adv.CredWriteW
        self.write.argtypes = [ctypes.POINTER(CREDENTIAL), wintypes.DWORD]
        self.write.restype = wintypes.BOOL
        self.delete = adv.CredDeleteW
        self.delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.delete.restype = wintypes.BOOL
        self.free = adv.CredFree
        self.free.argtypes = [ctypes.c_void_p]

    def get(self) -> str | None:
        p = self.ct.POINTER(self.CRED)()
        if not self.read(self.target, 1, 0, self.ct.byref(p)):
            return None
        try:
            c = p.contents
            raw = self.ct.string_at(c.CredentialBlob, c.CredentialBlobSize)
            return raw.decode("utf-8") or None
        finally:
            self.free(p)

    def set(self, key: str) -> None:
        blob = key.encode("utf-8")
        buf = (self.ct.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = self.CRED()
        cred.Type, cred.TargetName, cred.UserName = 1, self.target, "RoomScape"
        cred.CredentialBlobSize, cred.CredentialBlob, cred.Persist = len(blob), buf, 2     # generic, kept for this PC
        if not self.write(self.ct.byref(cred), 0):
            raise OSError(f"Credential Manager refused the key (error {self.ct.get_last_error()})")

    def remove(self) -> None:
        self.delete(self.target, 1, 0)


class _MemoryBackend:
    name = "memory (this session only)"

    def __init__(self):
        self._key: str | None = None

    def get(self) -> str | None:
        return self._key

    def set(self, key: str) -> None:
        self._key = key

    def remove(self) -> None:
        self._key = None


class KeyStore:
    def __init__(self, backend=None):
        if backend is None:
            try:
                backend = _WindowsBackend() if sys.platform == "win32" else _MemoryBackend()
            except Exception:                               # noqa: BLE001 - ctypes problems must never break start-up
                backend = _MemoryBackend()
        self.backend = backend

    @property
    def location(self) -> str:
        return self.backend.name

    def get(self) -> tuple[str | None, str]:
        """(key, where it came from).  The environment variable only fills in when nothing was saved."""
        try:
            key = self.backend.get()
        except Exception:                                   # noqa: BLE001
            key = None
        if key:
            return key, self.backend.name
        env = os.environ.get(ENV_VAR, "").strip()
        return (env, f"environment variable {ENV_VAR}") if env else (None, "")

    def set(self, key: str) -> None:
        key = key.strip()
        if not key.startswith("sk-ant-") or len(key) < 20 or any(c.isspace() for c in key):
            raise ValueError("That does not look like an Anthropic API key (they start with 'sk-ant-').")
        self.backend.set(key)

    def forget(self) -> None:
        self.backend.remove()
