"""AI analyst (Camera tab, step 7, optional): Claude reads the measured room data and says where and why the room rings.

Needs the user's own Anthropic API key (kept in the Windows Credential Manager).  Nothing is sent until "Analyse" is pressed, and
"Preview" shows exactly what would go.  The numbers always come from the app's own analysis; Claude interprets them, and the
result is labelled as an estimate.
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from roomscape.ai.analyst import SYSTEM_PROMPT, TOOLS, Report, ToolBox, build_evidence, parse_report, user_message
from roomscape.ai.client import MODELS, AiError, ClaudeClient
from roomscape.ai.keystore import KeyStore
from roomscape.core.decaymap import build_fields
from roomscape.ui import theme
from roomscape.video import overlay

MARK_BGR = (200, 80, 255)


class _Worker(QtCore.QThread):
    progress = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, client: ClaudeClient, content, handler, parent=None):
        super().__init__(parent)
        self.client, self.content, self.handler = client, content, handler
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            res = self.client.run(SYSTEM_PROMPT, self.content, TOOLS, self.handler, progress=self.progress.emit, cancelled=lambda: self._cancel)
        except AiError as e:
            self.failed.emit(str(e))
        except Exception as e:                                                       # noqa: BLE001 - never kill the GUI thread
            self.failed.emit(f"Unexpected error: {e}")
        else:
            self.done.emit(res)


class AiPanel(QtCore.QObject):
    def __init__(self, tab, decay, keystore: KeyStore | None = None):
        super().__init__(tab)
        self.tab, self.decay, self.state = tab, decay, tab.state
        self.keys = keystore or KeyStore()
        self.transport = None                       # tests plug a fake server in here
        self.worker: _Worker | None = None
        self.report: Report | None = None
        self.marks: list = []
        self._frame: np.ndarray | None = None
        self._build()
        tab.extra_draw.append(self._grab)
        tab.extra_draw.append(self.draw)
        tab.shutdown_hooks.append(self.shutdown)
        self._refresh_key()

    # ================================================================== UI
    def _build(self) -> None:
        box = QtWidgets.QGroupBox("7  AI analyst (optional)")
        f = QtWidgets.QFormLayout(box)
        how = QtWidgets.QLabel(
            "Claude reads your measured numbers (room model, sweep measurements, the decay map) and explains where the room rings and "
            "why, with suggestions and pins on the picture. Needs your own Anthropic API key; usage is billed to your account. Nothing "
            "is sent until you press Analyse - 'Preview' shows exactly what would go. The numbers come from this app; the explanation "
            "is Claude's interpretation, so treat it as an informed estimate.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.MUTED};")
        f.addRow(how)
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("sk-ant-...  (paste, then Save key)")
        self.save_btn = QtWidgets.QPushButton("Save key")
        self.forget_btn = QtWidgets.QPushButton("Forget key")
        self.save_btn.clicked.connect(self._save_key)
        self.forget_btn.clicked.connect(self._forget_key)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.key_edit, 1)
        row.addWidget(self.save_btn)
        row.addWidget(self.forget_btn)
        f.addRow("API key", row)
        self.key_status = QtWidgets.QLabel("")
        f.addRow(self.key_status)
        self.model = QtWidgets.QComboBox()
        for label, mid in MODELS:
            self.model.addItem(label, mid)
        saved = self.state.settings.value("ai/model", "", str)
        if saved:
            self.model.setCurrentIndex(max(self.model.findData(saved), 0))
        self.model.currentIndexChanged.connect(lambda _i: self.state.settings.setValue("ai/model", self.model.currentData()))
        f.addRow("Model", self.model)
        self.send_image = QtWidgets.QCheckBox("Also send the camera picture")
        self.send_image.setToolTip("Sends one JPEG of the current camera view with the overlay (and your pins) to Anthropic.")
        self.send_image.setChecked(self.state.settings.value("ai/image", False, bool))
        self.send_image.toggled.connect(lambda v: self.state.settings.setValue("ai/image", bool(v)))
        f.addRow(self.send_image)
        self.question = QtWidgets.QLineEdit()
        self.question.setPlaceholderText("Optional question, e.g. why is the back corner boomy?")
        f.addRow("Question", self.question)
        row = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview what is sent...")
        self.run_btn = QtWidgets.QPushButton("Analyse with Claude")
        self.run_btn.setObjectName("primary")
        self.preview_btn.clicked.connect(self._preview)
        self.run_btn.clicked.connect(self._run_or_cancel)
        row.addWidget(self.preview_btn)
        row.addWidget(self.run_btn, 1)
        f.addRow(row)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        f.addRow(self.status)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(200)
        self.out.setPlaceholderText("The analysis appears here.")
        f.addRow(self.out)
        row = QtWidgets.QHBoxLayout()
        self.show_marks = QtWidgets.QCheckBox("Pins on the picture")
        self.show_marks.setChecked(True)
        self.show_marks.toggled.connect(lambda _v: self.tab._invalidate())
        save = QtWidgets.QPushButton("Save report...")
        save.clicked.connect(self._save_report)
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(self._clear)
        row.addWidget(self.show_marks)
        row.addStretch(1)
        row.addWidget(save)
        row.addWidget(clear)
        f.addRow(row)
        self.tab.extra_box.addWidget(box)

    # ================================================================== key
    def _refresh_key(self) -> None:
        key, where = self.keys.get()
        if key:
            self.key_status.setText(f"<span style='color:{theme.GOOD}'>Key found</span> ({where}), ends in ...{key[-4:]}")
        else:
            self.key_status.setText(f"No key yet. It is stored in: {self.keys.location}.")
        self.run_btn.setEnabled(bool(key) or self.worker is not None)

    def _save_key(self) -> None:
        try:
            self.keys.set(self.key_edit.text())
        except (ValueError, OSError) as e:
            self.key_status.setText(f"<span style='color:{theme.BAD}'>{e}</span>")
            return
        self.key_edit.clear()
        self._refresh_key()

    def _forget_key(self) -> None:
        self.keys.forget()
        self._refresh_key()

    # ================================================================== run
    def _snapshot(self):
        """Everything the model may see or look up, frozen on the GUI thread so the worker never touches live objects."""
        d = self.decay
        cfg = self.state.room
        meas = list(self.state.measurements)
        frames = list(d.mapper.frames) if d.mapper is not None else []
        shown = d.fields()
        bw = shown.bw_oct if shown is not None else 1.0 / 3.0
        cache: dict = {}

        def fields_for(f_hz):
            if shown is None and not frames:
                return None
            if f_hz is None or shown is not None and abs(np.log(max(f_hz, 1e-3) / shown.fc)) < 0.02:
                return shown
            return build_fields(frames, cfg.dims, float(np.clip(f_hz, 10.0, 600.0)), bw, cache=cache)

        return cfg, meas, shown, ToolBox(cfg, meas, fields_for)

    def _image_b64(self) -> str | None:
        if not self.send_image.isChecked() or self._frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", self._frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    def _payload(self):
        cfg, meas, shown, tools = self._snapshot()
        ev = build_evidence(cfg, meas, shown, self.question.text())
        return ev, tools

    def _preview(self) -> None:
        ev, _ = self._payload()
        dlg = QtWidgets.QDialog(self.tab)
        dlg.setWindowTitle("What would be sent to Claude")
        dlg.resize(720, 640)
        v = QtWidgets.QVBoxLayout(dlg)
        note = "Everything below is sent as text with your question. Claude may also ask the app for extra detail (the same kind of numbers)."
        if self.send_image.isChecked():
            note += " One camera picture is attached." if self._frame is not None else " (No camera picture is available right now, so none is sent.)"
        lab = QtWidgets.QLabel(note)
        lab.setWordWrap(True)
        v.addWidget(lab)
        te = QtWidgets.QPlainTextEdit(json.dumps(ev, indent=1))
        te.setReadOnly(True)
        te.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        v.addWidget(te, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec()

    def _run_or_cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelling...")
            return
        key, _ = self.keys.get()
        if not key:
            self._refresh_key()
            return
        ev, tools = self._payload()
        if ev["decay_map"] is None and not ev["measurements"]:
            self.status.setText("There is nothing to analyse yet: run a sweep measurement (Measure tab) or a decay-mapping walk first.")
            return
        client = ClaudeClient(key, self.model.currentData(), transport=self.transport)
        self.worker = _Worker(client, user_message(ev, self._image_b64()), tools, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._worker_gone)
        self.run_btn.setText("Cancel")
        self.out.clear()
        self.worker.start()

    def _worker_gone(self) -> None:
        w, self.worker = self.worker, None
        if w is not None:
            w.deleteLater()
        self.run_btn.setText("Analyse with Claude")
        self._refresh_key()

    def _finished(self, res) -> None:
        self.report = parse_report(res.text, self.state.room.dims)
        self.marks = self.report.marks
        text = self.report.to_text() or "(empty answer)"
        if not self.report.parsed:
            text = "(Claude's answer was not in the expected format, shown as received.)\n\n" + text
        self.out.setPlainText("AI estimate - an interpretation of the measured numbers, not a measurement.\n\n" + text)
        self.status.setText(f"Done: {res.turns} request(s), {res.input_tokens:,} tokens in / {res.output_tokens:,} out, "
                            f"{len(res.tool_calls)} lookup(s), {len(self.marks)} pin(s).")
        self.tab._invalidate()

    def _failed(self, msg: str) -> None:
        self.status.setText(f"<span style='color:{theme.BAD}'>{msg}</span>")

    # ================================================================== picture
    def _grab(self, out: np.ndarray, cm, tab) -> None:
        self._frame = out.copy() if self.send_image.isChecked() else None

    def draw(self, out: np.ndarray, cm, tab) -> None:
        if not self.marks or not self.show_marks.isChecked():
            return
        z = tab.plane.value()
        for m in self.marks:
            overlay.draw_marker(out, cm, (m.x, m.y, z), MARK_BGR, "AI: " + m.label if m.label else "AI", 6)

    # ================================================================== housekeeping
    def _clear(self) -> None:
        self.report, self.marks = None, []
        self.out.clear()
        self.status.setText("")
        self.tab._invalidate()

    def _save_report(self) -> None:
        if self.report is None:
            self.status.setText("Nothing to save yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.tab, "Save AI report", "roomscape-ai-report.txt", "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("AI estimate - an interpretation of RoomScape's measured numbers, not a measurement.\n\n" + self.report.to_text() + "\n")
        self.status.setText(f"Saved {path}")

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(3000)
