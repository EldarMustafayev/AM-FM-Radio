"""
Software radio: mix two audio files onto carrier frequencies you choose, then
tune between them with a knob, like an old radio.

Based on the experiment in radio.ipynb (carrier * audio -> sum -> recover), but
streamed in real time so the knob changes what you hear instantly.

  transmitter : each file is modulated (AM or FM) onto its own carrier
  "air"       : both modulated signals are added together (+ a little RF noise)
  receiver    : knob frequency -> mix down to baseband -> channel filter ->
                demodulate.  Off-station you hear static, like a real radio.

Run:  .venv-1\\Scripts\\python radio.py

The simulation runs at 6x the audio rate (264.6 kHz) so carriers can live in
the 20-120 kHz range ("dial" units are simulated kHz, not real RF).
"""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter

# ---------------------------------------------------------------- constants
AUDIO_SR = 44100
UP = 6                          # simulation oversampling
SIM_SR = AUDIO_SR * UP          # 264600 Hz -> carriers up to ~132 kHz
BLOCK = 2205                    # 50 ms audio blocks

F_MIN, F_MAX = 20.0, 120.0      # dial range, kHz
AUDIO_BW = 5000.0               # transmitted audio bandwidth, Hz
FM_DEV = 5000.0                 # FM peak deviation, Hz
AM_INDEX = 0.8                  # AM modulation index (same as the notebook)
TX_AMP = 1.0 / (1.0 + AM_INDEX)  # keeps AM peaks <= 1
CHANNEL_CUTOFF = 11000.0        # receiver half-bandwidth, Hz (Carson: 5k+5k)
MIN_SPACING_KHZ = 25.0          # closer than this the two stations overlap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = ("dakiti.mp3", "alors_on_danse.mp3")


# -------------------------------------------------------------- DSP engine
def load_audio(path):
    """Load mono audio, band-limit to AUDIO_BW (so stations stay in-channel)."""
    y, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
    sos = butter(8, AUDIO_BW, fs=AUDIO_SR, output="sos")
    y = sosfiltfilt(sos, y)
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak * 0.95).astype(np.float64)


class Station:
    def __init__(self, name, freq_khz):
        self.name = name
        self.freq = freq_khz     # carrier, kHz
        self.audio = None        # np.ndarray or None
        self.phase = 0.0         # carrier phase accumulator (continuous)

    def transmit(self, pos, n, mode):
        """n audio samples starting at pos -> n*UP samples of modulated carrier."""
        audio = self.audio
        if audio is None:
            return None
        idx = (pos + np.arange(n + 1)) % len(audio)
        a = np.interp(np.arange(n * UP) / UP, np.arange(n + 1), audio[idx])
        w = 2 * np.pi * self.freq * 1000.0 / SIM_SR
        if mode == "FM":
            # frequency follows the audio: f(t) = fc + dev * audio(t)
            inc = w + 2 * np.pi * FM_DEV * a / SIM_SR
            ph = self.phase + np.cumsum(inc)
            sig = TX_AMP * np.cos(ph)
        else:
            # AM exactly as in the notebook: (1 + m*audio) * cos(2*pi*fc*t)
            ph = self.phase + w * np.arange(1, n * UP + 1)
            sig = TX_AMP * (1 + AM_INDEX * a) * np.cos(ph)
        self.phase = ph[-1] % (2 * np.pi)
        return sig


class Radio:
    """Holds both stations and the receiver; render() is called from the
    audio callback thread."""

    def __init__(self):
        self.stations = [Station("A", 55.0), Station("B", 85.0)]
        self.mode = "FM"
        self.tune = 55.0         # knob, kHz
        self.volume = 0.6
        self.static = 0.5
        self.rf_noise = 0.02
        self.pos = 0

        self.lp = butter(6, CHANNEL_CUTOFF, fs=SIM_SR, output="sos")
        self.zi_re = np.zeros((self.lp.shape[0], 2))
        self.zi_im = np.zeros((self.lp.shape[0], 2))
        self.lo_phase = 0.0
        self.prev_bb = 1.0 + 0j
        self.dc_zi = np.zeros(1)
        self.q = 0.0             # crossfade station<->static (0..1)
        self.strength = 0.0      # for the meter (0..1)
        self.rng = np.random.default_rng()

    def render(self, n):
        mode = self.mode
        air = self.rng.normal(0.0, self.rf_noise, n * UP)
        for st in self.stations:
            tx = st.transmit(self.pos, n, mode)
            if tx is not None:
                air += tx
        self.pos = (self.pos + n) % (1 << 40)

        # --- receiver: mix the tuned frequency down to 0 Hz
        w = 2 * np.pi * self.tune * 1000.0 / SIM_SR
        ph = self.lo_phase + w * np.arange(1, n * UP + 1)
        self.lo_phase = ph[-1] % (2 * np.pi)
        bb = air * np.exp(-1j * ph)

        # --- channel filter (the "tuner"), then drop to audio rate
        re, self.zi_re = sosfilt(self.lp, bb.real, zi=self.zi_re)
        im, self.zi_im = sosfilt(self.lp, bb.imag, zi=self.zi_im)
        bb = (re + 1j * im)[::UP]

        mag = float(np.mean(np.abs(bb)))
        self.strength = min(1.0, mag / (TX_AMP / 2))
        target = float(np.clip((mag - 0.05) / 0.2, 0.0, 1.0))
        self.q += 0.6 * (target - self.q)

        if mode == "FM":
            z = np.concatenate(([self.prev_bb], bb))
            self.prev_bb = bb[-1]
            dphi = np.angle(z[1:] * np.conj(z[:-1]))   # phase step = freq offset
            demod = dphi * AUDIO_SR / (2 * np.pi * FM_DEV)
        else:
            demod = (2 * np.abs(bb) / TX_AMP - 1) / AM_INDEX   # envelope
        demod, self.dc_zi = lfilter([1, -1], [1, -0.995], demod, zi=self.dc_zi)

        hiss = self.rng.normal(0.0, 0.25, n) * self.static
        g = self.q ** 2          # weak/garbled signals fade out quickly
        out = g * demod + (1 - g) * hiss
        return np.clip(out * self.volume, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- GUI
BG, PANEL, FG, DIM = "#1b1713", "#2a241d", "#f0e6d2", "#9a8d78"
BRASS, NEEDLE = "#d9a441", "#e5483b"
STATION_COLORS = ("#5ec2b7", "#e58bd0")


class KnobDial(tk.Canvas):
    """Rotary tuning knob. Drag with the mouse, scroll wheel for fine tuning."""
    SWEEP = 270.0

    def __init__(self, master, app, size=240):
        super().__init__(master, width=size, height=size, bg=PANEL,
                         highlightthickness=0)
        self.app, self.size = app, size
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self.bind("<MouseWheel>", self._wheel)
        self.redraw()

    def _angle(self, f):
        return -self.SWEEP / 2 + (f - F_MIN) / (F_MAX - F_MIN) * self.SWEEP

    def _drag(self, e):
        c = self.size / 2
        ang = np.degrees(np.arctan2(e.x - c, c - e.y))      # 0 = up, clockwise
        ang = float(np.clip(ang, -self.SWEEP / 2, self.SWEEP / 2))
        f = F_MIN + (ang + self.SWEEP / 2) / self.SWEEP * (F_MAX - F_MIN)
        self.app.set_tune(round(f, 1))

    def _wheel(self, e):
        step = 0.1 if not (e.state & 0x0001) else 1.0       # Shift = coarse
        self.app.set_tune(self.app.radio.tune + (step if e.delta > 0 else -step))

    def redraw(self):
        self.delete("all")
        c, r = self.size / 2, self.size / 2 - 14
        # tick marks every 5 kHz, labels every 20
        for k in range(int(F_MIN), int(F_MAX) + 1, 5):
            a = np.radians(self._angle(k))
            major = k % 20 == 0
            r1 = r - (14 if major else 7)
            self.create_line(c + r1 * np.sin(a), c - r1 * np.cos(a),
                             c + r * np.sin(a), c - r * np.cos(a),
                             fill=FG if major else DIM, width=2 if major else 1)
            if major:
                rl = r + 9
                self.create_text(c + rl * np.sin(a), c - rl * np.cos(a),
                                 text=str(k), fill=DIM, font=("Segoe UI", 8))
        kr = r - 24
        self.create_oval(c - kr, c - kr, c + kr, c + kr, fill="#3a3027",
                         outline=BRASS, width=3)
        self.create_oval(c - kr + 12, c - kr + 12, c + kr - 12, c + kr - 12,
                         fill="#2e261f", outline="#4b3f32")
        a = np.radians(self._angle(self.app.radio.tune))
        self.create_line(c + 10 * np.sin(a), c - 10 * np.cos(a),
                         c + (kr - 6) * np.sin(a), c - (kr - 6) * np.cos(a),
                         fill=NEEDLE, width=5, capstyle="round")


class App:
    def __init__(self, root):
        self.root = root
        self.radio = Radio()
        self.stream = None
        root.title("Two-Station Radio")
        root.configure(bg=BG)
        root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Panel.TLabel", background=PANEL, foreground=FG)
        style.configure("TButton", padding=4)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("Horizontal.TScale", background=PANEL)

        self._build_stations()
        self._build_receiver()
        self._build_controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.status.set("Pick files for both stations, then press Power.")
        for i, name in enumerate(DEFAULT_FILES):
            p = os.path.join(HERE, name)
            if os.path.exists(p):
                self.load_file(i, p)
        self.poll()

    # -- station panels (file + carrier frequency)
    def _build_stations(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=14, pady=(14, 6))
        self.file_vars, self.freq_vars = [], []
        for i, st in enumerate(self.radio.stations):
            box = ttk.Frame(top, style="Panel.TFrame", padding=10)
            box.grid(row=0, column=i, padx=5, sticky="nsew")
            top.columnconfigure(i, weight=1)
            tk.Label(box, text=f"Station {st.name}", bg=PANEL,
                     fg=STATION_COLORS[i], font=("Segoe UI", 12, "bold")
                     ).grid(row=0, column=0, columnspan=3, sticky="w")
            fv = tk.StringVar(value="(no file)")
            self.file_vars.append(fv)
            ttk.Label(box, textvariable=fv, style="Panel.TLabel", width=26
                      ).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
            ttk.Button(box, text="Browse…",
                       command=lambda i=i: self.browse(i)).grid(row=1, column=2)
            ttk.Label(box, text="Carrier (kHz):", style="Panel.TLabel"
                      ).grid(row=2, column=0, sticky="w")
            v = tk.StringVar(value=f"{st.freq:.1f}")
            self.freq_vars.append(v)
            sp = ttk.Spinbox(box, from_=F_MIN, to=F_MAX, increment=0.5,
                             textvariable=v, width=8, format="%.1f",
                             command=lambda i=i: self.apply_freq(i))
            sp.grid(row=2, column=1, sticky="w")
            sp.bind("<Return>", lambda e, i=i: self.apply_freq(i))
            sp.bind("<FocusOut>", lambda e, i=i: self.apply_freq(i))

    # -- frequency readout, dial strip, knob
    def _build_receiver(self):
        mid = ttk.Frame(self.root, style="Panel.TFrame", padding=10)
        mid.pack(fill="x", padx=19, pady=6)
        left = ttk.Frame(mid, style="Panel.TFrame")
        left.pack(side="left", padx=(0, 12))
        self.knob = KnobDial(left, self)
        self.knob.pack()
        ttk.Label(left, text="drag knob · scroll = ±0.1 kHz · Shift+scroll = ±1",
                  style="Panel.TLabel", font=("Segoe UI", 8)).pack(pady=(2, 0))

        right = ttk.Frame(mid, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True)
        self.readout = tk.Label(right, text="", bg="#120f0b", fg=BRASS,
                                font=("Consolas", 34, "bold"), width=11,
                                relief="sunken", bd=3)
        self.readout.pack(fill="x")
        self.strip = tk.Canvas(right, width=380, height=70, bg="#120f0b",
                               highlightthickness=0)
        self.strip.pack(pady=8, fill="x")
        self.strip.bind("<Button-1>", self._strip_click)
        self.strip.bind("<B1-Motion>", self._strip_click)
        ttk.Label(right, text="Signal strength", style="Panel.TLabel").pack(anchor="w")
        self.meter = tk.Canvas(right, width=380, height=16, bg="#120f0b",
                               highlightthickness=0)
        self.meter.pack(fill="x")
        self.hearing = tk.StringVar()
        ttk.Label(right, textvariable=self.hearing, style="Panel.TLabel"
                  ).pack(anchor="w", pady=(6, 0))

    # -- power, mode, volume, static
    def _build_controls(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=14, pady=(6, 4))
        self.power_btn = ttk.Button(bar, text="⏻  Power: OFF", command=self.toggle_power)
        self.power_btn.grid(row=0, column=0, padx=4)

        ttk.Label(bar, text="Mode").grid(row=0, column=1, padx=(14, 4))
        self.mode_var = tk.StringVar(value=self.radio.mode)
        cb = ttk.Combobox(bar, textvariable=self.mode_var, values=("FM", "AM"),
                          width=4, state="readonly")
        cb.grid(row=0, column=2)
        cb.bind("<<ComboboxSelected>>", lambda e: setattr(
            self.radio, "mode", self.mode_var.get()))

        ttk.Label(bar, text="Volume").grid(row=0, column=3, padx=(14, 4))
        self.vol = ttk.Scale(bar, from_=0, to=1, length=100, value=self.radio.volume,
                             command=lambda v: setattr(self.radio, "volume", float(v)))
        self.vol.grid(row=0, column=4)
        ttk.Label(bar, text="Static").grid(row=0, column=5, padx=(14, 4))
        self.stat = ttk.Scale(bar, from_=0, to=1, length=100, value=self.radio.static,
                              command=lambda v: setattr(self.radio, "static", float(v)))
        self.stat.grid(row=0, column=6)

        self.show_markers = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show station markers", variable=self.show_markers,
                        command=self.refresh).grid(row=0, column=7, padx=(14, 0))

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, foreground=DIM
                  ).pack(anchor="w", padx=18, pady=(2, 12))

    # ---- actions
    def set_tune(self, f):
        self.radio.tune = float(np.clip(f, F_MIN, F_MAX))
        self.refresh()

    def _strip_click(self, e):
        w = max(self.strip.winfo_width(), 1)
        self.set_tune(round(F_MIN + (F_MAX - F_MIN) * np.clip(e.x / w, 0, 1), 1))

    def apply_freq(self, i):
        try:
            f = float(self.freq_vars[i].get())
        except ValueError:
            f = self.radio.stations[i].freq
        f = float(np.clip(f, F_MIN, F_MAX))
        self.radio.stations[i].freq = f
        self.freq_vars[i].set(f"{f:.1f}")
        a, b = (s.freq for s in self.radio.stations)
        if abs(a - b) < MIN_SPACING_KHZ:
            self.status.set(f"Warning: carriers are only {abs(a-b):.1f} kHz apart "
                            f"(< {MIN_SPACING_KHZ:.0f} kHz) – stations will bleed into each other.")
        self.refresh()

    def browse(self, i):
        p = filedialog.askopenfilename(
            initialdir=HERE, title=f"Audio for station {self.radio.stations[i].name}",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a"), ("All files", "*.*")])
        if p:
            self.load_file(i, p)

    def load_file(self, i, path):
        self.file_vars[i].set("loading…")
        self.status.set(f"Loading {os.path.basename(path)} …")

        def work():
            try:
                audio = load_audio(path)
            except Exception as ex:                       # noqa: BLE001
                self.root.after(0, lambda: (self.file_vars[i].set("(failed)"),
                                            messagebox.showerror("Load failed", str(ex))))
                return

            def done():
                self.radio.stations[i].audio = audio
                self.file_vars[i].set(os.path.basename(path))
                self.status.set(f"Loaded {os.path.basename(path)} "
                                f"({len(audio)/AUDIO_SR:.0f}s, loops).")
            self.root.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def toggle_power(self):
        if self.stream is None:
            try:
                self.stream = sd.OutputStream(
                    samplerate=AUDIO_SR, channels=1, blocksize=BLOCK,
                    dtype="float32", latency=0.15, callback=self._audio_cb)
                self.stream.start()
            except Exception as ex:                       # noqa: BLE001
                self.stream = None
                messagebox.showerror("Audio error", str(ex))
                return
            self.power_btn.config(text="⏻  Power: ON")
            self.status.set("Turn the knob to find the stations.")
        else:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.power_btn.config(text="⏻  Power: OFF")

    def _audio_cb(self, outdata, frames, time_info, status):
        outdata[:, 0] = self.radio.render(frames)

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        self.root.destroy()

    # ---- drawing
    def refresh(self):
        r = self.radio
        self.readout.config(text=f"{r.tune:7.1f} kHz")
        self.knob.redraw()
        s = self.strip
        s.delete("all")
        w = max(s.winfo_width(), 380)
        x = lambda f: (f - F_MIN) / (F_MAX - F_MIN) * (w - 20) + 10
        for k in range(int(F_MIN), int(F_MAX) + 1, 5):
            major = k % 10 == 0
            s.create_line(x(k), 40, x(k), 52 if major else 46, fill=DIM)
            if major:
                s.create_text(x(k), 60, text=str(k), fill=DIM, font=("Segoe UI", 8))
        if self.show_markers.get():
            for i, st in enumerate(r.stations):
                s.create_polygon(x(st.freq) - 5, 8, x(st.freq) + 5, 8, x(st.freq), 20,
                                 fill=STATION_COLORS[i])
                s.create_text(x(st.freq), 30, text=st.name, fill=STATION_COLORS[i],
                              font=("Segoe UI", 8, "bold"))
        s.create_line(x(r.tune), 4, x(r.tune), 56, fill=NEEDLE, width=2)

    def poll(self):
        r = self.radio
        m = self.meter
        m.delete("all")
        w = max(m.winfo_width(), 380)
        m.create_rectangle(0, 0, w * r.strength, 16,
                           fill=BRASS if r.strength < 0.85 else "#7ddc7a", width=0)
        # which station is the knob closest to?
        near = min(r.stations, key=lambda s: abs(s.freq - r.tune))
        off = r.tune - near.freq
        if r.q > 0.5:
            self.hearing.set(f"Hearing station {near.name}"
                             f"{'' if abs(off) < 0.5 else f'  (off by {off:+.1f} kHz)'}")
        elif r.q > 0.05:
            self.hearing.set(f"Fading in/out near station {near.name}…")
        else:
            self.hearing.set("Static")
        self.root.after(50, self.poll)


def main():
    root = tk.Tk()
    app = App(root)
    app.refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
