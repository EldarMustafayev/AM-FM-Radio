# Software Radio

A real-time software radio simulation. Two audio files are modulated (AM or FM)
onto carrier frequencies you choose, added together on a simulated "air" channel
with a little RF noise, and you tune between them with a knob, like an old radio.
Off-station you hear static.

## How it works

- **Transmitter** – each file is band-limited to 5 kHz and modulated (AM or FM) onto its own carrier.
- **Air** – both signals are summed, plus noise.
- **Receiver** – the knob frequency mixes the signal down to baseband, a channel filter selects the station, then it is demodulated.

The simulation runs at 6x the audio rate (264.6 kHz), so carriers live in the
20–120 kHz range. Dial units are simulated kHz, not real RF.

## Setup

Requires Python 3.9+ with Tkinter (included in the standard Windows installer).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python radio.py
```

## Usage

On launch, the app looks for `dakiti.mp3` and `alors_on_danse.mp3` next to
`radio.py`. These are not included in the repo; place your own audio files
there, or pick files in the GUI. Stations closer than 25 kHz apart overlap.

## Notes

Based on the experiment in `radio.ipynb` (carrier × audio → sum → recover),
streamed in real time. Audio and notebooks are git-ignored.
