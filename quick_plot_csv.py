#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot aus combined CSV (nutzt stim-Spalte für Pulse).")
    ap.add_argument("csv", help="Pfad zur *_combined.csv")
    ap.add_argument("--channel", help="Kanalname (z.B. CSC1). Default: erster Kanal", default=None)
    ap.add_argument("--channel-index", type=int, help="Index des Kanals (0-basiert), falls kein Name angegeben ist", default=None)
    ap.add_argument("--tmin", type=float, help="Zeitfenster Start (s)", default=None)
    ap.add_argument("--tmax", type=float, help="Zeitfenster Ende (s)", default=None)
    ap.add_argument("--downsample", type=int, help="Nur jedes n-te Sample plotten (schneller)", default=1)
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        sys.exit(f"CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)

    # Zeitspalte finden
    time_col = "time" if "time" in df.columns else ("timesamples" if "timesamples" in df.columns else None)
    if time_col is None:
        raise KeyError("Keine Zeitspalte (erwarte 'time' oder 'timesamples').")
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    # Kanäle bestimmen (alles außer Zeit und stim)
    chan_cols = [c for c in df.columns if c not in (time_col, "stim")]
    if not chan_cols:
        raise RuntimeError("Keine Kanalspalten gefunden.")

    # Kanal auswählen
    if args.channel and args.channel in chan_cols:
        ch = args.channel
    elif args.channel_index is not None:
        if args.channel_index < 0 or args.channel_index >= len(chan_cols):
            sys.exit(f"channel-index außerhalb des Bereichs (0..{len(chan_cols)-1}).")
        ch = chan_cols[args.channel_index]
    else:
        ch = chan_cols[0]

    y = pd.to_numeric(df[ch], errors="coerce").to_numpy()

    # Stimuli aus 'stim' -> Rising Edges
    pulses_s = np.array([], dtype=float)
    if "stim" in df.columns:
        stim = pd.to_numeric(df["stim"], errors="coerce").fillna(0).astype(int).to_numpy()
        rising = (np.diff(stim, prepend=0) == 1)
        pulses_s = time[rising]
    else:
        print("⚠️ Keine 'stim'-Spalte gefunden – es werden keine Pulse gezeichnet.")

    # Zeitfenster anwenden (optional)
    if args.tmin is not None or args.tmax is not None:
        tmin = time.min() if args.tmin is None else args.tmin
        tmax = time.max() if args.tmax is None else args.tmax
        win_mask = (time >= tmin) & (time <= tmax)
        time = time[win_mask]
        y = y[win_mask]
        if pulses_s.size:
            pulses_s = pulses_s[(pulses_s >= tmin) & (pulses_s <= tmax)]

    # Downsample (optional, einfaches Striding)
    ds = max(1, int(args.downsample))
    if ds > 1:
        time = time[::ds]
        y = y[::ds]
        # Pulse werden als Linien gezeichnet; kein DS nötig

    # Plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(time, y, lw=0.9, color="black", label=ch)

    if pulses_s.size:
        # sicherstellen, dass nur Pulszeiten im sichtbaren Bereich liegen
        p = pulses_s[(pulses_s >= time.min()) & (pulses_s <= time.max())]
        if p.size:
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                pad = 0.05 * (ymax - ymin)
            else:
                pad = 1.0
            ax.vlines(p, ymin=ymin - pad, ymax=ymax + pad,
                      color="red", linestyles="--", alpha=0.7, label="stim")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (a.u.)")
    ax.set_title(f"{os.path.basename(csv_path)} — {ch}")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
