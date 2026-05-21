#!/usr/bin/env python3
"""
Sammelt alle UP-Amplitude-Trend-Daten aus Unterordnern von ROOT_DIR,
zeichnet pro Parent-Ordner einen Trend-Plot (mit Foldernamen als Überschrift)
und speichert alles in eine einzige PDF unter ROOT_DIR.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR  = "/run/media/ananym/7FBA-F71B/invivo/hM4Di"
OUT_PDF   = os.path.join(ROOT_DIR, "ALL_trend_amplitude.pdf")
UNIT_LABEL = "µV/mm²"


def _nat_session_key(sess_name):
    m = re.match(r"^\s*(\d+)", str(sess_name))
    if m:
        return (0, int(m.group(1)), str(sess_name).lower())
    return (1, str(sess_name).lower())


def load_trend_rows(parent_dir):
    """Lädt pro Session den mittleren Amplitude-Wert (spont & trig)."""
    session_dirs = sorted(
        [d for d in os.scandir(parent_dir) if d.is_dir()],
        key=lambda e: _nat_session_key(e.name)
    )
    rows = []
    for entry in session_dirs:
        amp_files = sorted(glob.glob(os.path.join(entry.path, "*__upstate_amplitudes.csv")))
        spont = np.array([], dtype=float)
        trig  = np.array([], dtype=float)
        if amp_files:
            try:
                dfm = pd.read_csv(amp_files[-1])
                st = (dfm.get("group", pd.Series([], dtype=str))
                         .astype(str).str.lower().str.strip()
                         .replace({"spont": "spontaneous", "trig": "triggered",
                                   "trigger": "triggered"}))
                amp = pd.to_numeric(
                    dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce"
                ).to_numpy(float)
                spont = amp[(st.to_numpy() == "spontaneous") & np.isfinite(amp)]
                trig  = amp[(st.to_numpy() == "triggered")   & np.isfinite(amp)]
            except Exception:
                pass
        rows.append({
            "session_name": entry.name,
            "spont_mean": float(np.nanmean(spont)) if spont.size else np.nan,
            "trig_mean":  float(np.nanmean(trig))  if trig.size  else np.nan,
        })
    # nur Zeilen mit mindestens einem Wert
    rows = [r for r in rows if np.isfinite(r["spont_mean"]) or np.isfinite(r["trig_mean"])]
    return rows


def plot_trend(ax, rows, folder_name):
    labels     = [r["session_name"] for r in rows]
    x          = np.arange(len(labels))
    spont_vals = np.array([r["spont_mean"] for r in rows])
    trig_vals  = np.array([r["trig_mean"]  for r in rows])

    sp_ok = np.isfinite(spont_vals)
    tr_ok = np.isfinite(trig_vals)

    if sp_ok.any():
        ax.plot(x[sp_ok], spont_vals[sp_ok],
                color="#4C78A8", marker="o", linewidth=1.5, markersize=7, label="Spontan")
    if tr_ok.any():
        ax.plot(x[tr_ok], trig_vals[tr_ok],
                color="#F58518", marker="o", linewidth=1.5, markersize=7, label="Getriggert")

    all_vals = np.concatenate([spont_vals[sp_ok], trig_vals[tr_ok]])
    if all_vals.size:
        vmax = float(np.nanpercentile(all_vals, 99))
        vmin = float(np.nanmin(all_vals))
        span = max(vmax - vmin, 1e-6)
        ax.set_ylim(min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(f"Mean Amplitude ({UNIT_LABEL})")
    ax.set_title(folder_name, fontsize=12, fontweight="bold", pad=8)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(fontsize=9)


def main():
    parent_dirs = sorted(
        [e.path for e in os.scandir(ROOT_DIR) if e.is_dir()],
        key=lambda p: _nat_session_key(os.path.basename(p))
    )

    entries = []
    for pd_path in parent_dirs:
        rows = load_trend_rows(pd_path)
        if rows:
            entries.append((os.path.basename(pd_path), rows))

    if not entries:
        print("Keine Amplitude-Daten gefunden.")
        return

    print(f"Gefundene Parent-Ordner mit Daten: {len(entries)}")
    for name, _ in entries:
        print(f"  {name}")

    with PdfPages(OUT_PDF) as pdf:
        for folder_name, rows in entries:
            fig_w = max(9, len(rows) * 0.9)
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            plot_trend(ax, rows, folder_name)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\n[DONE] PDF gespeichert: {OUT_PDF}")


if __name__ == "__main__":
    main()
