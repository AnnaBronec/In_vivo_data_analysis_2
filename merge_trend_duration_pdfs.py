#!/usr/bin/env python3
"""
Sammelt alle UP-Dauer-Trend-Daten aus Unterordnern von ROOT_DIR,
zeichnet pro Parent-Ordner einen Trend-Plot (mit Foldernamen als Überschrift)
und speichert alles in eine einzige PDF unter ROOT_DIR.
"""

import os
import re
import glob
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = "/run/media/ananym/7FBA-F71B/invivo/hM4Di"
OUT_PDF  = os.path.join(ROOT_DIR, "ALL_trend_duration.pdf")
FIG_SIZE = (12, 5.5)
STAGGER_THRESHOLD = 8


def _wrap_label(label, width=22):
    """Lange Labels auf mehrere Zeilen umbrechen (an '/' bevorzugt, sonst nach Breite)."""
    parts = str(label).split("/")
    return "\n".join(textwrap.fill(p, width=width) for p in parts)


def _set_trend_xticklabels(ax, x, labels, fontsize=8):
    """Feste Größe pro Plot: Beschriftungen bei vielen Sessions auf 2 Zeilen staffeln,
    damit sie sich nicht überlappen statt die Plotbreite zu vergrößern."""
    n = len(labels)
    labels = [_wrap_label(lbl) for lbl in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if n > STAGGER_THRESHOLD else 40,
                        ha="right", fontsize=fontsize)
    if n > STAGGER_THRESHOLD:
        from matplotlib.transforms import offset_copy
        fig = ax.figure
        for i, tick in enumerate(ax.get_xticklabels()):
            if i % 2 == 1:
                tick.set_transform(offset_copy(tick.get_transform(), fig=fig, x=0, y=-12, units="points"))


def _nat_session_key(sess_name):
    m = re.match(r"^\s*(\d+)", str(sess_name))
    if m:
        return (0, int(m.group(1)), str(sess_name).lower())
    return (1, str(sess_name).lower())


def _robust_filter(arr, thresh=5.0):
    """Schließt extreme Ausreißer (z.B. Fehldetektionen) per MAD-Kriterium aus,
    damit ein einzelner Artefaktwert nicht Mittelwert/Fehlerbalken des Trends verzerrt."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return arr
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return arr
    modified_z = 0.6745 * (arr - med) / mad
    return arr[np.abs(modified_z) <= thresh]


def load_trend_rows(parent_dir):
    """Lädt pro Session den mittleren Dauer-Wert (spont & trig)."""
    session_dirs = sorted(
        [d for d in os.scandir(parent_dir) if d.is_dir()],
        key=lambda e: _nat_session_key(e.name)
    )
    rows = []
    for entry in session_dirs:
        dur_files = sorted(glob.glob(os.path.join(entry.path, "*__upstate_durations.csv")), key=os.path.getmtime)
        spont = np.array([], dtype=float)
        trig  = np.array([], dtype=float)
        if dur_files:
            try:
                dfm = pd.read_csv(dur_files[-1])
                st = (dfm.get("group", pd.Series([], dtype=str))
                         .astype(str).str.lower().str.strip()
                         .replace({"spont": "spontaneous", "trig": "triggered",
                                   "trigger": "triggered"}))
                dur = pd.to_numeric(
                    dfm.get("duration_s", pd.Series([], dtype=float)), errors="coerce"
                ).to_numpy(float)
                spont = dur[(st.to_numpy() == "spontaneous") & np.isfinite(dur)]
                trig  = dur[(st.to_numpy() == "triggered")   & np.isfinite(dur)]
            except Exception:
                pass
        spont_clean = _robust_filter(spont)
        trig_clean  = _robust_filter(trig)
        rows.append({
            "session_name": entry.name,
            "spont_mean": float(np.nanmean(spont_clean)) if spont_clean.size else np.nan,
            "trig_mean":  float(np.nanmean(trig_clean))  if trig_clean.size  else np.nan,
            "spont_std":  float(np.nanstd(spont_clean))  if spont_clean.size else np.nan,
            "trig_std":   float(np.nanstd(trig_clean))   if trig_clean.size  else np.nan,
            "spont_raw":  spont.tolist(),
            "trig_raw":   trig.tolist(),
        })
    rows = [r for r in rows if np.isfinite(r["spont_mean"]) or np.isfinite(r["trig_mean"])]
    return rows


def plot_trend(ax, rows, folder_name, ylim=None):
    labels     = [r["session_name"] for r in rows]
    x          = np.arange(len(labels))
    spont_vals = np.array([r["spont_mean"] for r in rows])
    trig_vals  = np.array([r["trig_mean"]  for r in rows])
    spont_stds = np.array([r.get("spont_std", np.nan) for r in rows])
    trig_stds  = np.array([r.get("trig_std",  np.nan) for r in rows])

    sp_ok = np.isfinite(spont_vals)
    tr_ok = np.isfinite(trig_vals)

    for i, r in enumerate(rows):
        raw_sp = r.get("spont_raw", [])
        raw_tr = r.get("trig_raw",  [])
        if raw_sp:
            ax.scatter([i] * len(raw_sp), raw_sp,
                       color="#4C78A8", alpha=0.25, s=12, zorder=2, linewidths=0)
        if raw_tr:
            ax.scatter([i] * len(raw_tr), raw_tr,
                       color="#F58518", alpha=0.25, s=12, zorder=2, linewidths=0)

    if sp_ok.any():
        ax.errorbar(x[sp_ok], spont_vals[sp_ok], yerr=spont_stds[sp_ok],
                    color="#4C78A8", marker="o", linewidth=1.5, markersize=8,
                    capsize=3, label="Spontan", zorder=3)
    if tr_ok.any():
        ax.errorbar(x[tr_ok], trig_vals[tr_ok], yerr=trig_stds[tr_ok],
                    color="#F58518", marker="o", linewidth=1.5, markersize=8,
                    capsize=3, label="Getriggert", zorder=3)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        all_raw = np.array([v for r in rows
                            for v in r.get("spont_raw", []) + r.get("trig_raw", [])
                            if np.isfinite(v)])
        if all_raw.size:
            vmax = float(np.nanpercentile(all_raw, 99))
            vmin = float(np.nanmin(all_raw))
            span = max(vmax - vmin, 1e-6)
            ax.set_ylim(min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)

    _set_trend_xticklabels(ax, x, labels)
    ax.set_ylabel("Dauer (s)")
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
        else:
            combined_rows = []
            for sub_path in sorted([e.path for e in os.scandir(pd_path) if e.is_dir()],
                                   key=lambda p: _nat_session_key(os.path.basename(p))):
                sub_rows = load_trend_rows(sub_path)
                if sub_rows:
                    sub_name = os.path.basename(sub_path)
                    for r in sub_rows:
                        r["session_name"] = f"{sub_name}/{r['session_name']}"
                    combined_rows.extend(sub_rows)
            if combined_rows:
                entries.append((os.path.basename(pd_path), combined_rows))

    if not entries:
        print("Keine Dauer-Daten gefunden.")
        return

    print(f"Gefundene Parent-Ordner mit Daten: {len(entries)}")
    for name, _ in entries:
        print(f"  {name}")

    # Globales y-Limit berechnen
    all_vals = np.array([v for _, rows in entries
                         for r in rows
                         for v in r.get("spont_raw", []) + r.get("trig_raw", [])
                         if np.isfinite(v)])
    if all_vals.size:
        vmax = float(np.nanpercentile(all_vals, 99))
        vmin = float(np.nanmin(all_vals))
        span = max(vmax - vmin, 1e-6)
        global_ylim = (min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)
    else:
        global_ylim = None

    with PdfPages(OUT_PDF) as pdf:
        for folder_name, rows in entries:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            plot_trend(ax, rows, folder_name, ylim=global_ylim)
            fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.32)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n[DONE] PDF gespeichert: {OUT_PDF}")


if __name__ == "__main__":
    main()
