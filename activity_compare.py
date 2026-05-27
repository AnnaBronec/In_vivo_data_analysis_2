#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
activity_compare.py
-------------------
Vergleicht die mittlere LFP-Breitband-Aktivität (RMS der guten Kanäle)
über Bedingungs-Unterordner für mehrere Tiere.

Erwartete Ordnerstruktur:
  <root>/
    <tier_1>/
      BS/          <- Baseline
        <BS>.csv
        runlog.txt   (optional, aber empfohlen)
      <bedingung_2>/
        <cond2>.csv
    <tier_2>/
      ...

Ausgabe:
  <root>/activity_compare.csv
  <root>/activity_compare.pdf

Aufruf:
  python activity_compare.py /pfad/zum/root-ordner
  python activity_compare.py  # nutzt Standard-Pfad aus Code
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Spalten im LFP-CSV die keine Kanaldaten sind
_NON_CH = {
    "time", "timestamps", "timesamples",
    "stim", "stim_on", "stim_off",
    "din_1", "din_2", "din1", "din2",
    "startstop", "ttl", "di0", "di1",
}

# Regex um good_idx aus runlog.txt zu extrahieren
_RUNLOG_RE = re.compile(
    r"Channel filter:.*?good_idx=\[([0-9, ]+)\]"
)

# Maximale Anzahl Samples für RMS (subsampling für Speichereffizienz)
_MAX_SAMPLES_RMS = 500_000


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _parse_good_idx_from_runlog(folder: Path) -> list[int] | None:
    """Liest den letzten good_idx-Eintrag aus runlog.txt (falls vorhanden)."""
    runlog = folder / "runlog.txt"
    if not runlog.is_file():
        return None
    try:
        text = runlog.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    # Alle Treffer suchen, letzten nehmen (= letzter Analyse-Lauf)
    matches = _RUNLOG_RE.findall(text)
    if not matches:
        return None
    idx_str = matches[-1]  # letzter Lauf
    try:
        return [int(x.strip()) for x in idx_str.split(",") if x.strip()]
    except Exception:
        return None


def _find_lfp_csv(folder: Path) -> Path | None:
    """Findet die LFP-CSV im Ordner. Bevorzugt <Ordnername>.csv."""
    skip = {"upstate_summary.csv", "runlog.txt", "activity_compare.csv"}
    # 1. Bevorzugt: Datei heißt wie der Ordner
    named = folder / f"{folder.name}.csv"
    if named.is_file():
        return named
    # 2. Beliebige andere CSV (keine Hilfsdateien)
    for p in sorted(folder.glob("*.csv")):
        if p.name.lower() not in {s.lower() for s in skip}:
            return p
    return None


def _load_lfp_array(csv_path: Path):
    """Lädt LFP-CSV und gibt (data_array [n_ch x n_t], fs_est) zurück."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Zeitspalte normalisieren
    if "time" not in df.columns:
        for alt in ("timesamples", "timestamps"):
            if alt in df.columns:
                df = df.rename(columns={alt: "time"})
                break

    if "time" not in df.columns:
        return None, None

    ch_cols = [c for c in df.columns if c.lower() not in _NON_CH]
    if not ch_cols:
        return None, None

    # Samplingrate schätzen
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    fs_est = None
    if len(time) > 5:
        dt = float(np.nanmedian(np.diff(time)))
        if np.isfinite(dt) and dt > 0:
            fs_est = 1.0 / dt

    # Datenmatrix aufbauen
    data = np.array(
        [pd.to_numeric(df[c], errors="coerce").to_numpy() for c in ch_cols],
        dtype=np.float32,
    )
    return data, fs_est


def _find_good_channels_auto(data: np.ndarray) -> list[int]:
    """
    Einfacher Kanalqualitätsfilter (vereinfachte Version aus Main_safe.py).
    Entfernt tote, rauschige und artifaktbehaftete Kanäle.
    """
    n_ch = data.shape[0]
    stds = np.full(n_ch, np.nan)

    for i in range(n_ch):
        x = data[i]
        finite_mask = np.isfinite(x)
        if finite_mask.mean() < 0.90:
            continue
        stds[i] = float(np.nanstd(x))

    valid = np.isfinite(stds) & (stds > 0)
    if not valid.any():
        return list(range(n_ch))
    std_med = float(np.nanmedian(stds[valid]))

    good = []
    for i in range(n_ch):
        x = data[i]
        finite_frac = float(np.isfinite(x).mean())
        if finite_frac < 0.90:
            continue
        s = stds[i]
        if not np.isfinite(s) or s == 0:
            continue
        if std_med > 0 and np.isfinite(std_med):
            rel = s / std_med
            if rel < 0.15 or rel > 5.0:
                continue
        med = float(np.nanmedian(x))
        art_frac = float(np.mean(np.abs(x - med) / s > 7.0))
        if art_frac > 0.02:
            continue
        good.append(i)

    return good if good else list(range(n_ch))


def _compute_rms(data: np.ndarray, good_idx: list[int]) -> float:
    """Mittlerer Breitband-RMS über alle guten Kanäle."""
    rms_vals = []
    for i in good_idx:
        x = data[i]
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        # Subsampling für Speichereffizienz
        if len(x) > _MAX_SAMPLES_RMS:
            step = len(x) // _MAX_SAMPLES_RMS
            x = x[::step]
        rms_vals.append(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
    return float(np.mean(rms_vals)) if rms_vals else np.nan


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_sessions(root: Path) -> list[tuple[str, str, Path]]:
    """
    Sucht alle (tier_name, condition_name, folder_path) Tripel.
    Erwartet: root/<tier>/<bedingung>/
    """
    sessions = []
    for tier_dir in sorted(root.iterdir()):
        if not tier_dir.is_dir() or tier_dir.name.startswith("."):
            continue
        for cond_dir in sorted(tier_dir.iterdir()):
            if not cond_dir.is_dir() or cond_dir.name.startswith("."):
                continue
            sessions.append((tier_dir.name, cond_dir.name, cond_dir))
    return sessions


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def run(root: Path):
    print(f"[activity_compare] Root: {root}")
    sessions = discover_sessions(root)
    if not sessions:
        print(f"[WARN] Keine Sessions gefunden unter {root}")
        return

    records = []
    for tier, cond, folder in sessions:
        print(f"  {tier}/{cond} ... ", end="", flush=True)

        csv_path = _find_lfp_csv(folder)
        if csv_path is None:
            print("übersprungen (keine CSV gefunden)")
            continue

        data, fs = _load_lfp_array(csv_path)
        if data is None:
            print("übersprungen (CSV konnte nicht geladen werden)")
            continue

        # Good channels: bevorzugt aus runlog.txt
        good_idx = _parse_good_idx_from_runlog(folder)
        source = "runlog"
        if good_idx is None or not good_idx:
            good_idx = _find_good_channels_auto(data)
            source = "auto"
        # Indizes auf gültigen Bereich beschränken
        good_idx = [i for i in good_idx if 0 <= i < data.shape[0]]
        if not good_idx:
            print(f"übersprungen (0 gute Kanäle von {data.shape[0]})")
            continue

        rms = _compute_rms(data, good_idx)
        n_t = data.shape[1]
        duration_s = n_t / fs if (fs and np.isfinite(fs)) else np.nan

        print(
            f"RMS={rms:.2f}  "
            f"({len(good_idx)}/{data.shape[0]} gute Kanäle, "
            f"Quelle={source}, "
            f"Dauer={duration_s:.0f}s)"
        )

        records.append({
            "tier": tier,
            "bedingung": cond,
            "rms_mittel_uV": rms,
            "n_gute_kanaele": len(good_idx),
            "n_kanaele_gesamt": data.shape[0],
            "gute_kanaele_quelle": source,
            "dauer_s": duration_s,
            "fs_hz": fs,
            "csv": str(csv_path.relative_to(root) if csv_path.is_relative_to(root) else csv_path),
        })

        del data  # Speicher freigeben

    if not records:
        print("[FEHLER] Keine gültigen Sessions verarbeitet.")
        return

    df = pd.DataFrame(records)

    # Normierung auf BS pro Tier
    bs_rms_map: dict[str, float] = {}
    for tier, grp in df.groupby("tier"):
        bs_rows = grp[grp["bedingung"].str.upper() == "BS"]
        if not bs_rows.empty:
            bs_rms_map[str(tier)] = float(bs_rows["rms_mittel_uV"].iloc[0])

    df["rms_norm_bs"] = df.apply(
        lambda r: r["rms_mittel_uV"] / bs_rms_map[r["tier"]]
        if r["tier"] in bs_rms_map and bs_rms_map[r["tier"]] > 0
        else np.nan,
        axis=1,
    )

    # CSV speichern
    out_csv = root / "activity_compare.csv"
    df.to_csv(out_csv, index=False, sep=";", decimal=",", float_format="%.4f")
    print(f"\n[OUT] CSV: {out_csv}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    # Bedingungen sortiert: BS zuerst, dann alphabetisch
    conditions = sorted(
        df["bedingung"].unique(),
        key=lambda c: (c.upper() != "BS", c.upper()),
    )
    tiere = sorted(df["tier"].unique())
    n_cond = len(conditions)
    n_tier = len(tiere)

    cmap = plt.cm.get_cmap("tab10", n_tier)
    colors = [cmap(i) for i in range(n_tier)]

    bar_w = 0.8 / max(n_tier, 1)
    offsets = np.linspace(
        -(n_tier - 1) / 2 * bar_w,
        (n_tier - 1) / 2 * bar_w,
        n_tier,
    )
    x = np.arange(n_cond)

    out_pdf = root / "activity_compare.pdf"
    with PdfPages(out_pdf) as pdf:

        # --- Plot 1: absoluter RMS ---
        fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.8 + 1), 5))
        for ai, (tier, color) in enumerate(zip(tiere, colors)):
            vals = []
            for cond in conditions:
                row = df[(df["tier"] == tier) & (df["bedingung"] == cond)]
                vals.append(float(row["rms_mittel_uV"].iloc[0]) if not row.empty else np.nan)
            bars = ax.bar(x + offsets[ai], vals, bar_w * 0.9, label=tier, color=color)
            # Wert über den Balken
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_ylabel("Mittlerer RMS (gute Kanäle) [µV]", fontsize=11)
        ax.set_title("Breitband-RMS pro Bedingung (absolut)", fontsize=13)
        ax.legend(title="Tier", fontsize=9)
        ax.set_xlim(-0.5, n_cond - 0.5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Plot 2: RMS normiert auf BS ---
        if bs_rms_map:
            fig, ax = plt.subplots(figsize=(max(6, n_cond * 1.8 + 1), 5))
            for ai, (tier, color) in enumerate(zip(tiere, colors)):
                vals = []
                for cond in conditions:
                    row = df[(df["tier"] == tier) & (df["bedingung"] == cond)]
                    vals.append(float(row["rms_norm_bs"].iloc[0]) if not row.empty else np.nan)
                bars = ax.bar(x + offsets[ai], vals, bar_w * 0.9, label=tier, color=color)
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01,
                            f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7,
                        )

            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="BS = 1")
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, fontsize=11)
            ax.set_ylabel("RMS normiert auf BS", fontsize=11)
            ax.set_title("Breitband-RMS pro Bedingung (normiert auf BS)", fontsize=13)
            ax.legend(title="Tier", fontsize=9)
            ax.set_xlim(-0.5, n_cond - 0.5)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Plot 3: Übersichtstabelle ---
        fig, ax = plt.subplots(figsize=(max(8, n_cond * 1.5 + 2), max(3, n_tier * 0.5 + 2)))
        ax.axis("off")
        tbl_cols = ["Tier"] + conditions
        tbl_data = []
        for tier in tiere:
            row_vals = [tier]
            for cond in conditions:
                match = df[(df["tier"] == tier) & (df["bedingung"] == cond)]
                if match.empty:
                    row_vals.append("—")
                else:
                    rms = match["rms_mittel_uV"].iloc[0]
                    norm = match["rms_norm_bs"].iloc[0]
                    if np.isfinite(norm):
                        row_vals.append(f"{rms:.1f} µV\n({norm:.2f}×)")
                    else:
                        row_vals.append(f"{rms:.1f} µV")
            tbl_data.append(row_vals)

        tbl = ax.table(
            cellText=tbl_data,
            colLabels=tbl_cols,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.8)
        ax.set_title("RMS-Übersicht (absolut [µV] + normiert auf BS)", fontsize=11, pad=20)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[OUT] PDF: {out_pdf}")
    print(f"[FERTIG] {len(records)} Sessions verarbeitet.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO"


def main():
    parser = argparse.ArgumentParser(description="Vergleicht Breitband-RMS über Bedingungen.")
    parser.add_argument(
        "root",
        nargs="?",
        default=_DEFAULT_ROOT,
        help=f"Wurzelordner mit <tier>/<bedingung>/-Struktur (Standard: {_DEFAULT_ROOT})",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[FEHLER] Ordner nicht gefunden: {root}", file=sys.stderr)
        sys.exit(1)

    run(root)


if __name__ == "__main__":
    main()
