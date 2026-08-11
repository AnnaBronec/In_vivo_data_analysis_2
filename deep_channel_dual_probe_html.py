#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay one deep Port-A channel and one deep Port-C channel from a dual-probe
(contralateral) session in a single interactive HTML, each with its own
independently detected UP-states shaded (own spectrogram + classify_states
per channel -- same functions/config Main_safe.py uses for its main channel).

Only reads the two selected raw channel columns (+ din_1/din_2 if present)
from the session CSV, so it is cheap even for large multi-channel files, and
does not touch/overwrite any of Main_safe.py's own pipeline outputs.

Port/depth convention (matches Main_safe.py's existing "deeper channel"
logic, e.g. the pri_33/pri_38 defaults and the good_idx "as far down as
possible" selection): within a probe's raw channel block, a HIGHER raw
channel index ("pri_N"/"chNN") = a deeper electrode. Port A/C ranges and
which raw indices actually carry a connected probe are read from the
session's *.xdat.json (biointerface_map) rather than hard-coded, so this
works for any dual-probe session using the same Allego/XDAT metadata.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from TimeFreq_plot import Run_spectrogram
from state_detection import classify_states
from preprocessing import filtering
from processing import _upstate_amplitudes, crop_up_intervals
from Exports import export_interactive_two_channel_lfp_html

DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF = 2
DEFAULT_FS_XDAT = 32000.0
FIXED_ALIGN_PRE_S = 0.5
FIXED_ALIGN_POST_S = 0.5
ARTIFACT_MAD_K = 20.0
ARTIFACT_DERIV_MAD_K = 20.0
ARTIFACT_PAD_S = 0.15


# --- copied verbatim from Main_safe.py (the local defs there shadow the same-named
# state_detection.py imports, so this is the version actually used at runtime) ---

def detect_artifact_windows(x, dt, mad_k=20.0, deriv_mad_k=20.0, pad_s=0.15, merge_gap_s=0.1):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or not np.isfinite(dt) or dt <= 0:
        return np.zeros(n, dtype=bool), []

    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.zeros(n, dtype=bool), []

    raw_mask = np.zeros(n, dtype=bool)

    med_x = float(np.nanmedian(x[finite]))
    mad_x = float(np.nanmedian(np.abs(x[finite] - med_x)))
    sigma_x = mad_x / 0.6745 if mad_x > 1e-12 else float(np.nanstd(x[finite]))
    if np.isfinite(sigma_x) and sigma_x > 1e-12:
        raw_mask |= finite & (np.abs(x - med_x) > (mad_k * sigma_x))

    d = np.diff(x)
    d_finite = finite[:-1] & finite[1:]
    if d_finite.any():
        med_d = float(np.nanmedian(d[d_finite]))
        mad_d = float(np.nanmedian(np.abs(d[d_finite] - med_d)))
        sigma_d = mad_d / 0.6745 if mad_d > 1e-12 else float(np.nanstd(d[d_finite]))
        if np.isfinite(sigma_d) and sigma_d > 1e-12:
            raw_mask_d = d_finite & (np.abs(d - med_d) > (deriv_mad_k * sigma_d))
            if raw_mask_d.any():
                jump_idx = np.flatnonzero(raw_mask_d)
                raw_mask[jump_idx] = True
                raw_mask[jump_idx + 1] = True

    if not raw_mask.any():
        return np.zeros(n, dtype=bool), []

    pad = max(1, int(round(pad_s / dt)))
    mask = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(raw_mask):
        mask[max(0, i - pad):min(n, i + pad + 1)] = True

    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[0::2], edges[1::2]

    merge_gap = max(0, int(round(merge_gap_s / dt)))
    merged_starts, merged_ends = [], []
    for s, e in zip(starts, ends):
        if merged_ends and (s - merged_ends[-1]) <= merge_gap:
            merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    mask[:] = False
    index_intervals = []
    for s, e in zip(merged_starts, merged_ends):
        mask[s:e] = True
        index_intervals.append((int(s), int(e)))

    return mask, index_intervals


def interpolate_masked(x, mask):
    x = np.asarray(x, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return x
    good = ~mask
    if good.sum() < 2:
        return x
    idx = np.arange(x.size)
    x[mask] = np.interp(idx[mask], idx[good], x[good])
    return x


# --- session-config / channel-geometry helpers ---

def _load_shared_analysis_config():
    """Mirror Main_safe.py's _bootstrap_analysis_config env loading
    (Source/analysis_config.env), so UP_* thresholds match whatever the
    main pipeline is currently configured with."""
    cfg_path = Path(__file__).resolve().with_name("analysis_config.env")
    if not cfg_path.is_file():
        print(f"[CONFIG][WARN] {cfg_path} not found -- using state_detection.py built-in defaults")
        return
    for raw in cfg_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v
    print(f"[CONFIG] loaded {cfg_path}")


def _deepest_probe_channel(xdat_json_path, port):
    with open(xdat_json_path) as f:
        meta = json.load(f)
    bm = meta["sapiens_base"]["biointerface_map"]
    probe_ids = bm.get("probe_id", [None] * len(bm["chan_name"]))
    idxs = [
        int(bm["ntv_chan_idx"][i])
        for i in range(len(bm["chan_name"]))
        if bm["chan_name"][i].startswith("pri")
        and bm["port"][i] == port
        and probe_ids[i] not in (None, "no-probe")
    ]
    if not idxs:
        raise SystemExit(f"No connected probe channels found for port {port} in {xdat_json_path}")
    return max(idxs)  # higher raw index = deeper electrode (pipeline convention)


def _rising_falling(x, t):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return np.array([], float), np.array([], float)
    lo, hi = np.nanpercentile(x, [10, 90])
    thr = (lo + hi) * 0.5
    b = (x > thr).astype(np.int8)
    rising = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
    falling = np.flatnonzero((b[1:] == 0) & (b[:-1] == 1)) + 1
    return t[rising], t[falling]


def _detect_states_for_channel(channel_raw, time_s, dt, ch_label,
                                pulse_times_1, pulse_times_2,
                                pulse_times_1_off, pulse_times_2_off):
    channel = np.asarray(channel_raw, dtype=float).copy()

    # Artifact rejection identical to Main_safe.py: detect on a lowpass-smoothed
    # copy (raw simple-stride downsampling has no anti-alias filter, which
    # inflates robust sigma and would mask real artifacts), then interpolate
    # the resulting mask over the real (unsmoothed) channel used for analysis.
    sos_art = signal.butter(5, HIGH_CUTOFF, btype="low", fs=1.0 / dt, output="sos")
    finite = np.isfinite(channel)
    smooth_in = np.nan_to_num(channel, nan=float(np.nanmedian(channel)))
    smoothed = signal.sosfiltfilt(sos_art, smooth_in)
    smoothed[~finite] = np.nan
    mask, intervals = detect_artifact_windows(
        smoothed, dt, mad_k=ARTIFACT_MAD_K, deriv_mad_k=ARTIFACT_DERIV_MAD_K, pad_s=ARTIFACT_PAD_S
    )
    if mask.any():
        channel = interpolate_masked(channel, mask)
        print(f"[{ch_label}] artifact-interpolated {int(mask.sum())} samples in {len(intervals)} window(s)")

    Spect_dat = Run_spectrogram(channel, time_s)
    b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)
    align_pre = int(round(FIXED_ALIGN_PRE_S / dt))
    align_post = int(round(FIXED_ALIGN_POST_S / dt))
    align_len = align_pre + align_post

    Up = classify_states(
        Spect_dat, time_s, pulse_times_1, pulse_times_2, dt,
        channel, channel[None, :], b_lp, a_lp, b_hp, a_hp,
        align_pre, align_post, align_len,
        pulse_times_1_off=pulse_times_1_off, pulse_times_2_off=pulse_times_2_off,
    )
    print(
        f"[{ch_label}] states: spont={len(Up['Spontaneous_UP'])} "
        f"trig={len(Up['Pulse_triggered_UP'])} assoc={len(Up['Pulse_associated_UP'])}"
    )
    return channel, Up


def _write_amplitude_csv(channel_uV, Up, dt, out_csv_path):
    """Same amplitude definition Main_safe.py uses for its main channel
    (p95-p5 of the steady-state 0.3s-1.0s window after UP onset)."""
    spon_up_crop, spon_down_crop = crop_up_intervals(
        Up["Spontaneous_UP"], Up["Spontaneous_DOWN"], dt, start_s=0.3, end_s=1.0
    )
    trig_up_crop, trig_down_crop = crop_up_intervals(
        Up["Pulse_triggered_UP"], Up["Pulse_triggered_DOWN"], dt, start_s=0.3, end_s=1.0
    )
    spont_amp = _upstate_amplitudes(channel_uV, spon_up_crop, spon_down_crop)
    trig_amp = _upstate_amplitudes(channel_uV, trig_up_crop, trig_down_crop)
    amp_df = pd.DataFrame({
        "group": (["spontaneous"] * len(spont_amp)) + (["triggered"] * len(trig_amp)),
        "amplitude": np.concatenate([spont_amp, trig_amp]) if (len(spont_amp) or len(trig_amp)) else np.array([], float),
    })
    amp_df.to_csv(out_csv_path, index=False)
    print(f"[CSV] {out_csv_path}  (spont={len(spont_amp)}, trig={len(trig_amp)})")
    return out_csv_path


def _update_deep_channel_amplitude_trend_pdf(base_path, port_label, glob_tag, out_pdf_name):
    """Rebuild the parent-level amplitude trend PDF for one deep-probe channel
    across all sibling session folders, mirroring processing.py's
    _write_parent_up_amplitude_trend_pdf but scoped to this port's own
    amplitude CSVs (glob_tag), so Port A and Port C each get their own plot
    instead of being mixed with -- or overwritten by -- the main-channel one.
    """
    parent_dir = base_path.parent

    def _nat_session_key(sess_dir):
        m = re.match(r"^\s*(\d+)", os.path.basename(sess_dir))
        if m:
            return (0, int(m.group(1)), os.path.basename(sess_dir).lower())
        return (1, os.path.basename(sess_dir).lower())

    session_dirs = sorted(
        {os.path.dirname(p) for p in glob.glob(os.path.join(str(parent_dir), "*", glob_tag))},
        key=_nat_session_key,
    )
    if not session_dirs:
        return

    rows = []
    for sess_dir in session_dirs:
        sess_name = os.path.basename(sess_dir)
        amp_files = sorted(glob.glob(os.path.join(sess_dir, glob_tag)), key=os.path.getmtime)
        spont = np.array([], dtype=float)
        trig = np.array([], dtype=float)
        if amp_files:
            try:
                dfm = pd.read_csv(amp_files[-1])
                st = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                amp = pd.to_numeric(dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                spont = amp[(st.to_numpy() == "spontaneous") & np.isfinite(amp)]
                trig = amp[(st.to_numpy() == "triggered") & np.isfinite(amp)]
            except Exception:
                pass
        rows.append({
            "session_name": sess_name,
            "spont_mean": float(np.nanmean(spont)) if spont.size else np.nan,
            "trig_mean": float(np.nanmean(trig)) if trig.size else np.nan,
            "spont_std": float(np.nanstd(spont)) if spont.size else np.nan,
            "trig_std": float(np.nanstd(trig)) if trig.size else np.nan,
            "spont_raw": spont.tolist(),
            "trig_raw": trig.tolist(),
        })

    labels = [r["session_name"] for r in rows]
    x = np.arange(len(labels))
    spont_vals = np.array([r["spont_mean"] for r in rows])
    trig_vals = np.array([r["trig_mean"] for r in rows])

    all_raw = np.array([v for r in rows for v in r["spont_raw"] + r["trig_raw"] if np.isfinite(v)])
    ylim = None
    if all_raw.size:
        vmax = float(np.nanpercentile(all_raw, 99))
        vmin = float(np.nanmin(all_raw))
        span = max(vmax - vmin, 1e-6)
        ylim = (min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)

    fig_w = max(8, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    spont_stds = np.array([r["spont_std"] for r in rows])
    trig_stds = np.array([r["trig_std"] for r in rows])
    sp_ok = np.isfinite(spont_vals)
    tr_ok = np.isfinite(trig_vals)

    for i, r in enumerate(rows):
        if r["spont_raw"]:
            ax.scatter([i] * len(r["spont_raw"]), r["spont_raw"], color="#4C78A8", alpha=0.25, s=12, zorder=2, linewidths=0)
        if r["trig_raw"]:
            ax.scatter([i] * len(r["trig_raw"]), r["trig_raw"], color="#F58518", alpha=0.25, s=12, zorder=2, linewidths=0)

    if sp_ok.any():
        ax.errorbar(x[sp_ok], spont_vals[sp_ok], yerr=spont_stds[sp_ok],
                     color="#4C78A8", marker="o", linewidth=1.5, markersize=8, capsize=3, label="Spontan", zorder=3)
    if tr_ok.any():
        ax.errorbar(x[tr_ok], trig_vals[tr_ok], yerr=trig_stds[tr_ok],
                     color="#F58518", marker="o", linewidth=1.5, markersize=8, capsize=3, label="Getriggert", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"UP Amplitude pro Session — {port_label}", fontsize=11)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_pdf = os.path.join(str(parent_dir), out_pdf_name)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[PDF] {out_pdf}  ({len(labels)} Sessions)")
    return out_pdf


def main(base_path, lfp_filename=None, ch_a=None, ch_c=None):
    _load_shared_analysis_config()

    base_path = Path(base_path).expanduser().resolve()
    base_tag = base_path.name
    if lfp_filename is None:
        lfp_filename = f"{base_tag}.csv"
    csv_path = base_path / lfp_filename
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    if ch_a is None or ch_c is None:
        json_candidates = sorted(base_path.glob("*.xdat.json"))
        if not json_candidates:
            raise SystemExit(
                "No *.xdat.json found to auto-detect port A/C channels; "
                "pass --ch-a/--ch-c explicitly."
            )
        if ch_a is None:
            ch_a = _deepest_probe_channel(json_candidates[0], "A")
        if ch_c is None:
            ch_c = _deepest_probe_channel(json_candidates[0], "C")
    print(f"[CH] port A deep channel = pri_{ch_a} (ch{ch_a:02d}) | port C deep channel = pri_{ch_c} (ch{ch_c:02d})")

    col_a, col_c = f"ch{ch_a:02d}", f"ch{ch_c:02d}"
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
    usecols = ["time", col_a, col_c]
    have_din = ("din_1" in header) and ("din_2" in header)
    if have_din:
        usecols += ["din_1", "din_2"]

    print(f"[LOAD] reading columns {usecols} from {csv_path.name} ...")
    df = pd.read_csv(csv_path, usecols=usecols)

    time_raw = df["time"].to_numpy(dtype=float)
    x_a_raw = df[col_a].to_numpy(dtype=float)
    x_c_raw = df[col_c].to_numpy(dtype=float)

    # Simple-stride downsample (matches Main_safe.py's downsampling_old: no anti-alias filter)
    time_ds = time_raw[::DOWNSAMPLE_FACTOR]
    x_a = x_a_raw[::DOWNSAMPLE_FACTOR]
    x_c = x_c_raw[::DOWNSAMPLE_FACTOR]

    # ticks -> seconds, matching Main_safe.py's unit-fix heuristic
    if np.nanmax(np.abs(time_ds)) > 1e6:
        time_ds = time_ds / DEFAULT_FS_XDAT
    time_s = time_ds
    dt = float(np.mean(np.diff(time_s)))

    pulse_times_1 = pulse_times_2 = np.array([], dtype=float)
    pulse_times_1_off = pulse_times_2_off = np.array([], dtype=float)
    if have_din:
        din1_ds = df["din_1"].to_numpy(dtype=float)[::DOWNSAMPLE_FACTOR]
        din2_ds = df["din_2"].to_numpy(dtype=float)[::DOWNSAMPLE_FACTOR]
        pulse_times_1, pulse_times_1_off = _rising_falling(din1_ds, time_s)
        pulse_times_2, pulse_times_2_off = _rising_falling(din2_ds, time_s)
        print(f"[PULSE] din_1: {len(pulse_times_1)} onsets | din_2: {len(pulse_times_2)} onsets")
    else:
        print("[PULSE] no din_1/din_2 columns in CSV -> no pulses")

    x_a, Up_a = _detect_states_for_channel(
        x_a, time_s, dt, f"pri_{ch_a}",
        pulse_times_1, pulse_times_2, pulse_times_1_off, pulse_times_2_off,
    )
    x_c, Up_c = _detect_states_for_channel(
        x_c, time_s, dt, f"pri_{ch_c}",
        pulse_times_1, pulse_times_2, pulse_times_1_off, pulse_times_2_off,
    )

    out_html = export_interactive_two_channel_lfp_html(
        f"{base_tag}__portA_pri{ch_a}_vs_portC_pri{ch_c}", str(base_path),
        time_s, x_a, x_c,
        pulse_times_1=pulse_times_1, pulse_times_2=pulse_times_2,
        pulse_times_1_off=pulse_times_1_off, pulse_times_2_off=pulse_times_2_off,
        top_spont=(Up_a["Spontaneous_UP"], Up_a["Spontaneous_DOWN"]),
        top_trig=(Up_a["Pulse_triggered_UP"], Up_a["Pulse_triggered_DOWN"]),
        top_assoc=(Up_a["Pulse_associated_UP"], Up_a["Pulse_associated_DOWN"]),
        bottom_spont=(Up_c["Spontaneous_UP"], Up_c["Spontaneous_DOWN"]),
        bottom_trig=(Up_c["Pulse_triggered_UP"], Up_c["Pulse_triggered_DOWN"]),
        bottom_assoc=(Up_c["Pulse_associated_UP"], Up_c["Pulse_associated_DOWN"]),
        top_spont_label="UP spontaneous (Port A)",
        top_trig_label="UP triggered (Port A)",
        top_assoc_label="UP associated (Port A)",
        bottom_spont_label="UP spontaneous (Port C)",
        bottom_trig_label="UP triggered (Port C)",
        bottom_assoc_label="UP associated (Port C)",
        title=(
            f"{base_tag} — Port A pri_{ch_a} (deep) vs Port C pri_{ch_c} (deep), "
            f"independent per-channel upstate detection"
        ),
        top_name=f"Port A pri_{ch_a} (deep)",
        bottom_name=f"Port C pri_{ch_c} (deep)",
        top_y_label=f"Port A — pri_{ch_a} (µV)",
        bottom_y_label=f"Port C — pri_{ch_c} (µV)",
    )
    print(f"[DONE] {out_html}")

    # Per-session amplitude CSVs (own glob tag per port, so they never collide
    # with Main_safe.py's own "*__upstate_amplitudes.csv") + parent-level trend
    # PDFs across all sibling sessions, one plot for Port A and one for Port C.
    amp_csv_a = base_path / f"{base_tag}__portA_pri{ch_a}__deep_channel_upstate_amplitudes.csv"
    amp_csv_c = base_path / f"{base_tag}__portC_pri{ch_c}__deep_channel_upstate_amplitudes.csv"
    _write_amplitude_csv(x_a, Up_a, dt, amp_csv_a)
    _write_amplitude_csv(x_c, Up_c, dt, amp_csv_c)

    _update_deep_channel_amplitude_trend_pdf(
        base_path, "Port A (deep)", "*__portA_pri*__deep_channel_upstate_amplitudes.csv",
        "upstate_summary_ALL_parent__up_amplitude_trend_portA.pdf",
    )
    _update_deep_channel_amplitude_trend_pdf(
        base_path, "Port C (deep)", "*__portC_pri*__deep_channel_upstate_amplitudes.csv",
        "upstate_summary_ALL_parent__up_amplitude_trend_portC.pdf",
    )

    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_path", help="Session folder (contains the CSV + *.xdat.json)")
    ap.add_argument("--lfp-filename", default=None, help="CSV filename (default: <folder-name>.csv)")
    ap.add_argument("--ch-a", type=int, default=None, help="Override port-A raw channel index (default: deepest connected)")
    ap.add_argument("--ch-c", type=int, default=None, help="Override port-C raw channel index (default: deepest connected)")
    args = ap.parse_args()
    main(args.base_path, args.lfp_filename, args.ch_a, args.ch_c)
