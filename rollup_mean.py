#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_FOR_ANNA = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO"
OUT_DIR = os.path.join(BASE_FOR_ANNA, "ROLLUPS_EXISTING")
os.makedirs(OUT_DIR, exist_ok=True)

def find_all_summaries(root):
    # sammelt alle .../upstate_summary.csv unterhalb von FOR ANNA IN VIVO
    return sorted(glob.glob(os.path.join(root, "**", "upstate_summary.csv"), recursive=True))

def weighted_mean(pairs):
    """pairs: Liste (mean, n). Gibt gewichteten Mittelwert oder np.nan."""
    num = 0.0; den = 0
    for m, n in pairs:
        if pd.notna(m) and n and n > 0:
            num += float(m) * int(n)
            den += int(n)
    return (num / den) if den > 0 else np.nan, den

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def load_session_spectra(sess_dir):
    """
    Optional: falls vorhanden, lade pro Session Spektren
    Erwartete Dateien:
      spectrum_spont.csv  (freq,power)
      spectrum_trig.csv   (freq,power)
    """
    out = {}
    for key, fname in [("spont", "spectrum_spont.csv"), ("trig", "spectrum_trig.csv")]:
        p = os.path.join(sess_dir, fname)
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p)
                f = df.iloc[:,0].to_numpy(dtype=float)
                y = df.iloc[:,1].to_numpy(dtype=float)
                out[key] = (f, y)
            except Exception:
                pass
    return out

def load_session_csd(sess_dir):
    """
    Optional: falls vorhanden, lade pro Session CSD Arrays
    Erwartete Dateien:
      csd_spont.npy  (Z x T)
      csd_trig.npy   (Z x T)
    """
    out = {}
    for key, fname in [("spont","csd_spont.npy"), ("trig","csd_trig.npy")]:
        p = os.path.join(sess_dir, fname)
        if os.path.isfile(p):
            try:
                out[key] = np.load(p)
            except Exception:
                pass
    return out

def interp_specs_to_common_grid(spec_list, df=1.0):
    """
    spec_list: Liste von (freq_vector, power_vector)
    -> (f_grid, mean, sem) oder (None,None,None)
    """
    if not spec_list:
        return None, None, None
    fmins=[]; fmaxs=[]
    cleaned=[]
    for f,y in spec_list:
        f = np.asarray(f); y = np.asarray(y)
        good = np.isfinite(f) & np.isfinite(y)
        if good.sum() >= 5:
            cleaned.append((f[good], y[good]))
            fmins.append(float(np.min(f[good])))
            fmaxs.append(float(np.max(f[good])))
    if not cleaned:
        return None, None, None
    f_lo = max(0.0, max(fmins))
    f_hi = min(fmaxs)
    if f_hi <= f_lo + df:
        return None, None, None
    f_grid = np.arange(math.ceil(f_lo), math.floor(f_hi)+1, df)
    stack=[]
    for f,y in cleaned:
        stack.append(np.interp(f_grid, f, y))
    arr = np.vstack(stack)
    mean = np.nanmean(arr, axis=0)
    sem  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return f_grid, mean, sem

    
def main():
    # ---- 1) UP-Durations aus upstate_summary.csv poolen
    summaries = find_all_summaries(BASE_FOR_ANNA)
    rows = []
    dur_pairs_spont = []
    dur_pairs_trig  = []

    for path in summaries:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            continue
        if df.empty:
            continue

        # nimm die letzte Zeile (neueste)
        r = df.iloc[-1].to_dict()
        parent = r.get("Parent", "")
        exp    = r.get("Experiment", "")

        # Counts (robust gegen "", NaN, "12.0")
        def safe_int(v, default=0):
            try:
                x = pd.to_numeric(v, errors="coerce")
                if pd.isna(x):
                    return default
                return int(float(x))
            except Exception:
                return default

        n_sp = safe_int(r.get("spon", 0), default=0)
        n_tr = safe_int(r.get("triggered", 0), default=0)

        # Means
        m_all = safe_float(r.get("Mean UP Dauer [s]", np.nan))
        m_tr  = safe_float(r.get("Mean UP Dauer Triggered [s]", np.nan))
        m_sp  = safe_float(r.get("Mean UP Dauer Spontaneous [s]", np.nan))

        rows.append({
            "Parent": parent, "Experiment": exp,
            "n_spon": n_sp, "n_trig": n_tr,
            "mean_all_s": m_all, "mean_spont_s": m_sp, "mean_trig_s": m_tr,
            "summary_csv": path
        })

        if not np.isnan(m_sp) and n_sp > 0:
            dur_pairs_spont.append((m_sp, n_sp))
        if not np.isnan(m_tr) and n_tr > 0:
            dur_pairs_trig.append((m_tr, n_tr))

    # Sessions-Tabelle ablegen (auch wenn leer)
    df_summ = pd.DataFrame(rows)
    df_summ.to_csv(os.path.join(OUT_DIR, "sessions_from_upstate_summary.csv"), index=False)

    pooled_sp, Nsp = weighted_mean(dur_pairs_spont)
    pooled_tr, Ntr = weighted_mean(dur_pairs_trig)

    with open(os.path.join(OUT_DIR, "up_duration_pooled.txt"), "w") as f:
        if not np.isnan(pooled_sp):
            f.write(f"Spontaneous: mean={pooled_sp:.6g} s  (events={Nsp})\n")
        else:
            f.write("Spontaneous: no data\n")
        if not np.isnan(pooled_tr):
            f.write(f"Triggered:   mean={pooled_tr:.6g} s  (events={Ntr})\n")
        else:
            f.write("Triggered:   no data\n")

    # kleiner Balkenplot (nur Mean)
    labels, vals = [], []
    if not np.isnan(pooled_sp): labels.append("Spont"); vals.append(pooled_sp)
    if not np.isnan(pooled_tr): labels.append("Trig");  vals.append(pooled_tr)
    if vals:
        fig = plt.figure(figsize=(4.2, 4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labels)
        ax.set_ylabel("UP duration (s)")
        ax.set_title("UP durations — pooled mean (existing summaries)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "up_durations_pooled_mean.png"), dpi=200)
        plt.close(fig)

    # ---- 2) Optional: vorhandene Spektren & CSD pro Session einsammeln
    session_dirs = sorted({os.path.dirname(p) for p in summaries})
    specs_spont, specs_trig = [], []
    csd_sp_list, csd_tr_list = [], []

    for sess in session_dirs:
        sp = load_session_spectra(sess)
        if "spont" in sp: specs_spont.append(sp["spont"])
        if "trig"  in sp: specs_trig.append(sp["trig"])
        csd = load_session_csd(sess)
        if "spont" in csd and "trig" in csd:
            csd_sp_list.append(csd["spont"])
            csd_tr_list.append(csd["trig"])

    # Spektren mitteln (falls verfügbar)
    fS, mS, sS = interp_specs_to_common_grid(specs_spont, df=1.0)
    fT, mT, sT = interp_specs_to_common_grid(specs_trig,  df=1.0)
    if fS is not None and fT is not None:
        f_lo = max(fS[0], fT[0]); f_hi = min(fS[-1], fT[-1])
        f_grid = np.arange(math.ceil(f_lo), math.floor(f_hi) + 1, 1.0)
        def on_grid(f, y): return np.interp(f_grid, f, y)
        S_m = on_grid(fS, mS); S_s = on_grid(fS, sS)
        T_m = on_grid(fT, mT); T_s = on_grid(fT, sT)

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(f_grid, S_m, label="Spont"); ax.fill_between(f_grid, S_m - S_s, S_m + S_s, alpha=0.25)
        ax.plot(f_grid, T_m, label="Triggered"); ax.fill_between(f_grid, T_m - T_s, T_m + T_s, alpha=0.25)
        ax.set_xlabel("Hz"); ax.set_ylabel("Power (a.u.)")
        ax.set_title("Power spectrum — Mean ± SEM (existing per-session files)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "power_spectrum_mean_sem_existing.png"), dpi=200)
        plt.close(fig)

        pd.DataFrame({
            "freq_hz": f_grid,
            "power_spont_mean": S_m, "power_spont_sem": S_s,
            "power_trig_mean":  T_m, "power_trig_sem":  T_s,
        }).to_csv(os.path.join(OUT_DIR, "power_spectrum_mean_sem_existing.csv"), index=False)

    # CSD mitteln (falls verfügbar, gleiche Form)
    if csd_sp_list and csd_tr_list:
        try:
            Z, T = csd_sp_list[0].shape
            if all((c.shape == (Z, T) for c in csd_sp_list + csd_tr_list)):
                meanS = np.nanmean(np.stack(csd_sp_list, 0), axis=0)
                meanT = np.nanmean(np.stack(csd_tr_list, 0), axis=0)

                fig, ax = plt.subplots(figsize=(7.2, 3.6))
                im = ax.imshow(meanS, aspect="auto", origin="upper")
                ax.set_title("CSD Spont — mean (existing files)")
                fig.colorbar(im, ax=ax); fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, "csd_spont_mean_existing.png"), dpi=200)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7.2, 3.6))
                im = ax.imshow(meanT, aspect="auto", origin="upper")
                ax.set_title("CSD Triggered — mean (existing files)")
                fig.colorbar(im, ax=ax); fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, "csd_trig_mean_existing.png"), dpi=200)
                plt.close(fig)

                np.save(os.path.join(OUT_DIR, "csd_spont_mean_existing.npy"), meanS)
                np.save(os.path.join(OUT_DIR, "csd_trig_mean_existing.npy"),  meanT)
        except Exception:
            pass

    print(f"[DONE] Outputs geschrieben nach: {OUT_DIR}")


if __name__ == "__main__":
    main()


