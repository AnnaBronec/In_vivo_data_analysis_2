#!/usr/bin/env python3
"""
Lädt alle *__upstate_waveforms.npy aus Unterordnern von ROOT_DIR,
führt PCA durch und speichert pro Parent-Ordner einen Plot (PC1 vs PC2,
farbkodiert nach Session) als PDF.
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

ROOT_DIR = "/run/media/ananym/7FBA-F71B/invivo/hM4Di"
SMOOTH_WINDOW = 12  # Punkte (von 100 gesamt) die geglättet werden — entspricht ~12% der Upstate-Dauer
OUT_PDF  = os.path.join(ROOT_DIR, "ALL_pca_upstates.pdf")
N_COMPONENTS = 2


def _nat_session_key(sess_name):
    m = re.match(r"^\s*(\d+)", str(sess_name))
    if m:
        return (0, int(m.group(1)), str(sess_name).lower())
    return (1, str(sess_name).lower())


def load_waveforms(parent_dir):
    """Lädt alle Wellenform-Arrays aus Unterordnern, gibt (waveforms, labels) zurück."""
    session_dirs = sorted(
        [d for d in os.scandir(parent_dir) if d.is_dir()],
        key=lambda e: _nat_session_key(e.name)
    )
    all_waveforms = []
    all_labels    = []
    session_names = []
    for entry in session_dirs:
        npy_files = sorted(glob.glob(os.path.join(entry.path, "*__upstate_waveforms.npy")),
                           key=os.path.getmtime)
        if not npy_files:
            continue
        try:
            wf = np.load(npy_files[-1])  # shape (n_upstates, N_PCA_POINTS)
            if wf.ndim != 2 or wf.shape[0] == 0:
                continue
            wf = uniform_filter1d(wf, size=SMOOTH_WINDOW, axis=1)
            all_waveforms.append(wf)
            all_labels.extend([entry.name] * wf.shape[0])
            session_names.append(entry.name)
        except Exception as e:
            print(f"  [WARN] {entry.name}: {e}")
    if not all_waveforms:
        return None, None, []
    return np.vstack(all_waveforms), np.array(all_labels), session_names


def plot_pca(ax, coords, labels, session_names, folder_name):
    cmap = plt.cm.get_cmap("tab10", len(session_names))
    colors = {name: cmap(i) for i, name in enumerate(session_names)}

    for name in session_names:
        mask = labels == name
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=colors[name], alpha=0.35, s=14, linewidths=0,
            label=name, zorder=2
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(folder_name, fontsize=12, fontweight="bold", pad=8)

    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(fontsize=7, markerscale=2, loc="best")


def main():
    parent_dirs = sorted(
        [e.path for e in os.scandir(ROOT_DIR) if e.is_dir()],
        key=lambda p: _nat_session_key(os.path.basename(p))
    )

    entries = []
    for pd_path in parent_dirs:
        waveforms, labels, session_names = load_waveforms(pd_path)
        if waveforms is not None and len(session_names) >= 1:
            entries.append((os.path.basename(pd_path), waveforms, labels, session_names))
        else:
            for sub_path in sorted([e.path for e in os.scandir(pd_path) if e.is_dir()],
                                   key=lambda p: _nat_session_key(os.path.basename(p))):
                sub_wf, sub_lbl, sub_sess = load_waveforms(sub_path)
                if sub_wf is not None and len(sub_sess) >= 1:
                    entries.append((os.path.basename(sub_path), sub_wf, sub_lbl, sub_sess))

    if not entries:
        print("Keine Wellenform-Daten gefunden. Bitte zuerst Main_safe.py laufen lassen.")
        return

    print(f"Gefundene Parent-Ordner mit Daten: {len(entries)}")
    for name, wf, _, sess in entries:
        print(f"  {name}: {wf.shape[0]} Upstates, {len(sess)} Sessions")

    # Pass 1: PCA berechnen und globales y-Max für Balkendiagramme bestimmen
    pca_results = []
    global_bar_ymax = 0.0
    for folder_name, waveforms, labels, session_names in entries:
        scaler = StandardScaler()
        wf_scaled = scaler.fit_transform(waveforms)
        pca = PCA(n_components=min(N_COMPONENTS, wf_scaled.shape[1], wf_scaled.shape[0]))
        coords = pca.fit_transform(wf_scaled)
        var_explained = pca.explained_variance_ratio_ * 100

        baseline_name = next(
            (n for n in session_names if n.startswith("0_")),
            session_names[0],
        )
        pc1_global_std = coords[:, 0].std(ddof=1) or 1.0
        per_upstate = {n: coords[labels == n, 0] for n in session_names}
        bl_vals = per_upstate[baseline_name]
        bl_mu   = bl_vals.mean()
        bl_se   = bl_vals.std(ddof=1) / np.sqrt(max(bl_vals.size, 1))

        pct_devs, sem_pcts = [], []
        for name in session_names:
            vals = per_upstate[name]
            mu_i = vals.mean()
            se_i = vals.std(ddof=1) / np.sqrt(max(vals.size, 1))
            pct = (mu_i - bl_mu) / pc1_global_std * 100
            sem = 0.0 if name == baseline_name else \
                  np.sqrt(se_i**2 + bl_se**2) / pc1_global_std * 100
            pct_devs.append(pct)
            sem_pcts.append(sem)

        pca_results.append((folder_name, coords, labels, session_names, var_explained,
                            baseline_name, pct_devs, sem_pcts))
        local_max = max((abs(p) + s) for p, s in zip(pct_devs, sem_pcts))
        global_bar_ymax = max(global_bar_ymax, local_max)

    bar_ylim = (0, global_bar_ymax * 1.15) if global_bar_ymax > 0 else (0, 1)

    # Pass 2: plotten mit einheitlicher Y-Achse für Balkendiagramme
    with PdfPages(OUT_PDF) as pdf:
        for (folder_name, coords, labels, session_names, var_explained,
             baseline_name, pct_devs, sem_pcts) in pca_results:

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # PCA Scatter
            plot_pca(axes[0], coords, labels, session_names, folder_name)
            axes[0].set_xlabel(f"PC 1 ({var_explained[0]:.1f}% variance)")
            axes[0].set_ylabel(f"PC 2 ({var_explained[1]:.1f}% variance)" if len(var_explained) > 1 else "PC 2")

            # Balkendiagramm mit globalem y-Limit
            cmap = plt.cm.get_cmap("tab10", len(session_names))
            x_pos  = np.arange(len(session_names))
            ax2    = axes[1]
            colors = [cmap(i) for i in range(len(session_names))]
            ax2.bar(x_pos, np.abs(pct_devs), yerr=sem_pcts, color=colors, alpha=0.8,
                    error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
            ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
            ax2.set_ylim(bar_ylim)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(session_names, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel(f"PC1 deviation from '{baseline_name}' (% total variance)")
            ax2.set_title(f"PCA – PC1 deviation from '{baseline_name}' ± SEM")
            ax2.grid(alpha=0.2, linestyle=":", axis="y")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\n[DONE] PDF gespeichert: {OUT_PDF}")


def _mahal_distance(coords_spont, coords_trig, n_pc=5):
    """
    Mahalanobis-Distanz des Triggered-Zentroids von der Spontaneous-Verteilung.
    Berechnet in den ersten n_pc PCA-Dimensionen.
    """
    n_use = min(n_pc, coords_spont.shape[1])
    s = coords_spont[:, :n_use]
    t = coords_trig[:, :n_use]
    mu_s = s.mean(axis=0)
    mu_t = t.mean(axis=0)
    if s.shape[0] <= n_use:
        cov_s = np.eye(n_use)
    else:
        cov_s = np.cov(s.T)
        if cov_s.ndim == 0:
            cov_s = np.array([[float(cov_s)]])
        cov_s += np.eye(n_use) * 1e-6  # Regularisierung
    try:
        cov_inv = np.linalg.inv(cov_s)
        diff = mu_t - mu_s
        d = float(np.sqrt(np.maximum(0.0, diff @ cov_inv @ diff)))
    except np.linalg.LinAlgError:
        d = np.nan
    return d


def mahal_summary(parent_dir, out_pdf=None, n_pc=5, smooth_window=12):
    """
    Scannt parent_dir nach Session-Unterordnern die sowohl
    *__upstate_waveforms.npy (spontaneous) als auch *__trig_waveforms.npy (triggered)
    enthalten.

    PDF-Inhalt:
      Seite 1  — Übersichts-Balkendiagramm: Mahalanobis-Distanz pro Session
      Seite 2+ — Pro Session: PCA-Scatter (spont=grün, trig=blau) + Mahalanobis-Wert
    """
    if out_pdf is None:
        out_pdf = os.path.join(parent_dir, "ALL_mahal_spont_vs_trig.pdf")

    # Rekursiv alle Ordner finden die BEIDE Waveform-Dateien enthalten,
    # gruppiert nach dem direkten Unterordner von parent_dir (= "DRD cross" etc.)
    from collections import defaultdict
    groups = defaultdict(list)  # group_name -> [session_path, ...]
    for root, dirs, files in os.walk(parent_dir):
        dirs.sort(key=_nat_session_key)
        has_spont = any(f.endswith("__upstate_waveforms.npy") for f in files)
        has_trig  = any(f.endswith("__trig_waveforms.npy")    for f in files)
        if has_spont and has_trig:
            rel = os.path.relpath(root, parent_dir)
            group = rel.split(os.sep)[0]
            groups[group].append(root)

    if not groups:
        print("[mahal_summary] Keine Sessions mit beiden Waveform-Dateien gefunden.")
        print("  -> Zuerst Main_safe.py laufen lassen (speichert jetzt auch *__trig_waveforms.npy)")
        return

    def _load_group(session_paths):
        """Lädt alle Sessions einer Gruppe, gibt Liste von (name, d, n_s, n_t) zurück."""
        rows = []
        for session_path in sorted(session_paths, key=lambda p: _nat_session_key(os.path.basename(p))):
            entry_name = os.path.basename(session_path)
            spont_files = sorted(glob.glob(os.path.join(session_path, "*__upstate_waveforms.npy")),
                                 key=os.path.getmtime)
            trig_files  = sorted(glob.glob(os.path.join(session_path, "*__trig_waveforms.npy")),
                                 key=os.path.getmtime)
            if not spont_files or not trig_files:
                continue
            try:
                spont = np.load(spont_files[-1])
                trig  = np.load(trig_files[-1])
                if spont.ndim != 2 or trig.ndim != 2:
                    continue
                if spont.shape[0] < 3 or trig.shape[0] < 3:
                    print(f"  [SKIP] {entry_name}: zu wenig Upstates "
                          f"(spont={spont.shape[0]}, trig={trig.shape[0]})")
                    continue
                spont = uniform_filter1d(spont, size=smooth_window, axis=1)
                trig  = uniform_filter1d(trig,  size=smooth_window, axis=1)
                combined = np.vstack([spont, trig])
                scaler = StandardScaler()
                cs = scaler.fit_transform(combined)
                n_comp = min(n_pc, cs.shape[1], cs.shape[0] - 1)
                coords = PCA(n_components=n_comp).fit_transform(cs)
                cs_s = coords[:spont.shape[0]]
                cs_t = coords[spont.shape[0]:]
                d = _mahal_distance(cs_s, cs_t, n_pc=n_comp)
                rows.append((entry_name, d, spont.shape[0], trig.shape[0]))
                print(f"  {entry_name}: mahal={d:.3f} (s={spont.shape[0]}, t={trig.shape[0]})")
            except Exception as e:
                print(f"  [WARN] {entry_name}: {e}")
        return rows

    # Pass 1: alle Gruppen laden, globales y-Max berechnen
    all_group_rows = {}
    for group_name in sorted(groups.keys(), key=_nat_session_key):
        print(f"\n[Gruppe] {group_name}")
        rows = _load_group(groups[group_name])
        if rows:
            all_group_rows[group_name] = rows

    if not all_group_rows:
        print("[mahal_summary] Keine verwertbaren Sessions gefunden.")
        return

    global_ymax = max(
        d for rows in all_group_rows.values()
        for (_, d, _, _) in rows if np.isfinite(d)
    )
    y_lim = (0, global_ymax * 1.15)

    def _bar_page(pdf, rows, group_name, n_pc, y_lim):
        names = [r[0] for r in rows]
        dists = [r[1] for r in rows]
        n_s   = [r[2] for r in rows]
        n_t   = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1 + 2), 5))
        x = np.arange(len(names))
        colors = [plt.colormaps["tab10"](i % 10) for i in range(len(names))]
        bars = ax.bar(x, dists, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        for bar, d, ns, nt in zip(bars, dists, n_s, n_t):
            if np.isfinite(d):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        d + 0.01 * y_lim[1],
                        f"s={ns}, t={nt}", ha="center", va="bottom",
                        fontsize=7.5, color="#333333")
        ax.set_ylim(y_lim)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(f"Mahalanobis distance (first {n_pc} PCs)", fontsize=11)
        ax.set_title(f"{group_name}\nSpontaneous vs. Triggered — Mahalanobis Distance",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25, linestyle=":", axis="y")
        ax.axhline(0, color="black", linewidth=0.8)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Pass 2: plotten
    with PdfPages(out_pdf) as pdf:
        for group_name, rows in all_group_rows.items():
            if not rows:
                print(f"  -> keine verwertbaren Sessions")
                continue
            _bar_page(pdf, rows, group_name, n_pc, y_lim)

    print(f"\n[DONE] Spont vs. Triggered PDF gespeichert: {out_pdf}")


def mean_waveform_summary(parent_dir, out_pdf=None, smooth_window=12):
    """
    Pro Session: mittlere spontane Wellenform (grün) vs. mittlere triggered Wellenform (blau)
    ± SEM, z-normiert auf die spontaneous-Gruppe. Pearson-r als Ähnlichkeitsmaß im Titel.
    Eine PDF-Seite pro Gruppe (direkte Unterordner von parent_dir).
    """
    if out_pdf is None:
        out_pdf = os.path.join(parent_dir, "ALL_mean_waveforms_spont_vs_trig.pdf")

    # Rekursiv nach Sessions mit beiden Waveform-Dateien suchen, gruppiert nach Oberordner
    from collections import defaultdict
    groups = defaultdict(list)
    for root, dirs, files in os.walk(parent_dir):
        dirs.sort(key=_nat_session_key)
        has_s = any(f.endswith("__upstate_waveforms.npy") for f in files)
        has_t = any(f.endswith("__trig_waveforms.npy")    for f in files)
        if has_s and has_t:
            rel   = os.path.relpath(root, parent_dir)
            group = rel.split(os.sep)[0]
            groups[group].append(root)

    if not groups:
        print("[mean_waveform_summary] Keine Sessions gefunden.")
        return

    _WF_PRE_MS  = 300
    _WF_POST_MS = 1000
    _N_NEW      = 130   # neues Format: fixes Onset-Fenster
    _N_OLD      = 100   # altes Format: onset→offset resampelt

    def _x_axis(n_pts):
        if n_pts == _N_NEW:
            return np.linspace(-_WF_PRE_MS, _WF_POST_MS, _N_NEW)
        return np.linspace(0, 100, n_pts)  # altes Format: % der Upstate-Dauer

    def _x_label(n_pts):
        return "Time relative to onset (ms)" if n_pts == _N_NEW else "Upstate duration (%)"

    def _zscore_rows(arr):
        n_pts = arr.shape[1]
        if n_pts == _N_NEW:
            # Neues Format: Baseline bereits subtrahiert, nur durch Baseline-Std dividieren
            n_bl = 25
            sig = arr[:, :n_bl].std(axis=1, keepdims=True)
            sig[sig < 1e-6] = 1.0
            return arr / sig
        else:
            # Altes Format: per-Waveform z-score
            mu  = arr.mean(axis=1, keepdims=True)
            sig = arr.std(axis=1, keepdims=True)
            sig[sig < 1e-6] = 1.0
            return (arr - mu) / sig

    def _load_group_rows(session_paths):
        rows = []
        for sp in sorted(session_paths, key=lambda p: _nat_session_key(os.path.basename(p))):
            sf = sorted(glob.glob(os.path.join(sp, "*__upstate_waveforms.npy")), key=os.path.getmtime)
            tf = sorted(glob.glob(os.path.join(sp, "*__trig_waveforms.npy")),    key=os.path.getmtime)
            if not sf or not tf:
                continue
            try:
                spont = np.load(sf[-1])
                trig  = np.load(tf[-1])
                if spont.ndim != 2 or trig.ndim != 2 or spont.shape[0] < 2 or trig.shape[0] < 2:
                    continue
                spont = uniform_filter1d(spont, size=smooth_window, axis=1)
                trig  = uniform_filter1d(trig,  size=smooth_window, axis=1)
                rows.append((os.path.basename(sp), _zscore_rows(spont), _zscore_rows(trig)))
            except Exception as e:
                print(f"  [WARN] {os.path.basename(sp)}: {e}")
        return rows

    # --- Pass 1: alle Daten laden, globales y-Limit berechnen ---
    all_group_rows = {g: _load_group_rows(paths)
                      for g, paths in groups.items()}

    y_all = []
    for rows in all_group_rows.values():
        for _, spont_z, trig_z in rows:
            for arr in (spont_z, trig_z):
                mu  = arr.mean(axis=0)
                sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
                y_all.extend([mu - sem, mu + sem])
    y_min = float(np.min([v.min() for v in y_all]))
    y_max = float(np.max([v.max() for v in y_all]))
    pad   = (y_max - y_min) * 0.08
    y_lim = (y_min - pad, y_max + pad)

    # --- Pass 2: plotten ---
    with PdfPages(out_pdf) as pdf:
        for group_name in sorted(all_group_rows.keys(), key=_nat_session_key):
            rows = all_group_rows[group_name]
            if not rows:
                continue

            n = len(rows)
            ncols = min(3, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 4.5, nrows * 3.2),
                                     squeeze=False)
            fig.suptitle(f"{group_name}\nMean Upstate Waveform: Spontaneous vs. Triggered",
                         fontsize=12, fontweight="bold", y=1.01)

            for idx, (name, spont_z, trig_z) in enumerate(rows):
                ax = axes[idx // ncols][idx % ncols]

                mu_s  = spont_z.mean(axis=0)
                sem_s = spont_z.std(axis=0) / np.sqrt(spont_z.shape[0])
                mu_t  = trig_z.mean(axis=0)
                sem_t = trig_z.std(axis=0) / np.sqrt(trig_z.shape[0])

                _x = _x_axis(spont_z.shape[1])
                ax.fill_between(_x, mu_s - sem_s, mu_s + sem_s,
                                color="#2ecc71", alpha=0.25)
                ax.fill_between(_x, mu_t - sem_t, mu_t + sem_t,
                                color="#2980b9", alpha=0.25)
                ax.plot(_x, mu_s, color="#27ae60", linewidth=2.0,
                        label=f"Spontaneous (n={spont_z.shape[0]})")
                ax.plot(_x, mu_t, color="#1a5276", linewidth=2.0,
                        label=f"Triggered (n={trig_z.shape[0]})")

                r = float(np.corrcoef(mu_s, mu_t)[0, 1])
                ax.set_ylim(y_lim)
                ax.set_title(f"{name}\nPearson r = {r:.2f}", fontsize=7.5, pad=4)
                ax.set_xlabel(_x_label(spont_z.shape[1]), fontsize=8)
                ax.set_ylabel("µV (baseline norm.)" if spont_z.shape[1] == _N_NEW else "z-score", fontsize=7)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(alpha=0.2, linestyle=":")
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
                if spont_z.shape[1] == _N_NEW:
                    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            for idx in range(len(rows), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  [Gruppe] {group_name}: {len(rows)} Sessions")

    print(f"[DONE] Mittlere Wellenformen PDF gespeichert: {out_pdf}")


ANNA_DIR = "/run/media/ananym/7FBA-F71B/Data/FOR ANNA IN VIVO"

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else ANNA_DIR
    mahal_summary(target)
    mean_waveform_summary(target)
