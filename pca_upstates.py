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
        if waveforms is None or len(session_names) < 1:
            continue
        entries.append((os.path.basename(pd_path), waveforms, labels, session_names))

    if not entries:
        print("Keine Wellenform-Daten gefunden. Bitte zuerst Main_safe.py laufen lassen.")
        return

    print(f"Gefundene Parent-Ordner mit Daten: {len(entries)}")
    for name, wf, _, sess in entries:
        print(f"  {name}: {wf.shape[0]} Upstates, {len(sess)} Sessions")

    with PdfPages(OUT_PDF) as pdf:
        for folder_name, waveforms, labels, session_names in entries:
            # z-score pro Upstate (Wellenform-Variabilität normalisieren)
            scaler = StandardScaler()
            wf_scaled = scaler.fit_transform(waveforms)

            pca = PCA(n_components=min(N_COMPONENTS, wf_scaled.shape[1], wf_scaled.shape[0]))
            coords = pca.fit_transform(wf_scaled)
            var_explained = pca.explained_variance_ratio_ * 100

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # PCA Scatter
            plot_pca(axes[0], coords, labels, session_names, folder_name)
            axes[0].set_xlabel(f"PC 1 ({var_explained[0]:.1f}% Varianz)")
            axes[0].set_ylabel(f"PC 2 ({var_explained[1]:.1f}% Varianz)" if len(var_explained) > 1 else "PC 2")

            # Mittlere Wellenform pro Session
            t = np.linspace(0, 1, waveforms.shape[1])
            cmap = plt.cm.get_cmap("tab10", len(session_names))
            for i, name in enumerate(session_names):
                mask = labels == name
                mean_wf = np.mean(waveforms[mask], axis=0)
                std_wf  = np.std(waveforms[mask], axis=0)
                axes[1].plot(t, mean_wf, color=cmap(i), label=name, linewidth=1.5)
                axes[1].fill_between(t, mean_wf - std_wf, mean_wf + std_wf,
                                     color=cmap(i), alpha=0.15)
            axes[1].set_xlabel("Normalisierte Zeit (Upstate-Start → Ende)")
            axes[1].set_ylabel("LFP (µV)")
            axes[1].set_title("Mittlere Upstate-Wellenform ± SD")
            axes[1].grid(alpha=0.2, linestyle=":")
            axes[1].legend(fontsize=7)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\n[DONE] PDF gespeichert: {OUT_PDF}")


if __name__ == "__main__":
    main()
