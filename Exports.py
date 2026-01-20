import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
from datetime import datetime
from matplotlib.colors import SymLogNorm
import os
import sys, os


ANALYSE_IN_AU = True
HTML_IN_uV    = True

_DEFAULT_SESSION = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/"
BASE_PATH   = globals().get("BASE_PATH", _DEFAULT_SESSION)
SAVE_DIR = BASE_PATH
LOGFILE = os.path.join(SAVE_DIR, "runlog.txt")


def export_interactive_lfp_html(
    base_tag, save_dir, time_s, y,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    up_spont=None,       # Tuple (UP_idx, DOWN_idx) in SAMPLE-INDIZES
    up_trig=None,        # Tuple (UP_idx, DOWN_idx)
    up_assoc=None,       # Tuple (UP_idx, DOWN_idx)
    max_points=300_000,
    title="LFP (interaktiv)",
    limit_to_last_pulse=False,
    y_label="LFP (µV)"
):


    t = np.asarray(time_s, dtype=float)
    x = np.asarray(y, dtype=float)

    # optional auf letzten Puls begrenzen
    if limit_to_last_pulse:
        last_p = None
        if pulse_times_1 is not None and len(pulse_times_1):
            last_p = float(np.max(pulse_times_1))
        if pulse_times_2 is not None and len(pulse_times_2):
            lp2 = float(np.max(pulse_times_2))
            last_p = lp2 if (last_p is None or lp2 > last_p) else last_p
        if last_p is not None and len(t):
            i1 = int(np.searchsorted(t, last_p, side="right"))
            i1 = max(1, min(i1, len(t)))
            t = t[:i1]
            x = x[:i1]

    # robustes Decimate (nur Darstellung)
    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        x = x[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name="LFP"))

    shapes = []

    # --- Helper: UP/DOWN-Indizes -> Zeitintervalle (Sekunden)
    def _mk_intervals(UP, DOWN):
        if UP is None or DOWN is None:
            return []
        UP   = np.asarray(UP,   dtype=int)
        DOWN = np.asarray(DOWN, dtype=int)
        m = min(len(UP), len(DOWN))
        if m == 0:
            return []
        UP, DOWN = UP[:m], DOWN[:m]
        # sortiert nach StartzeitStartzeit
        order = np.argsort(UP)
        UP, DOWN = UP[order], DOWN[order]
        out = []
        for u, d in zip(UP, DOWN):
            if 0 <= u < len(time_s) and 0 < d <= len(time_s) and d > u:
                out.append((float(time_s[u]), float(time_s[d-1])))
        return out

    # --- UP-Intervalle vorbereiten (Farben an deine Matplotlib-Plots angelehnt)
    intervals = []
    if up_spont:
        intervals.append(("UP spontaneous", _mk_intervals(*up_spont), "rgba(46, 204, 113, 0.22)"))  # grün
    if up_trig:
        intervals.append(("UP triggered",   _mk_intervals(*up_trig),  "rgba(31, 119, 180, 0.22)"))  # blau
    if up_assoc:
        intervals.append(("UP associated",  _mk_intervals(*up_assoc), "rgba(255, 127, 14, 0.22)"))  # orange

    # --- Schattierungen als Shapes (über gesamte Plot-Höhe)
    for label, spans, fill in intervals:
        for (t0, t1) in spans:
            # auf ggf. gekürzte Zeitachse clippen
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor=fill
            ))
        # --- Pulse-Intervalle (Onset->Offset) als rote Fläche, bewusst NACH den UP-Flächen
    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals)/2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0); t1 = float(t1)
            if t1 <= t0:
                continue
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor=fill
            ))

    # deutlich sichtbarer als vorher
    _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.28)")
    _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.28)")

    # --- Pulse-Linien
    def _add_pulses(ts, dash):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size/1200))]
        for p in tt:
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref="x", yref="paper",
                opacity=0.35,
                line=dict(width=1, dash=dash, color="red")
            ))

    _add_pulses(pulse_times_1, "dot")
    _add_pulses(pulse_times_2, "dash")
        # --- Pulse-OFF-Linien (Offsets) in dunklerem Rot

    def _add_pulse_offs(ts, dash):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size/1200))]
        for p in tt:
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref="x", yref="paper",
                opacity=0.55,
                line=dict(width=1, dash=dash, color="darkred")
            ))

    _add_pulse_offs(pulse_times_1_off, "dot")
    _add_pulse_offs(pulse_times_2_off, "dash")

        # --- Pulse-Intervalle als transparente Rechtecke (Onset->Offset)
    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        # optional ausdünnen, falls extrem viele Intervalle
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals)/2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0); t1 = float(t1)
            if t1 <= t0:
                continue
            # auf ggf. gekürzte Zeitachse clippen
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor=fill
            ))

    _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.12)")   # rot transparent
    _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.12)")   # optional: 2. Spur etwas schwächer


    


    # --- Dummy-Traces für Legende (damit Shapes in der Legende erscheinen)
    if intervals:
        for label, _, fill in intervals:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=12, color=fill),
                name=label,
                showlegend=True
            ))
    # if pulse_times_1 is not None and len(pulse_times_1):
    #     fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
    #                              line=dict(width=1, dash="dot", color="red"),
    #                              name="Pulse 1"))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(width=1, dash="dot", color="darkred"),
                                 name="Pulse 1 OFF"))
    # if pulse_times_2 is not None and len(pulse_times_2):
    #     fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
    #                              line=dict(width=1, dash="dash", color="red"),
    #                              name="Pulse 2"))
    if pulse_intervals_1 is not None and len(pulse_intervals_1):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(width=12, color="rgba(255, 0, 0, 0.12)"),
                                 name="Pulse 1 duration"))


    fig.update_layout(
        title=title,
        xaxis=dict(title="Zeit (s)", rangeslider=dict(visible=True)),
        yaxis=dict(title=y_label),
        shapes=shapes,
        margin=dict(l=60, r=20, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    out_html = os.path.join(save_dir, f"{base_tag}__lfp_interactive.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] interaktiver LFP-Plot: {out_html}")
    return out_html


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()



def _nan_stats(name, arr):
    import numpy as np
    if arr is None:
        print(f"[DIAG] {name}: None"); return
    a = np.asarray(arr, float)
    nan_rate = np.mean(~np.isfinite(a))*100.0 if a.size else 100.0
    print(f"[DIAG] {name}: shape={a.shape}, NaN%={nan_rate:.2f}%")
    if a.size == 0 or not np.isfinite(a).any():
        print(f"[DIAG] {name}: empty/invalid -> skip quantiles")
        return
    aa = np.abs(a[np.isfinite(a)])
    if aa.size == 0:
        print(f"[DIAG] {name}: no finite values -> skip quantiles")
        return
    try:
        qs = np.nanpercentile(aa, [50, 90, 99, 99.9])
        print(f"[DIAG] {name} |abs| quantiles: 50%={qs[0]:.3g}, 90%={qs[1]:.3g}, 99%={qs[2]:.3g}, 99.9%={qs[3]:.3g}")
    except Exception as e:
        print(f"[DIAG] {name}: quantiles failed: {e}")


def _rms(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a*a))) if a.size else np.nan
