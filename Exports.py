import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
from plotly.subplots import make_subplots
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
    up_spont_label="UP spontaneous",
    up_trig_label="UP triggered",
    up_assoc_label="UP associated",
    spindle_spont=None,  # Tuple (UP_idx, DOWN_idx)
    spindle_trig=None,   # Tuple (UP_idx, DOWN_idx)
    spindle_assoc=None,  # Tuple (UP_idx, DOWN_idx)
    spindle_spont_label="Spindle spontaneous",
    spindle_trig_label="Spindle triggered",
    spindle_assoc_label="Spindle associated",
    spindle_intervals=None,  # list[(t0, t1)] in Sekunden
    ripple_spont=None,   # Tuple (UP_idx, DOWN_idx)
    ripple_trig=None,    # Tuple (UP_idx, DOWN_idx)
    ripple_assoc=None,   # Tuple (UP_idx, DOWN_idx)
    ripple_spont_label="SWR spontaneous",
    ripple_trig_label="SWR triggered",
    ripple_assoc_label="SWR associated",
    ripple_intervals=None,  # list[(t0, t1)] in Sekunden
    max_points=300_000,
    title="LFP (interaktiv)",
    limit_to_last_pulse=False,
    y_label="LFP (µV)",
    show_pulse_intervals=True,
    y_range=None,
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
        intervals.append((str(up_spont_label), _mk_intervals(*up_spont), "rgba(46, 204, 113, 0.22)"))  # grün
    if up_trig:
        intervals.append((str(up_trig_label), _mk_intervals(*up_trig),  "rgba(31, 119, 180, 0.22)"))  # blau
    if up_assoc:
        intervals.append((str(up_assoc_label), _mk_intervals(*up_assoc), "rgba(255, 127, 14, 0.22)"))  # orange

    has_spindle_layers = bool(spindle_spont or spindle_trig or spindle_assoc)
    has_ripple_layers = bool(ripple_spont or ripple_trig or ripple_assoc)

    def _add_interval_spans(spans, fill, y0, y1):
        for (t0, t1) in spans:
            # auf ggf. gekürzte Zeitachse clippen
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=y0, y1=y1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor=fill
            ))

    # --- Schattierungen als Shapes (UP / Spindle / Ripple in getrennten Bändern)
    if has_spindle_layers and has_ripple_layers:
        up_y0, up_y1 = 0.00, 0.30
        spindle_y0, spindle_y1 = 0.35, 0.65
        ripple_y0, ripple_y1 = 0.70, 1.00
    elif has_spindle_layers:
        up_y0, up_y1 = 0.00, 0.48
        spindle_y0, spindle_y1 = 0.52, 1.00
        ripple_y0, ripple_y1 = 0.52, 1.00
    elif has_ripple_layers:
        up_y0, up_y1 = 0.00, 0.48
        spindle_y0, spindle_y1 = 0.52, 1.00
        ripple_y0, ripple_y1 = 0.52, 1.00
    else:
        up_y0, up_y1 = 0.00, 1.00
        spindle_y0, spindle_y1 = 0.00, 1.00
        ripple_y0, ripple_y1 = 0.00, 1.00
    for label, spans, fill in intervals:
        _add_interval_spans(spans, fill, up_y0, up_y1)

    spindle_layers = []
    if spindle_spont:
        spindle_layers.append((str(spindle_spont_label), _mk_intervals(*spindle_spont), "rgba(46, 204, 113, 0.38)"))
    if spindle_trig:
        spindle_layers.append((str(spindle_trig_label), _mk_intervals(*spindle_trig), "rgba(31, 119, 180, 0.38)"))
    if spindle_assoc:
        spindle_layers.append((str(spindle_assoc_label), _mk_intervals(*spindle_assoc), "rgba(255, 127, 14, 0.38)"))
    if spindle_layers:
        for _, spans, fill in spindle_layers:
            _add_interval_spans(spans, fill, spindle_y0, spindle_y1)

    ripple_layers = []
    if ripple_spont:
        ripple_layers.append((str(ripple_spont_label), _mk_intervals(*ripple_spont), "rgba(46, 204, 113, 0.46)"))
    if ripple_trig:
        ripple_layers.append((str(ripple_trig_label), _mk_intervals(*ripple_trig), "rgba(31, 119, 180, 0.46)"))
    if ripple_assoc:
        ripple_layers.append((str(ripple_assoc_label), _mk_intervals(*ripple_assoc), "rgba(255, 127, 14, 0.46)"))
    if ripple_layers:
        for _, spans, fill in ripple_layers:
            _add_interval_spans(spans, fill, ripple_y0, ripple_y1)

    if spindle_intervals:
        for (t0, t1) in spindle_intervals:
            t0 = float(t0); t1 = float(t1)
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                continue
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor="rgba(138, 43, 226, 0.30)"
            ))
    if ripple_intervals:
        for (t0, t1) in ripple_intervals:
            t0 = float(t0); t1 = float(t1)
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                continue
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(width=0),
                fillcolor="rgba(220, 20, 60, 0.22)"
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
    if show_pulse_intervals:
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
                line=dict(width=2, dash=dash, color="red")
            ))

    _add_pulses(pulse_times_1, "dot")
    _add_pulses(pulse_times_2, "dash")
        # --- Pulse-OFF-Linien (Offsets) in Rot

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
                line=dict(width=2, dash=dash, color="red")
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

    if show_pulse_intervals:
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
    if spindle_layers:
        for label, _, fill in spindle_layers:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=12, color=fill),
                name=label,
                showlegend=True
            ))
    if ripple_layers:
        for label, _, fill in ripple_layers:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=12, color=fill),
                name=label,
                showlegend=True
            ))
    if spindle_intervals:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color="rgba(138, 43, 226, 0.30)"),
            name="Spindles 10-15 Hz",
            showlegend=True
        ))
    if ripple_intervals:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color="rgba(220, 20, 60, 0.22)"),
            name="Sharp-wave ripples",
            showlegend=True
        ))
    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON"
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON"
        ))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 OFF"
        ))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 OFF"
        ))
    if show_pulse_intervals and (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.12)"),
            name="Pulse duration (ON→OFF)"
        ))


    yaxis_cfg = dict(
        title=y_label,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
    )
    if y_range is not None:
        yr = np.asarray(y_range, dtype=float).ravel()
        if yr.size >= 2 and np.isfinite(yr[0]) and np.isfinite(yr[1]) and yr[1] > yr[0]:
            yaxis_cfg["range"] = [float(yr[0]), float(yr[1])]
            yaxis_cfg["autorange"] = False

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Zeit (s)",
            rangeslider=dict(visible=True),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror="allticks",
            tickfont=dict(size=16),
        ),
        yaxis=yaxis_cfg,
        shapes=shapes,
        margin=dict(l=60, r=20, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

            # RICHTIG: nimm den Parameter pulse_times_1
    if pulse_times_1 is not None and len(pulse_times_1):
        for x in pulse_times_1:
            fig.add_vline(x=float(x), line_width=5, line_dash="solid", line_color="red")

    if pulse_times_1_off is not None and len(pulse_times_1_off):
        for x in pulse_times_1_off:
            fig.add_vline(x=float(x), line_width=4, line_dash="dot", line_color="red")


    out_html = os.path.join(save_dir, f"{base_tag}__lfp_interactive.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] interaktiver LFP-Plot: {out_html}")

    return out_html


def export_interactive_dual_lfp_html(
    base_tag, save_dir,
    time_s, y_top, y_bottom,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    top_spont=None,
    top_trig=None,
    top_assoc=None,
    bottom_spont=None,
    bottom_trig=None,
    bottom_assoc=None,
    top_spont_label="Spindle spontaneous",
    top_trig_label="Spindle triggered",
    top_assoc_label="Spindle associated",
    bottom_spont_label="UP spontaneous",
    bottom_trig_label="UP triggered",
    bottom_assoc_label="UP associated",
    max_points=300_000,
    title="Dual LFP (interaktiv)",
    top_y_label="10-15 Hz bandpass",
    bottom_y_label="LFP",
    y_range_top=None,
    y_range_bottom=None,
    show_pulse_intervals=True,
):
    t = np.asarray(time_s, dtype=float).ravel()
    x_top = np.asarray(y_top, dtype=float).ravel()
    x_bottom = np.asarray(y_bottom, dtype=float).ravel()

    m = min(t.size, x_top.size, x_bottom.size)
    t = t[:m]
    x_top = x_top[:m]
    x_bottom = x_bottom[:m]

    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        x_top = x_top[::step]
        x_bottom = x_bottom[::step]

    # For display only: add a smooth spindle envelope so bursts stay readable when zoomed in.
    x_top_env = np.abs(np.asarray(x_top, float))
    if x_top_env.size:
        dt_top = float(np.median(np.diff(t))) if t.size >= 2 and np.all(np.isfinite(t)) else np.nan
        win = int(round(0.08 / dt_top)) if np.isfinite(dt_top) and dt_top > 0 else 5
        win = max(5, win)
        win = min(win, max(5, x_top_env.size))
        if win > 1 and x_top_env.size >= win:
            ker = np.ones(int(win), dtype=float) / float(win)
            x_top_env = np.convolve(x_top_env, ker, mode="same")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.50],
    )
    fig.add_trace(go.Scatter(
        x=t, y=x_top, mode="lines", name="10-15 Hz bandpass",
        line=dict(color="magenta", width=1.2),
        opacity=0.45,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_top_env, mode="lines", name="Spindle envelope",
        line=dict(color="#8b0000", width=2.6)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_bottom, mode="lines", name="LFP"), row=2, col=1)

    shapes = []

    def _mk_intervals(UP, DOWN):
        if UP is None or DOWN is None:
            return []
        UP = np.asarray(UP, dtype=int)
        DOWN = np.asarray(DOWN, dtype=int)
        n = min(len(UP), len(DOWN))
        if n == 0:
            return []
        out = []
        for u, d in zip(UP[:n], DOWN[:n]):
            if 0 <= u < len(time_s) and 0 < d <= len(time_s) and d > u:
                out.append((float(time_s[u]), float(time_s[d - 1])))
        return out

    def _add_spans(groups, xref, yref):
        for _, spans, fill in groups:
            for (t0, t1) in spans:
                if len(t) and (t1 < t[0] or t0 > t[-1]):
                    continue
                shapes.append(dict(
                    type="rect",
                    x0=t0, x1=t1,
                    y0=0, y1=1,
                    xref=xref, yref=yref,
                    line=dict(width=0),
                    fillcolor=fill,
                ))

    top_groups = []
    if top_spont:
        top_groups.append((str(top_spont_label), _mk_intervals(*top_spont), "rgba(46, 204, 113, 0.30)"))
    if top_trig:
        top_groups.append((str(top_trig_label), _mk_intervals(*top_trig), "rgba(31, 119, 180, 0.30)"))
    if top_assoc:
        top_groups.append((str(top_assoc_label), _mk_intervals(*top_assoc), "rgba(255, 127, 14, 0.30)"))

    bottom_groups = []
    if bottom_spont:
        bottom_groups.append((str(bottom_spont_label), _mk_intervals(*bottom_spont), "rgba(46, 204, 113, 0.22)"))
    if bottom_trig:
        bottom_groups.append((str(bottom_trig_label), _mk_intervals(*bottom_trig), "rgba(31, 119, 180, 0.22)"))
    if bottom_assoc:
        bottom_groups.append((str(bottom_assoc_label), _mk_intervals(*bottom_assoc), "rgba(255, 127, 14, 0.22)"))

    _add_spans(top_groups, "x", "y domain")
    _add_spans(bottom_groups, "x2", "y2 domain")

    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals) / 2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0)
            t1 = float(t1)
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
                fillcolor=fill,
            ))

    def _add_pulse_lines(ts, dash, opacity, xref):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size / 1200))]
        for p in tt:
            if len(t) and (p < t[0] or p > t[-1]):
                continue
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref=xref, yref="paper",
                opacity=opacity,
                line=dict(width=2, dash=dash, color="red"),
            ))

    if show_pulse_intervals:
        _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.12)")
        _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.12)")

    for xref in ("x", "x2"):
        _add_pulse_lines(pulse_times_1, "dot", 0.35, xref)
        _add_pulse_lines(pulse_times_2, "dash", 0.35, xref)
        _add_pulse_lines(pulse_times_1_off, "dot", 0.55, xref)
        _add_pulse_lines(pulse_times_2_off, "dash", 0.55, xref)

    for label, _, fill in top_groups + bottom_groups:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color=fill),
            name=label,
        ))
    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON",
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON",
        ))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 OFF",
        ))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 OFF",
        ))
    if show_pulse_intervals and (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.12)"),
            name="Pulse duration (ON→OFF)",
        ))

    fig.update_layout(
        title=title,
        shapes=shapes,
        margin=dict(l=60, r=20, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        title_text="Zeit (s)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        row=2, col=1,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=top_y_label,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        title_font=dict(size=18),
        fixedrange=True,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=bottom_y_label,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=12),
        title_font=dict(size=14),
        row=2, col=1,
    )
    fig.update_xaxes(rangeslider=dict(visible=True), row=2, col=1)

    def _apply_y_range(range_vals, row, data=None, pad_frac=0.12):
        if range_vals is not None:
            yr = np.asarray(range_vals, dtype=float).ravel()
            if yr.size >= 2 and np.isfinite(yr[0]) and np.isfinite(yr[1]) and yr[1] > yr[0]:
                fig.update_yaxes(range=[float(yr[0]), float(yr[1])], autorange=False, row=row, col=1)
            return
        if data is None:
            return
        yy = np.asarray(data, dtype=float).ravel()
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return
        y0 = float(np.nanmin(yy))
        y1 = float(np.nanmax(yy))
        if not np.isfinite(y0) or not np.isfinite(y1):
            return
        if y1 <= y0:
            pad = max(abs(y0) * 0.1, 1.0)
            fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)
            return
        pad = (y1 - y0) * float(pad_frac)
        fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)

    _apply_y_range(y_range_top, 1, data=x_top, pad_frac=0.20)
    _apply_y_range(y_range_bottom, 2, data=x_bottom, pad_frac=0.08)

    out_html = os.path.join(save_dir, f"{base_tag}__dual_lfp_interactive.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] dual interaktiver LFP-Plot: {out_html}")

    return out_html


def export_interactive_two_channel_lfp_html(
    base_tag, save_dir,
    time_s, y_top, y_bottom,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    max_points=300_000,
    title="Two LFP channels (interaktiv)",
    top_name="Channel top",
    bottom_name="Channel bottom",
    top_y_label="LFP",
    bottom_y_label="LFP",
    y_range_top=None,
    y_range_bottom=None,
    show_pulse_intervals=True,
):
    t = np.asarray(time_s, dtype=float).ravel()
    x_top = np.asarray(y_top, dtype=float).ravel()
    x_bottom = np.asarray(y_bottom, dtype=float).ravel()

    m = min(t.size, x_top.size, x_bottom.size)
    t = t[:m]
    x_top = x_top[:m]
    x_bottom = x_bottom[:m]

    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        x_top = x_top[::step]
        x_bottom = x_bottom[::step]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.50],
    )
    fig.add_trace(go.Scatter(
        x=t, y=x_top, mode="lines", name=str(top_name),
        line=dict(color="#1f77b4", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_bottom, mode="lines", name=str(bottom_name),
        line=dict(color="#111111", width=1.0),
    ), row=2, col=1)

    shapes = []

    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals) / 2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0)
            t1 = float(t1)
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
                fillcolor=fill,
            ))

    def _add_pulse_lines(ts, dash, opacity, xref):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size / 1200))]
        for p in tt:
            if len(t) and (p < t[0] or p > t[-1]):
                continue
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref=xref, yref="paper",
                opacity=opacity,
                line=dict(width=2, dash=dash, color="red"),
            ))

    if show_pulse_intervals:
        _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.12)")
        _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.12)")

    for xref in ("x", "x2"):
        _add_pulse_lines(pulse_times_1, "dot", 0.35, xref)
        _add_pulse_lines(pulse_times_2, "dash", 0.35, xref)
        _add_pulse_lines(pulse_times_1_off, "dot", 0.55, xref)
        _add_pulse_lines(pulse_times_2_off, "dash", 0.55, xref)

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON",
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON",
        ))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 OFF",
        ))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 OFF",
        ))
    if show_pulse_intervals and (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.12)"),
            name="Pulse duration (ON→OFF)",
        ))

    fig.update_layout(
        title=title,
        shapes=shapes,
        margin=dict(l=60, r=20, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        title_text="Zeit (s)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        row=2, col=1,
    )
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=top_y_label,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=16),
        title_font=dict(size=18),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=bottom_y_label,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=12),
        title_font=dict(size=14),
        row=2, col=1,
    )
    fig.update_xaxes(rangeslider=dict(visible=True), row=2, col=1)

    def _apply_y_range(range_vals, row, data=None, pad_frac=0.12):
        if range_vals is not None:
            yr = np.asarray(range_vals, dtype=float).ravel()
            if yr.size >= 2 and np.isfinite(yr[0]) and np.isfinite(yr[1]) and yr[1] > yr[0]:
                fig.update_yaxes(range=[float(yr[0]), float(yr[1])], autorange=False, row=row, col=1)
            return
        if data is None:
            return
        yy = np.asarray(data, dtype=float).ravel()
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return
        y0 = float(np.nanmin(yy))
        y1 = float(np.nanmax(yy))
        if not np.isfinite(y0) or not np.isfinite(y1):
            return
        if y1 <= y0:
            pad = max(abs(y0) * 0.1, 1.0)
            fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)
            return
        pad = (y1 - y0) * float(pad_frac)
        fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)

    _apply_y_range(y_range_top, 1, data=x_top, pad_frac=0.08)
    _apply_y_range(y_range_bottom, 2, data=x_bottom, pad_frac=0.08)

    out_html = os.path.join(save_dir, f"{base_tag}__dual_channel_no_spindle.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] dual channel (no spindle): {out_html}")

    return out_html


def export_interactive_three_channel_lfp_html(
    base_tag, save_dir,
    time_s, y_top, y_mid, y_bottom,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    top_spont=None,
    top_trig=None,
    top_assoc=None,
    mid_spont=None,
    mid_trig=None,
    mid_assoc=None,
    bottom_spont=None,
    bottom_trig=None,
    bottom_assoc=None,
    top_spont_label="SWR spontaneous",
    top_trig_label="SWR triggered",
    top_assoc_label="SWR associated",
    mid_spont_label="UP spontaneous",
    mid_trig_label="UP triggered",
    mid_assoc_label="UP associated",
    bottom_spont_label="Spindle spontaneous",
    bottom_trig_label="Spindle triggered",
    bottom_assoc_label="Spindle associated",
    max_points=300_000,
    title="Three-channel LFP (interaktiv)",
    top_name="Channel top",
    mid_name="Channel mid",
    bottom_name="Channel bottom",
    top_y_label="Top",
    mid_y_label="Mid",
    bottom_y_label="Bottom",
    y_range_top=None,
    y_range_mid=None,
    y_range_bottom=None,
    show_pulse_intervals=True,
):
    t = np.asarray(time_s, dtype=float).ravel()
    x_top = np.asarray(y_top, dtype=float).ravel()
    x_mid = np.asarray(y_mid, dtype=float).ravel()
    x_bottom = np.asarray(y_bottom, dtype=float).ravel()

    m = min(t.size, x_top.size, x_mid.size, x_bottom.size)
    t = t[:m]
    x_top = x_top[:m]
    x_mid = x_mid[:m]
    x_bottom = x_bottom[:m]

    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        x_top = x_top[::step]
        x_mid = x_mid[::step]
        x_bottom = x_bottom[::step]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.34, 0.33, 0.33],
    )
    fig.add_trace(go.Scatter(
        x=t, y=x_top, mode="lines", name=str(top_name),
        line=dict(color="#8b0000", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_mid, mode="lines", name=str(mid_name),
        line=dict(color="#111111", width=1.0),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_bottom, mode="lines", name=str(bottom_name),
        line=dict(color="#cc00cc", width=1.0),
    ), row=3, col=1)

    shapes = []

    def _mk_intervals(UP, DOWN):
        if UP is None or DOWN is None:
            return []
        UP = np.asarray(UP, dtype=int)
        DOWN = np.asarray(DOWN, dtype=int)
        n = min(len(UP), len(DOWN))
        if n == 0:
            return []
        out = []
        for u, d in zip(UP[:n], DOWN[:n]):
            if 0 <= u < len(time_s) and 0 < d <= len(time_s) and d > u:
                out.append((float(time_s[u]), float(time_s[d - 1])))
        return out

    def _add_spans(groups, xref, yref):
        for _, spans, fill in groups:
            for (t0, t1) in spans:
                if len(t) and (t1 < t[0] or t0 > t[-1]):
                    continue
                shapes.append(dict(
                    type="rect",
                    x0=t0, x1=t1,
                    y0=0, y1=1,
                    xref=xref, yref=yref,
                    line=dict(width=0),
                    fillcolor=fill,
                ))

    top_groups = []
    if top_spont:
        top_groups.append((str(top_spont_label), _mk_intervals(*top_spont), "rgba(46, 204, 113, 0.34)"))
    if top_trig:
        top_groups.append((str(top_trig_label), _mk_intervals(*top_trig), "rgba(31, 119, 180, 0.34)"))
    if top_assoc:
        top_groups.append((str(top_assoc_label), _mk_intervals(*top_assoc), "rgba(255, 127, 14, 0.34)"))

    mid_groups = []
    if mid_spont:
        mid_groups.append((str(mid_spont_label), _mk_intervals(*mid_spont), "rgba(46, 204, 113, 0.22)"))
    if mid_trig:
        mid_groups.append((str(mid_trig_label), _mk_intervals(*mid_trig), "rgba(31, 119, 180, 0.22)"))
    if mid_assoc:
        mid_groups.append((str(mid_assoc_label), _mk_intervals(*mid_assoc), "rgba(255, 127, 14, 0.22)"))

    bottom_groups = []
    if bottom_spont:
        bottom_groups.append((str(bottom_spont_label), _mk_intervals(*bottom_spont), "rgba(46, 204, 113, 0.34)"))
    if bottom_trig:
        bottom_groups.append((str(bottom_trig_label), _mk_intervals(*bottom_trig), "rgba(31, 119, 180, 0.34)"))
    if bottom_assoc:
        bottom_groups.append((str(bottom_assoc_label), _mk_intervals(*bottom_assoc), "rgba(255, 127, 14, 0.34)"))

    _add_spans(top_groups, "x", "y domain")
    _add_spans(mid_groups, "x2", "y2 domain")
    _add_spans(bottom_groups, "x3", "y3 domain")

    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals) / 2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0)
            t1 = float(t1)
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
                fillcolor=fill,
            ))

    def _add_pulse_lines(ts, dash, opacity, xref):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size / 1200))]
        for p in tt:
            if len(t) and (p < t[0] or p > t[-1]):
                continue
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref=xref, yref="paper",
                opacity=opacity,
                line=dict(width=2, dash=dash, color="red"),
            ))

    if show_pulse_intervals:
        _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.10)")
        _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.10)")

    for xref in ("x", "x2", "x3"):
        _add_pulse_lines(pulse_times_1, "dot", 0.35, xref)
        _add_pulse_lines(pulse_times_2, "dash", 0.35, xref)
        _add_pulse_lines(pulse_times_1_off, "dot", 0.55, xref)
        _add_pulse_lines(pulse_times_2_off, "dash", 0.55, xref)

    for label, _, fill in top_groups + mid_groups + bottom_groups:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color=fill),
            name=label,
        ))

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON",
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON",
        ))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 OFF",
        ))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 OFF",
        ))
    if show_pulse_intervals and (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.10)"),
            name="Pulse duration (ON→OFF)",
        ))

    fig.update_layout(
        title=title,
        shapes=shapes,
        margin=dict(l=60, r=20, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        title_text="Zeit (s)",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror="allticks",
        tickfont=dict(size=14),
        row=3, col=1,
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=12), row=2, col=1)
    fig.update_yaxes(title_text=top_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=1, col=1)
    fig.update_yaxes(title_text=mid_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=2, col=1)
    fig.update_yaxes(title_text=bottom_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=3, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True), row=3, col=1)

    def _apply_y_range(range_vals, row, data=None, pad_frac=0.10):
        if range_vals is not None:
            yr = np.asarray(range_vals, dtype=float).ravel()
            if yr.size >= 2 and np.isfinite(yr[0]) and np.isfinite(yr[1]) and yr[1] > yr[0]:
                fig.update_yaxes(range=[float(yr[0]), float(yr[1])], autorange=False, row=row, col=1)
            return
        if data is None:
            return
        yy = np.asarray(data, dtype=float).ravel()
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return
        y0 = float(np.nanmin(yy))
        y1 = float(np.nanmax(yy))
        if not np.isfinite(y0) or not np.isfinite(y1):
            return
        if y1 <= y0:
            pad = max(abs(y0) * 0.1, 1.0)
            fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)
            return
        pad = (y1 - y0) * float(pad_frac)
        fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)

    _apply_y_range(y_range_top, 1, data=x_top, pad_frac=0.10)
    _apply_y_range(y_range_mid, 2, data=x_mid, pad_frac=0.08)
    _apply_y_range(y_range_bottom, 3, data=x_bottom, pad_frac=0.10)

    out_html = os.path.join(save_dir, f"{base_tag}__triple_lfp_interactive.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] triple interaktiver LFP-Plot: {out_html}")
    return out_html


def export_interactive_spectrogram_html(
    base_tag, save_dir, spect_dat,
    spindle_trace, spindle_time_s,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    up_spont=None,
    up_trig=None,
    up_assoc=None,
    up_spont_label="UP spontaneous",
    up_trig_label="UP triggered",
    up_assoc_label="UP associated",
    max_points=300_000,
    title="Spectrogram + spindle bandpass (interaktiv)",
    spectrogram_label="Power (norm.)",
    spindle_label="10-15 Hz bandpass",
    spindle_y_label="Amplitude",
):
    S = np.asarray(spect_dat[0], dtype=float)
    t_feat = np.asarray(spect_dat[1], dtype=float).ravel()
    freqs = np.asarray(spect_dat[2], dtype=float).ravel()
    x = np.asarray(spindle_time_s, dtype=float).ravel()
    y = np.asarray(spindle_trace, dtype=float).ravel()

    if S.ndim != 2 or t_feat.size == 0 or freqs.size == 0:
        raise ValueError("Invalid spectrogram data for HTML export.")

    if x.size != y.size:
        m = min(x.size, y.size)
        x = x[:m]
        y = y[:m]

    if x.size > max_points:
        step = int(np.ceil(x.size / max_points))
        x = x[::step]
        y = y[::step]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.30, 0.70],
    )
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=spindle_label,
        line=dict(color="magenta", width=1.4),
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        x=t_feat,
        y=freqs,
        z=S,
        colorscale="Viridis",
        colorbar=dict(title=spectrogram_label),
        name=spectrogram_label,
        showscale=True,
    ), row=2, col=1)

    shapes = []

    def _mk_intervals(UP, DOWN):
        if UP is None or DOWN is None:
            return []
        tt = np.asarray(spindle_time_s, dtype=float)
        if tt.size == 0:
            return []
        UP = np.asarray(UP, dtype=int)
        DOWN = np.asarray(DOWN, dtype=int)
        m = min(len(UP), len(DOWN))
        if m == 0:
            return []
        out = []
        for u, d in zip(UP[:m], DOWN[:m]):
            if 0 <= u < tt.size and 0 < d <= tt.size and d > u:
                out.append((float(tt[u]), float(tt[d - 1])))
        return out

    intervals = []
    if up_spont:
        intervals.append((str(up_spont_label), _mk_intervals(*up_spont), "rgba(46, 204, 113, 0.20)"))
    if up_trig:
        intervals.append((str(up_trig_label), _mk_intervals(*up_trig), "rgba(31, 119, 180, 0.20)"))
    if up_assoc:
        intervals.append((str(up_assoc_label), _mk_intervals(*up_assoc), "rgba(255, 127, 14, 0.20)"))

    def _add_pulse_intervals(intervals, fill):
        if intervals is None or len(intervals) == 0:
            return
        if len(intervals) > 2000:
            step = int(np.ceil(len(intervals) / 2000))
            intervals = intervals[::step]
        for (t0, t1) in intervals:
            t0 = float(t0)
            t1 = float(t1)
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                continue
            if t_feat.size and (t1 < t_feat[0] or t0 > t_feat[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x2", yref="paper",
                line=dict(width=0),
                fillcolor=fill,
            ))

    def _add_pulse_lines(ts, dash, opacity, xref):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float)
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size / 1200))]
        for p in tt:
            if t_feat.size and (p < t_feat[0] or p > t_feat[-1]):
                continue
            shapes.append(dict(
                type="line",
                x0=float(p), x1=float(p),
                y0=0, y1=1,
                xref=xref, yref="paper",
                opacity=opacity,
                line=dict(width=2, dash=dash, color="red"),
            ))

    for _, spans, fill in intervals:
        for (t0, t1) in spans:
            if t_feat.size and (t1 < t_feat[0] or t0 > t_feat[-1]):
                continue
            shapes.append(dict(
                type="rect",
                x0=t0, x1=t1,
                y0=0, y1=1,
                xref="x2", yref="y2 domain",
                line=dict(width=0),
                fillcolor=fill,
            ))

    _add_pulse_intervals(pulse_intervals_1, "rgba(255, 0, 0, 0.12)")
    _add_pulse_intervals(pulse_intervals_2, "rgba(255, 0, 0, 0.12)")
    _add_pulse_lines(pulse_times_1, "dot", 0.35, "x")
    _add_pulse_lines(pulse_times_2, "dash", 0.35, "x")
    _add_pulse_lines(pulse_times_1_off, "dot", 0.55, "x")
    _add_pulse_lines(pulse_times_2_off, "dash", 0.55, "x")
    _add_pulse_lines(pulse_times_1, "dot", 0.35, "x2")
    _add_pulse_lines(pulse_times_2, "dash", 0.35, "x2")
    _add_pulse_lines(pulse_times_1_off, "dot", 0.55, "x2")
    _add_pulse_lines(pulse_times_2_off, "dash", 0.55, "x2")

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON",
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON",
        ))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 OFF",
        ))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 OFF",
        ))
    if (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.12)"),
            name="Pulse duration (ON→OFF)",
        ))
    if intervals:
        for label, _, fill in intervals:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=12, color=fill),
                name=label,
            ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Zeit (s)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror="allticks",
        ),
        xaxis2=dict(
            title="Zeit (s)",
            rangeslider=dict(visible=True),
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror="allticks",
        ),
        yaxis=dict(
            title=spindle_y_label,
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror="allticks",
        ),
        yaxis2=dict(
            title="Frequenz (Hz)",
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror="allticks",
        ),
        shapes=shapes,
        margin=dict(l=60, r=60, t=50, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_html = os.path.join(save_dir, f"{base_tag}__spectrogram_interactive.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] interaktiver Spectrogramm-Plot: {out_html}")

    return out_html


def export_interactive_swr_scan_html(
    base_tag,
    save_dir,
    time_s,
    raw_signals,
    bp_signals,
    swr_intervals_by_channel,
    *,
    channel_indices=None,
    pulse_times_1=None,
    pulse_times_2=None,
    max_points=60_000,
    title="SWR scan (raw + ripple bandpass)",
    y_label="Amplitude",
):
    t = np.asarray(time_s, dtype=float).ravel()
    X_raw = np.asarray(raw_signals, dtype=float)
    X_bp = np.asarray(bp_signals, dtype=float)

    if X_raw.ndim != 2 or X_bp.ndim != 2:
        raise ValueError("raw_signals and bp_signals must be 2D arrays [n_channels, n_time].")
    if X_raw.shape != X_bp.shape:
        raise ValueError("raw_signals and bp_signals must have the same shape.")
    if X_raw.shape[1] != t.size:
        raise ValueError("time_s length must match signal length.")

    n_ch = int(X_raw.shape[0])
    if n_ch <= 0:
        raise ValueError("No channels provided for SWR scan HTML export.")

    if channel_indices is None:
        channel_indices = list(range(n_ch))
    else:
        channel_indices = [int(v) for v in channel_indices]
        if len(channel_indices) != n_ch:
            raise ValueError("channel_indices length must match number of channels.")

    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        X_raw = X_raw[:, ::step]
        X_bp = X_bp[:, ::step]

    def _robust_scale(y):
        yy = np.asarray(y, float).ravel()
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return 1.0
        s = float(np.nanpercentile(np.abs(yy), 95))
        if (not np.isfinite(s)) or s <= 1e-12:
            return 1.0
        return s

    # Scale the bandpass for visibility while keeping raw in original units.
    X_bp_vis = np.zeros_like(X_bp, dtype=float)
    for i in range(n_ch):
        s_raw = _robust_scale(X_raw[i])
        s_bp = _robust_scale(X_bp[i])
        gain = (0.75 * s_raw) / max(s_bp, 1e-12)
        X_bp_vis[i] = np.asarray(X_bp[i], float) * float(gain)

    fig = make_subplots(
        rows=n_ch,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.002, 0.02 / max(n_ch, 1)),
        row_heights=[1.0 / n_ch] * n_ch,
    )

    for i in range(n_ch):
        ch = int(channel_indices[i])
        show_leg = (i == 0)
        fig.add_trace(
            go.Scattergl(
                x=t, y=X_raw[i], mode="lines",
                name="Raw signal",
                line=dict(color="#202020", width=1.0),
                opacity=0.90,
                legendgroup="raw",
                showlegend=show_leg,
            ),
            row=i + 1, col=1
        )
        fig.add_trace(
            go.Scattergl(
                x=t, y=X_bp_vis[i], mode="lines",
                name="Ripple bandpass (scaled)",
                line=dict(color="#b30000", width=1.0),
                opacity=0.75,
                legendgroup="bp",
                showlegend=show_leg,
            ),
            row=i + 1, col=1
        )

        spans = swr_intervals_by_channel[i] if i < len(swr_intervals_by_channel) else []
        first_swr = True
        for (t0, t1) in spans:
            t0 = float(t0)
            t1 = float(t1)
            if (not np.isfinite(t0)) or (not np.isfinite(t1)) or (t1 <= t0):
                continue
            if len(t) and (t1 < t[0] or t0 > t[-1]):
                continue
            fig.add_vrect(
                x0=t0, x1=t1,
                fillcolor="rgba(220, 20, 60, 0.20)",
                line_width=0,
                row=i + 1, col=1,
            )
            if first_swr and show_leg:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        line=dict(width=10, color="rgba(220, 20, 60, 0.30)"),
                        name="SWR interval",
                        showlegend=True,
                    )
                )
                first_swr = False

        fig.update_yaxes(title_text=f"pri_{ch}", row=i + 1, col=1)

    def _add_pulse_lines(ts, dash):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float).ravel()
        tt = tt[np.isfinite(tt)]
        if tt.size > 1200:
            tt = tt[::int(np.ceil(tt.size / 1200))]
        for i in range(n_ch):
            for p in tt:
                if len(t) and (p < t[0] or p > t[-1]):
                    continue
                fig.add_vline(
                    x=float(p),
                    line=dict(color="red", width=1.1, dash=dash),
                    opacity=0.30,
                    row=i + 1, col=1,
                )

    _add_pulse_lines(pulse_times_1, "dot")
    _add_pulse_lines(pulse_times_2, "dash")

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name="Pulse 1 ON",
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name="Pulse 2 ON",
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=70, r=20, t=55, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Zeit (s)",
        showline=True, linewidth=1.5, linecolor="black", mirror="allticks",
        row=n_ch, col=1
    )
    fig.update_xaxes(rangeslider=dict(visible=True), row=n_ch, col=1)
    fig.update_yaxes(showline=True, linewidth=1.2, linecolor="black", mirror="allticks")

    out_html = os.path.join(save_dir, f"{base_tag}__SWR_CH00_17_SCAN.html")
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    print(f"[HTML] SWR scan interactive: {out_html}")
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


import os
import struct
import numpy as np

def read_nev_ttl_events(nev_path):
    """
    Returns arrays: ts_us (uint64), ttl (uint16), event_id (uint16), event_str (list[str])
    NEV format: 16 kB ASCII header, then fixed-size records (184 bytes).
    """
    RECORD_SIZE = 184
    HEADER_SIZE = 16 * 1024

    ts_list = []
    ttl_list = []
    id_list  = []
    str_list = []

    with open(nev_path, "rb") as f:
        f.seek(HEADER_SIZE)

        while True:
            rec = f.read(RECORD_SIZE)
            if len(rec) < RECORD_SIZE:
                break

            # Neuralynx NEV record layout (common):
            # uint64 TimeStamp
            # uint16 EventID
            # uint16 TTL
            # uint16 CRC
            # uint16 Dummy1
            # uint16 Dummy2
            # int32 ExtraData[8]
            # char EventString[128]
            ts_us, event_id, ttl = struct.unpack_from("<QHH", rec, 0)

            # EventString starts after 8+2+2+2+2+2 + 8*4 = 48 bytes
            # 0..47 = fixed fields, 48..175 = event string (128 bytes)
            ev_raw = rec[48:48+128]
            ev_str = ev_raw.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()

            ts_list.append(ts_us)
            ttl_list.append(ttl)
            id_list.append(event_id)
            str_list.append(ev_str)

    ts_us = np.asarray(ts_list, dtype=np.uint64)
    ttl   = np.asarray(ttl_list, dtype=np.uint16)
    eid   = np.asarray(id_list,  dtype=np.uint16)
    return ts_us, ttl, eid, str_list


def ttl_on_off_from_nev(ts_us, ttl, *, bitmask=None):
    """
    Compute (on_us, off_us) from TTL transitions.
    If bitmask is None: uses ttl != 0 as 'high'.
    If bitmask is given (e.g. 1<<0): uses that bit only.
    """
    ts_us = np.asarray(ts_us, dtype=np.uint64)
    ttl   = np.asarray(ttl, dtype=np.uint16)

    if ts_us.size == 0:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.uint64)

    if bitmask is None:
        high = (ttl != 0).astype(np.int8)
    else:
        high = ((ttl & np.uint16(bitmask)) != 0).astype(np.int8)

    # transitions:
    dh = np.diff(high)
    on_idx  = np.where(dh ==  1)[0] + 1
    off_idx = np.where(dh == -1)[0] + 1

    # Handle if starts already high
    if high[0] == 1:
        on_idx = np.r_[0, on_idx]
    # Handle if ends high
    if high[-1] == 1:
        off_idx = np.r_[off_idx, high.size - 1]

    m = min(on_idx.size, off_idx.size)
    on_idx, off_idx = on_idx[:m], off_idx[:m]

    on_us  = ts_us[on_idx].astype(np.uint64)
    off_us = ts_us[off_idx].astype(np.uint64)

    # guard: ensure off > on
    good = off_us > on_us
    return on_us[good], off_us[good]
