import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
from plotly.subplots import make_subplots
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm
import os
import sys, os
import html


ANALYSE_IN_AU = True
HTML_IN_uV    = True

_DEFAULT_SESSION = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/"
BASE_PATH   = globals().get("BASE_PATH", _DEFAULT_SESSION)
SAVE_DIR = BASE_PATH
LOGFILE = os.path.join(SAVE_DIR, "runlog.txt")


def _write_plotly_html(fig, out_html, title):
    # stamps the browser-tab <title> with the plot title so the channel is visible across tabs
    plotly_offline_plot(fig, filename=out_html, auto_open=False, include_plotlyjs="cdn")
    title_text = title.get("text") if isinstance(title, dict) else title
    if title_text:
        with open(out_html, "r", encoding="utf-8") as f:
            content = f.read()
        if "<title>" not in content:
            content = content.replace(
                "<head>", f"<head><title>{html.escape(str(title_text))}</title>", 1
            )
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(content)


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
    exclude_intervals=None,  # list[(t0, t1)] in Sekunden, z.B. automatisch erkannte Artefakte
    max_points=600_000,
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
    if exclude_intervals:
        for (t0, t1) in exclude_intervals:
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
                line=dict(width=1, color="rgba(80, 80, 80, 0.6)"),
                fillcolor="rgba(120, 120, 120, 0.6)",
                layer="above",  # ueber der Kurve, damit sie schwach durchscheint
            ))

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
        _add_pulse_intervals(pulse_intervals_1, "rgba(255, 50, 50, 0.45)")
        _add_pulse_intervals(pulse_intervals_2, "rgba(255, 50, 50, 0.45)")


    


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
    if exclude_intervals:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color="rgba(120, 120, 120, 0.6)"),
            name="Artifact (excluded)",
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
            line=dict(width=12, color="rgba(255, 50, 50, 0.45)"),
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
        margin=dict(l=60, r=20, t=85, b=50),
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
    _write_plotly_html(fig, out_html, title)
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
    max_points=600_000,
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
        margin=dict(l=60, r=20, t=85, b=50),
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
        y0 = float(np.nanpercentile(yy, 1))
        y1 = float(np.nanpercentile(yy, 99))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
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
    _write_plotly_html(fig, out_html, title)
    print(f"[HTML] dual interaktiver LFP-Plot: {out_html}")

    return out_html


def export_interactive_two_channel_lfp_html(
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
    top_spont_label="UP spontaneous",
    top_trig_label="UP triggered",
    top_assoc_label="UP associated",
    bottom_spont_label="UP spontaneous",
    bottom_trig_label="UP triggered",
    bottom_assoc_label="UP associated",
    max_points=600_000,
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
        top_groups.append((str(top_spont_label), _mk_intervals(*top_spont), "rgba(46, 204, 113, 0.22)"))
    if top_trig:
        top_groups.append((str(top_trig_label), _mk_intervals(*top_trig), "rgba(31, 119, 180, 0.22)"))
    if top_assoc:
        top_groups.append((str(top_assoc_label), _mk_intervals(*top_assoc), "rgba(255, 127, 14, 0.22)"))

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
        margin=dict(l=60, r=20, t=85, b=50),
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
        y0 = float(np.nanpercentile(yy, 1))
        y1 = float(np.nanpercentile(yy, 99))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
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
    _write_plotly_html(fig, out_html, title)
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
    trace_time_shifts_s=None,
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
        margin=dict(l=60, r=20, t=85, b=50),
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
        y0 = float(np.nanpercentile(yy, 1))
        y1 = float(np.nanpercentile(yy, 99))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
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
    _write_plotly_html(fig, out_html, title)
    print(f"[HTML] triple interaktiver LFP-Plot: {out_html}")
    return out_html


def export_interactive_four_channel_lfp_html(
    base_tag, save_dir,
    time_s, y_raw_top, y_top, y_mid, y_bottom,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    pulse_intervals_1=None, pulse_intervals_2=None,
    *,
    raw_top_spont=None,
    raw_top_trig=None,
    raw_top_assoc=None,
    top_spont=None,
    top_trig=None,
    top_assoc=None,
    mid_spont=None,
    mid_trig=None,
    mid_assoc=None,
    bottom_spont=None,
    bottom_trig=None,
    bottom_assoc=None,
    raw_top_spont_label="SWR spontaneous",
    raw_top_trig_label="SWR triggered",
    raw_top_assoc_label="SWR associated",
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
    title="Four-channel LFP (interaktiv)",
    raw_top_name="Channel raw",
    top_name="Channel top",
    mid_name="Channel mid",
    bottom_name="Channel bottom",
    raw_top_y_label="Raw",
    top_y_label="Top",
    mid_y_label="Mid",
    bottom_y_label="Bottom",
    y_range_raw_top=None,
    y_range_top=None,
    y_range_mid=None,
    y_range_bottom=None,
    show_pulse_intervals=True,
    trace_time_shifts_s=None,
):
    t = np.asarray(time_s, dtype=float).ravel()
    x_raw_top = np.asarray(y_raw_top, dtype=float).ravel()
    x_top = np.asarray(y_top, dtype=float).ravel()
    x_mid = np.asarray(y_mid, dtype=float).ravel()
    x_bottom = np.asarray(y_bottom, dtype=float).ravel()

    m = min(t.size, x_raw_top.size, x_top.size, x_mid.size, x_bottom.size)
    t = t[:m]
    x_raw_top = x_raw_top[:m]
    x_top = x_top[:m]
    x_mid = x_mid[:m]
    x_bottom = x_bottom[:m]

    if t.size > max_points:
        step = int(np.ceil(t.size / max_points))
        t = t[::step]
        x_raw_top = x_raw_top[::step]
        x_top = x_top[::step]
        x_mid = x_mid[::step]
        x_bottom = x_bottom[::step]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.24, 0.26, 0.25, 0.25],
    )
    fig.add_trace(go.Scatter(
        x=t, y=x_raw_top, mode="lines", name=str(raw_top_name),
        line=dict(color="#444444", width=0.9),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_top, mode="lines", name=str(top_name),
        line=dict(color="#8b0000", width=1.2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_mid, mode="lines", name=str(mid_name),
        line=dict(color="#111111", width=1.0),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=x_bottom, mode="lines", name=str(bottom_name),
        line=dict(color="#cc00cc", width=1.0),
    ), row=4, col=1)

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

    raw_top_groups = []
    if raw_top_spont:
        raw_top_groups.append((str(raw_top_spont_label), _mk_intervals(*raw_top_spont), "rgba(39, 174, 96, 0.32)"))
    if raw_top_trig:
        raw_top_groups.append((str(raw_top_trig_label), _mk_intervals(*raw_top_trig), "rgba(39, 174, 96, 0.42)"))
    if raw_top_assoc:
        raw_top_groups.append((str(raw_top_assoc_label), _mk_intervals(*raw_top_assoc), "rgba(39, 174, 96, 0.52)"))

    top_groups = []
    if top_spont:
        top_groups.append((str(top_spont_label), _mk_intervals(*top_spont), "rgba(39, 174, 96, 0.32)"))
    if top_trig:
        top_groups.append((str(top_trig_label), _mk_intervals(*top_trig), "rgba(39, 174, 96, 0.42)"))
    if top_assoc:
        top_groups.append((str(top_assoc_label), _mk_intervals(*top_assoc), "rgba(39, 174, 96, 0.52)"))

    mid_groups = []
    if mid_spont:
        mid_groups.append((str(mid_spont_label), _mk_intervals(*mid_spont), "rgba(52, 152, 219, 0.22)"))
    if mid_trig:
        mid_groups.append((str(mid_trig_label), _mk_intervals(*mid_trig), "rgba(52, 152, 219, 0.34)"))
    if mid_assoc:
        mid_groups.append((str(mid_assoc_label), _mk_intervals(*mid_assoc), "rgba(52, 152, 219, 0.46)"))

    bottom_groups = []
    if bottom_spont:
        bottom_groups.append((str(bottom_spont_label), _mk_intervals(*bottom_spont), "rgba(214, 51, 132, 0.28)"))
    if bottom_trig:
        bottom_groups.append((str(bottom_trig_label), _mk_intervals(*bottom_trig), "rgba(214, 51, 132, 0.40)"))
    if bottom_assoc:
        bottom_groups.append((str(bottom_assoc_label), _mk_intervals(*bottom_assoc), "rgba(214, 51, 132, 0.52)"))

    _add_spans(raw_top_groups, "x", "y domain")
    _add_spans(top_groups, "x2", "y2 domain")
    _add_spans(mid_groups, "x3", "y3 domain")
    _add_spans(bottom_groups, "x4", "y4 domain")

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

    for xref in ("x", "x2", "x3", "x4"):
        _add_pulse_lines(pulse_times_1, "dot", 0.35, xref)
        _add_pulse_lines(pulse_times_2, "dash", 0.35, xref)
        _add_pulse_lines(pulse_times_1_off, "dot", 0.55, xref)
        _add_pulse_lines(pulse_times_2_off, "dash", 0.55, xref)

    for label, _, fill in raw_top_groups + top_groups + mid_groups + bottom_groups:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=12, color=fill),
            name=label,
        ))

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, dash="dot", color="red"), name="Pulse 1 ON"))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, dash="dash", color="red"), name="Pulse 2 ON"))
    if pulse_times_1_off is not None and len(pulse_times_1_off):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, dash="dot", color="red"), name="Pulse 1 OFF"))
    if pulse_times_2_off is not None and len(pulse_times_2_off):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, dash="dash", color="red"), name="Pulse 2 OFF"))
    if show_pulse_intervals and (
        (pulse_intervals_1 is not None and len(pulse_intervals_1)) or
        (pulse_intervals_2 is not None and len(pulse_intervals_2))
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=12, color="rgba(255, 0, 0, 0.10)"),
            name="Pulse duration (ON->OFF)",
        ))

    fig.update_layout(
        title=title,
        shapes=shapes,
        margin=dict(l=60, r=20, t=85, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Zeit (s)", showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=14), row=4, col=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=12), row=2, col=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror="allticks", tickfont=dict(size=12), row=3, col=1)
    fig.update_yaxes(title_text=raw_top_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=1, col=1)
    fig.update_yaxes(title_text=top_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=2, col=1)
    fig.update_yaxes(title_text=mid_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=3, col=1)
    fig.update_yaxes(title_text=bottom_y_label, showline=True, linewidth=2, linecolor="black", mirror="allticks", row=4, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True), row=4, col=1)

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
        y0 = float(np.nanpercentile(yy, 1))
        y1 = float(np.nanpercentile(yy, 99))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
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

    _apply_y_range(y_range_raw_top, 1, data=x_raw_top, pad_frac=0.08)
    _apply_y_range(y_range_top, 2, data=x_top, pad_frac=0.10)
    _apply_y_range(y_range_mid, 3, data=x_mid, pad_frac=0.08)
    _apply_y_range(y_range_bottom, 4, data=x_bottom, pad_frac=0.10)

    out_html = os.path.join(save_dir, f"{base_tag}__four_lfp_interactive.html")
    _write_plotly_html(fig, out_html, title)
    print(f"[HTML] four interaktiver LFP-Plot: {out_html}")
    return out_html


def export_pulse_qa_four_channel_html(
    base_tag, save_dir,
    time_s, y_ripple, y_sharp, y_up, y_spindle,
    pulse_times_1=None, pulse_times_2=None,
    pulse_times_1_off=None, pulse_times_2_off=None,
    *,
    swr_spont=None,
    swr_trig=None,
    swr_assoc=None,
    up_spont=None,
    up_trig=None,
    up_assoc=None,
    spindle_spont=None,
    spindle_trig=None,
    spindle_assoc=None,
    pre_s=0.25,
    post_s=0.75,
    max_pulses=0,
    max_points_per_panel=2500,
    title="Pulse QA: SWR / Sharp-wave / UP / Spindle",
    ripple_name="Ripple band",
    sharp_name="Sharp-wave band",
    up_name="UP channel",
    spindle_name="Spindle band",
    ripple_y_label="Ripple",
    sharp_y_label="Sharp-wave",
    up_y_label="UP",
    spindle_y_label="Spindle",
    raw_swr=None,
    raw_swr_name="Raw SWR channel",
    raw_swr_y_label="Raw SWR",
    show_pulse_durations=True,
    require_full_window=True,
    trace_time_shifts_s=None,
):
    t_all = np.asarray(time_s, dtype=float).ravel()
    traces = [
        np.asarray(y_ripple, dtype=float).ravel(),
        np.asarray(y_sharp, dtype=float).ravel(),
        np.asarray(y_up, dtype=float).ravel(),
        np.asarray(y_spindle, dtype=float).ravel(),
    ]
    has_raw_swr = raw_swr is not None
    if has_raw_swr:
        traces = [np.asarray(raw_swr, dtype=float).ravel()] + traces
    m = min([t_all.size] + [x.size for x in traces])
    if m < 3:
        raise ValueError("Not enough points for pulse QA export.")
    t_all = t_all[:m]
    traces = [x[:m] for x in traces]
    n_rows = len(traces)
    panel_tag = f"{n_rows}panel"

    def _clean_times(ts):
        if ts is None:
            return np.array([], dtype=float)
        tt = np.asarray(ts, dtype=float).ravel()
        tt = tt[np.isfinite(tt)]
        return tt

    p1 = _clean_times(pulse_times_1)
    p2 = _clean_times(pulse_times_2)
    p1_off = _clean_times(pulse_times_1_off)
    p2_off = _clean_times(pulse_times_2_off)

    pulses = []
    for i, p in enumerate(p1):
        off = float(p1_off[i]) if i < p1_off.size and np.isfinite(p1_off[i]) and p1_off[i] > p else None
        pulses.append((float(p), "Pulse 1", i + 1, off))
    for i, p in enumerate(p2):
        off = float(p2_off[i]) if i < p2_off.size and np.isfinite(p2_off[i]) and p2_off[i] > p else None
        pulses.append((float(p), "Pulse 2", i + 1, off))
    if require_full_window:
        pulses = [p for p in pulses if (p[0] - float(pre_s) >= t_all[0]) and (p[0] + float(post_s) <= t_all[-1])]
    else:
        pulses = [p for p in pulses if (p[0] + float(post_s) >= t_all[0]) and (p[0] - float(pre_s) <= t_all[-1])]
    pulses.sort(key=lambda z: z[0])
    if int(max_pulses) > 0 and len(pulses) > int(max_pulses):
        pulses = pulses[:int(max_pulses)]
    if not pulses:
        os.makedirs(save_dir, exist_ok=True)
        out_html = os.path.join(save_dir, f"{base_tag}__pulse_qa_{panel_tag}.html")
        out_pdf = os.path.join(save_dir, f"{base_tag}__pulse_qa_{panel_tag}.pdf")
        msg = (
            "No pulses with a complete pre/post window found for pulse QA export."
            if require_full_window
            else "No pulses found in current time window for pulse QA export."
        )
        with open(out_html, "w", encoding="utf-8") as f:
            f.write("<!doctype html><html><head><meta charset='utf-8'>")
            f.write(f"<title>{html.escape(str(title))}</title></head><body>")
            f.write(f"<h1>{html.escape(str(title))}</h1>")
            f.write(f"<p>{html.escape(msg)}</p>")
            f.write("</body></html>")
        with PdfPages(out_pdf) as pdf:
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            ax.axis("off")
            ax.text(0.5, 0.55, str(title), ha="center", va="center", fontsize=13, transform=ax.transAxes)
            ax.text(0.5, 0.45, msg, ha="center", va="center", fontsize=11, transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        print(f"[HTML] pulse QA {n_rows}-panel placeholder: {out_html}")
        print(f"[PDF] pulse QA {n_rows}-panel placeholder: {out_pdf}")
        return out_html

    def _mk_intervals(pair):
        if not pair:
            return []
        up, down = pair
        up = np.asarray(up, dtype=int).ravel()
        down = np.asarray(down, dtype=int).ravel()
        out = []
        for u, d in zip(up[:min(up.size, down.size)], down[:min(up.size, down.size)]):
            if 0 <= u < t_all.size and 0 < d <= t_all.size and d > u:
                out.append((float(t_all[u]), float(t_all[d - 1])))
        return out

    swr_groups = [
        ("SWR spontaneous", _mk_intervals(swr_spont), "rgba(39, 174, 96, 0.32)"),
        ("SWR triggered", _mk_intervals(swr_trig), "rgba(39, 174, 96, 0.42)"),
        ("SWR associated", _mk_intervals(swr_assoc), "rgba(39, 174, 96, 0.52)"),
    ]
    up_groups = [
        ("UP spontaneous", _mk_intervals(up_spont), "rgba(52, 152, 219, 0.22)"),
        ("UP triggered", _mk_intervals(up_trig), "rgba(52, 152, 219, 0.34)"),
        ("UP associated", _mk_intervals(up_assoc), "rgba(52, 152, 219, 0.46)"),
    ]
    spindle_groups = [
        ("Spindle spontaneous", _mk_intervals(spindle_spont), "rgba(214, 51, 132, 0.28)"),
        ("Spindle triggered", _mk_intervals(spindle_trig), "rgba(214, 51, 132, 0.40)"),
        ("Spindle associated", _mk_intervals(spindle_assoc), "rgba(214, 51, 132, 0.52)"),
    ]

    def _axis_ref(row):
        return "x" if row == 1 else f"x{row}"

    def _y_domain_ref(row):
        return "y domain" if row == 1 else f"y{row} domain"

    def _add_event_spans(shapes, groups, row, p, w0, w1):
        for _, spans, fill in groups:
            for t0, t1 in spans:
                if t1 < w0 or t0 > w1:
                    continue
                shapes.append(dict(
                    type="rect",
                    x0=max(t0, w0) - p,
                    x1=min(t1, w1) - p,
                    y0=0,
                    y1=1,
                    xref=_axis_ref(row),
                    yref=_y_domain_ref(row),
                    line=dict(width=0),
                    fillcolor=fill,
                ))

    def _apply_local_y(fig, row, y):
        yy = np.asarray(y, dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return
        y0 = float(np.nanpercentile(yy, 1))
        y1 = float(np.nanpercentile(yy, 99))
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
            y0 = float(np.nanmin(yy))
            y1 = float(np.nanmax(yy))
        if y1 <= y0:
            pad = max(abs(y0) * 0.1, 1.0)
        else:
            pad = 0.08 * (y1 - y0)
        fig.update_yaxes(range=[y0 - pad, y1 + pad], autorange=False, row=row, col=1)

    pulse_pages = []
    snippets = []
    for i_pulse, (p, label, ordinal, off) in enumerate(pulses):
        w0 = p - float(pre_s)
        w1 = p + float(post_s)
        i0 = int(np.searchsorted(t_all, w0, side="left"))
        i1 = int(np.searchsorted(t_all, w1, side="right"))
        i0 = max(0, min(i0, t_all.size - 1))
        i1 = max(i0 + 2, min(i1, t_all.size))
        tt = t_all[i0:i1] - p
        local = [x[i0:i1] for x in traces]
        max_pts = max(100, int(max_points_per_panel))
        if tt.size > max_pts:
            step = int(np.ceil(tt.size / max_pts))
            tt = tt[::step]
            local = [x[::step] for x in local]
        pulse_pages.append((tt.copy(), [np.asarray(x, float).copy() for x in local], p, label, ordinal, off, w0, w1))

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[1.0 / n_rows] * n_rows,
        )
        names = [ripple_name, sharp_name, up_name, spindle_name]
        y_labels = [ripple_y_label, sharp_y_label, up_y_label, spindle_y_label]
        colors = ["#444444", "#8b0000", "#111111", "#cc00cc"]
        widths = [0.9, 1.1, 1.0, 1.0]
        group_sets = [swr_groups, swr_groups, up_groups, spindle_groups]
        if has_raw_swr:
            names = [raw_swr_name] + names
            y_labels = [raw_swr_y_label] + y_labels
            colors = ["#111111"] + colors
            widths = [0.8] + widths
            group_sets = [swr_groups] + group_sets
        for row, yy in enumerate(local, start=1):
            fig.add_trace(go.Scatter(
                x=tt,
                y=yy,
                mode="lines",
                name=str(names[row - 1]),
                line=dict(color=colors[row - 1], width=widths[row - 1]),
                showlegend=(i_pulse == 0),
            ), row=row, col=1)
            _apply_local_y(fig, row, yy)

        shapes = []
        for row in range(1, n_rows + 1):
            shapes.append(dict(
                type="line",
                x0=0,
                x1=0,
                y0=0,
                y1=1,
                xref=_axis_ref(row),
                yref=_y_domain_ref(row),
                line=dict(width=2, dash="dot", color="red"),
            ))
            if show_pulse_durations and off is not None and off > p:
                off_rel = min(float(off), w1) - p
                shapes.append(dict(
                    type="rect",
                    x0=0,
                    x1=off_rel,
                    y0=0,
                    y1=1,
                    xref=_axis_ref(row),
                    yref=_y_domain_ref(row),
                    line=dict(width=0),
                    fillcolor="rgba(255, 0, 0, 0.08)",
                ))
                shapes.append(dict(
                    type="line",
                    x0=off_rel,
                    x1=off_rel,
                    y0=0,
                    y1=1,
                    xref=_axis_ref(row),
                    yref=_y_domain_ref(row),
                    line=dict(width=2, dash="dash", color="red"),
                ))
        for row, groups in enumerate(group_sets, start=1):
            _add_event_spans(shapes, groups, row, p, w0, w1)

        if i_pulse == 0:
            for legend_label, _, fill in swr_groups + up_groups + spindle_groups:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(width=12, color=fill),
                    name=legend_label,
                ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=2, dash="dot", color="red"),
                name="Pulse onset",
            ))
            if show_pulse_durations:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(width=2, dash="dash", color="red"),
                    name="Pulse offset",
                ))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(width=12, color="rgba(255, 0, 0, 0.08)"),
                    name="Pulse duration",
                ))

        fig.update_layout(
            title=(
                f"{label} #{ordinal} at {p:.3f}s"
                + (f" | offset +{(float(off) - p):.3f}s" if show_pulse_durations and off is not None and off > p else "")
            ),
            shapes=shapes,
            margin=dict(l=68, r=20, t=80, b=42),
            template="plotly_white",
            height=760 if has_raw_swr else 640,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        for row, y_label in enumerate(y_labels, start=1):
            fig.update_yaxes(title_text=y_label, row=row, col=1)
        fig.update_xaxes(title_text="Time from pulse onset (s)", row=n_rows, col=1)
        for row in range(1, n_rows + 1):
            fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror="allticks", row=row, col=1)
            fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror="allticks", row=row, col=1)

        snippets.append(fig.to_html(
            full_html=False,
            include_plotlyjs=("cdn" if i_pulse == 0 else False),
            config={"responsive": True, "displaylogo": False},
        ))

    out_html = os.path.join(save_dir, f"{base_tag}__pulse_qa_{panel_tag}.html")
    css = """
    body { font-family: Arial, sans-serif; margin: 20px; color: #222; }
    h1 { font-size: 22px; margin: 0 0 6px; }
    .meta { color: #555; margin-bottom: 18px; }
    .pulse-block { border-top: 1px solid #ddd; padding-top: 14px; margin-top: 18px; }
    """
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write(f"<title>{html.escape(str(title))}</title><style>{css}</style></head><body>")
        f.write(f"<h1>{html.escape(str(title))}</h1>")
        f.write(
            "<div class='meta'>"
            f"n_pulses={len(pulses)} | window=-{float(pre_s):.3f}s..+{float(post_s):.3f}s | "
            + (
                "Panel 1: raw SWR, Panel 2: ripple, Panel 3: sharp-wave, Panel 4: UP, Panel 5: spindle"
                if has_raw_swr
                else "Panel 1: ripple, Panel 2: sharp-wave, Panel 3: UP, Panel 4: spindle"
            )
            + "</div>"
        )
        for snip in snippets:
            f.write("<div class='pulse-block'>")
            f.write(snip)
            f.write("</div>")
        f.write("</body></html>")
    print(f"[HTML] pulse QA {n_rows}-panel: {out_html}")
    out_pdf = os.path.join(save_dir, f"{base_tag}__pulse_qa_{panel_tag}.pdf")

    def _rgba_to_mpl(c):
        if isinstance(c, str) and c.startswith("rgba(") and c.endswith(")"):
            vals = [v.strip() for v in c[5:-1].split(",")]
            if len(vals) == 4:
                return (float(vals[0]) / 255.0, float(vals[1]) / 255.0, float(vals[2]) / 255.0, float(vals[3]))
        return c

    def _shade_pdf_groups(ax, groups, p, w0, w1):
        for _, spans, fill in groups:
            color = _rgba_to_mpl(fill)
            for t0, t1 in spans:
                if t1 < w0 or t0 > w1:
                    continue
                ax.axvspan(max(t0, w0) - p, min(t1, w1) - p, color=color, linewidth=0)

    with PdfPages(out_pdf) as pdf:
        for tt, local, p, label, ordinal, off, w0, w1 in pulse_pages:
            fig_pdf, axes = plt.subplots(n_rows, 1, figsize=(11.0, 9.4 if has_raw_swr else 8.5), sharex=True)
            fig_pdf.suptitle(
                f"{label} #{ordinal} at {p:.3f}s"
                + (f" | offset +{(float(off) - p):.3f}s" if show_pulse_durations and off is not None and off > p else ""),
                fontsize=11,
            )
            y_labels = [ripple_y_label, sharp_y_label, up_y_label, spindle_y_label]
            colors = ["#444444", "#8b0000", "#111111", "#cc00cc"]
            group_sets = [swr_groups, swr_groups, up_groups, spindle_groups]
            if has_raw_swr:
                y_labels = [raw_swr_y_label] + y_labels
                colors = ["#111111"] + colors
                group_sets = [swr_groups] + group_sets
            for ax, yy, ylab, line_color, groups in zip(axes, local, y_labels, colors, group_sets):
                _shade_pdf_groups(ax, groups, p, w0, w1)
                ax.axvline(0.0, color="red", linestyle=":", linewidth=1.2)
                if show_pulse_durations and off is not None and off > p:
                    off_rel = min(float(off), w1) - p
                    ax.axvspan(0.0, off_rel, color=(1.0, 0.0, 0.0, 0.08), linewidth=0)
                    ax.axvline(off_rel, color="red", linestyle="--", linewidth=1.2)
                ax.plot(tt, yy, color=line_color, linewidth=0.8)
                ax.set_ylabel(ylab, fontsize=8)
                ax.grid(True, alpha=0.18, linewidth=0.5)
                finite = np.asarray(yy, float)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    y0 = float(np.nanpercentile(finite, 1))
                    y1 = float(np.nanpercentile(finite, 99))
                    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
                        y0 = float(np.nanmin(finite))
                        y1 = float(np.nanmax(finite))
                    pad = max(abs(y0) * 0.1, 1.0) if y1 <= y0 else 0.08 * (y1 - y0)
                    ax.set_ylim(y0 - pad, y1 + pad)
            axes[-1].set_xlabel("Time from pulse onset (s)")
            axes[-1].set_xlim(-float(pre_s), float(post_s))
            fig_pdf.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig_pdf)
            plt.close(fig_pdf)
    print(f"[PDF] pulse QA {n_rows}-panel: {out_pdf}")
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
        margin=dict(l=60, r=60, t=85, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    out_html = os.path.join(save_dir, f"{base_tag}__spectrogram_interactive.html")
    _write_plotly_html(fig, out_html, title)
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
    pulse_times_1_off=None,
    pulse_times_2_off=None,
    max_points=60_000,
    max_spans_per_channel=600,
    max_pulses_per_channel=800,
    title="SWR scan (raw + ripple bandpass)",
    y_label="Amplitude",
    out_suffix="SWR_CH00_17_SCAN",
    bp_band_label=None,
    slow_band_label=None,
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

    n_rows = 2 * n_ch

    # Adaptive caps for large stacked plots so HTML export remains responsive.
    eff_max_points = int(max(1200, min(int(max_points), int(np.ceil(90_000 / max(n_ch, 1))))))
    eff_max_spans_per_channel = int(
        max(6, min(int(max_spans_per_channel), int(np.ceil(260 / max(n_ch, 1)))))
    )
    eff_max_pulses = int(max(24, min(int(max_pulses_per_channel), 80)))
    if n_ch >= 16:
        eff_max_points = int(max(1000, min(eff_max_points, 3500)))
        eff_max_spans_per_channel = int(max(4, min(eff_max_spans_per_channel, 10)))
        eff_max_pulses = int(max(16, min(eff_max_pulses, 40)))
    print(
        f"[HTML-SWR] n_ch={n_ch} points_cap={eff_max_points} "
        f"spans_cap_per_ch={eff_max_spans_per_channel} pulses_cap={eff_max_pulses}"
    )

    if t.size > eff_max_points:
        step = int(np.ceil(t.size / eff_max_points))
        t = t[::step]
        X_raw = X_raw[:, ::step]
        X_bp = X_bp[:, ::step]

    def _robust_ylim(y):
        yy = np.asarray(y, float).ravel()
        yy = yy[np.isfinite(yy)]
        if yy.size < 10:
            return None
        med = float(np.nanmedian(yy))
        dev = np.abs(yy - med)
        span = float(np.nanpercentile(dev, 99.0))
        if (not np.isfinite(span)) or span <= 1e-12:
            return None
        pad = 1.25 * span
        return [med - pad, med + pad]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.001, 0.02 / max(n_rows, 1)),
        row_heights=[1.0 / n_rows] * n_rows,
    )

    for i in range(n_ch):
        ch = int(channel_indices[i])
        show_leg = (i == 0)
        row_bp = 2 * i + 1
        row_raw = 2 * i + 2
        bp_name = f"Fast {bp_band_label}" if bp_band_label else "Fast component"
        slow_name = f"Slow {slow_band_label}" if slow_band_label else "Slow component"
        fig.add_trace(
            go.Scattergl(
                x=t, y=X_bp[i], mode="lines",
                name=bp_name,
                line=dict(color="#1f4fff", width=1.0),
                opacity=0.85,
                legendgroup="bp",
                showlegend=show_leg,
            ),
            row=row_bp, col=1
        )
        fig.add_trace(
            go.Scattergl(
                x=t, y=X_raw[i], mode="lines",
                name=slow_name,
                line=dict(color="#128a2e", width=1.0),
                opacity=0.90,
                legendgroup="raw",
                showlegend=show_leg,
            ),
            row=row_raw, col=1
        )

        spans = swr_intervals_by_channel[i] if i < len(swr_intervals_by_channel) else []
        if len(spans) > eff_max_spans_per_channel:
            step_sp = int(np.ceil(len(spans) / max(1, eff_max_spans_per_channel)))
            spans = spans[::step_sp]
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
                row=row_bp, col=1,
            )
            fig.add_vrect(
                x0=t0, x1=t1,
                fillcolor="rgba(220, 20, 60, 0.20)",
                line_width=0,
                row=row_raw, col=1,
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

        bp_ylim = _robust_ylim(X_bp[i])
        raw_ylim = _robust_ylim(X_raw[i])
        bp_axis = f"pri_{ch} {bp_band_label}" if bp_band_label else f"pri_{ch} BP"
        fig.update_yaxes(title_text=bp_axis, row=row_bp, col=1, range=bp_ylim)
        slow_axis = f"pri_{ch} {slow_band_label}" if slow_band_label else f"pri_{ch} Slow"
        fig.update_yaxes(title_text=slow_axis, row=row_raw, col=1, range=raw_ylim)

    def _pair_on_off(ts_on, ts_off):
        if ts_on is None or ts_off is None:
            return []
        on = np.asarray(ts_on, float).ravel()
        off = np.asarray(ts_off, float).ravel()
        on = on[np.isfinite(on)]
        off = off[np.isfinite(off)]
        m = min(on.size, off.size)
        if m <= 0:
            return []
        on = on[:m]
        off = off[:m]
        out = []
        for a, b in zip(on, off):
            a = float(a)
            b = float(b)
            if b <= a:
                continue
            if len(t) and (b < t[0] or a > t[-1]):
                continue
            out.append((a, b))
        return out

    def _add_pulse_lines(ts, dash):
        if ts is None or len(ts) == 0:
            return
        tt = np.asarray(ts, float).ravel()
        tt = tt[np.isfinite(tt)]
        if tt.size > eff_max_pulses:
            tt = tt[::int(np.ceil(tt.size / max(1, eff_max_pulses)))]
        for p in tt:
            if len(t) and (p < t[0] or p > t[-1]):
                continue
            for ri in range(1, n_rows + 1):
                fig.add_vline(
                    x=float(p),
                    line=dict(color="red", width=2.2, dash=dash),
                    opacity=0.55,
                    row=ri, col=1,
                )

    p1_intervals = _pair_on_off(pulse_times_1, pulse_times_1_off)
    p2_intervals = _pair_on_off(pulse_times_2, pulse_times_2_off)
    if len(p1_intervals) > eff_max_pulses:
        step = int(np.ceil(len(p1_intervals) / max(1, eff_max_pulses)))
        p1_intervals = p1_intervals[::step]
    if len(p2_intervals) > eff_max_pulses:
        step = int(np.ceil(len(p2_intervals) / max(1, eff_max_pulses)))
        p2_intervals = p2_intervals[::step]

    for a, b in p1_intervals:
        for ri in range(1, n_rows + 1):
            fig.add_vrect(
                x0=float(a), x1=float(b),
                fillcolor="rgba(255, 0, 0, 0.22)",
                line_width=0,
                row=ri, col=1,
            )
    for a, b in p2_intervals:
        for ri in range(1, n_rows + 1):
            fig.add_vrect(
                x0=float(a), x1=float(b),
                fillcolor="rgba(255, 0, 0, 0.14)",
                line_width=0,
                row=ri, col=1,
            )

    # Always draw ON lines for visibility, add ON->OFF duration as extra shading when available.
    _add_pulse_lines(pulse_times_1, "dot")
    _add_pulse_lines(pulse_times_2, "dash")

    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dot", color="red"),
            name=("Pulse 1 ON->OFF" if p1_intervals else "Pulse 1 ON"),
        ))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=2, dash="dash", color="red"),
            name=("Pulse 2 ON->OFF" if p2_intervals else "Pulse 2 ON"),
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=70, r=20, t=85, b=50),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="Zeit (s)",
        showline=True, linewidth=1.5, linecolor="black", mirror="allticks",
        row=n_rows, col=1
    )
    fig.update_xaxes(rangeslider=dict(visible=(n_ch <= 5)), row=n_rows, col=1)
    fig.update_yaxes(showline=True, linewidth=1.2, linecolor="black", mirror="allticks")

    out_html = os.path.join(save_dir, f"{base_tag}__{out_suffix}.html")
    plotly_offline_plot(
        fig,
        filename=out_html,
        auto_open=False,
        include_plotlyjs="cdn",
        validate=False,
    )
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


def export_mua_html(
    base_tag, save_dir, time_s, lfp_signal,
    spike_times_s,
    up_spont=None, up_trig=None, up_assoc=None,
    bin_s=1.0,
    title="MUA (interaktiv)",
    y_label="LFP (µV)",
    max_lfp_points=600_000,
    max_spike_markers=100_000,
):
    """Two-panel interactive HTML: LFP + spike markers (top), binned MUA rate (bottom)."""
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(lfp_signal, dtype=float)
    spk = np.asarray(spike_times_s, dtype=float)
    spk = spk[(spk >= t[0]) & (spk <= t[-1])] if t.size > 0 else spk

    # Downsample LFP trace for rendering
    if t.size > max_lfp_points:
        step = int(np.ceil(t.size / max_lfp_points))
        t_plot, x_plot = t[::step], x[::step]
    else:
        t_plot, x_plot = t, x

    # Binned MUA rate
    if t.size > 0 and spk.size > 0:
        bins = np.arange(float(t[0]), float(t[-1]) + bin_s, bin_s)
        counts, edges = np.histogram(spk, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        rates = counts.astype(float) / bin_s
    else:
        bin_centers = np.array([])
        rates = np.array([])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("LFP + Spike-Ereignisse", f"MUA Rate ({bin_s}s Bins)"),
    )

    # --- LFP trace ---
    fig.add_trace(go.Scatter(
        x=t_plot, y=x_plot, mode="lines", name="LFP",
        line=dict(color="#1f77b4", width=0.7),
    ), row=1, col=1)

    # --- Spike markers (WebGL scatter, fixed y at bottom of LFP panel) ---
    if spk.size > 0:
        rng = float(np.nanmax(x_plot) - np.nanmin(x_plot)) or 1.0
        y_spk = float(np.nanmin(x_plot)) - 0.08 * rng
        if spk.size > max_spike_markers:
            rng_seed = np.random.default_rng(0)
            spk_plot = np.sort(rng_seed.choice(spk, max_spike_markers, replace=False))
        else:
            spk_plot = spk
        fig.add_trace(go.Scattergl(
            x=spk_plot,
            y=np.full(spk_plot.size, y_spk),
            mode="markers",
            marker=dict(symbol="line-ns", size=10, color="rgba(220,40,40,0.45)",
                        line=dict(width=1, color="rgba(220,40,40,0.45)")),
            name=f"Spikes (n={spk.size:,})",
        ), row=1, col=1)

    # --- Binned rate bars ---
    if bin_centers.size > 0:
        fig.add_trace(go.Bar(
            x=bin_centers, y=rates, name=f"Rate ({bin_s}s)",
            marker_color="#2ca02c", opacity=0.75,
        ), row=2, col=1)

    # --- UP-state shading (yref=paper spans both panels) ---
    def _mk_intervals(UP, DOWN):
        if UP is None or DOWN is None:
            return []
        UP, DOWN = np.asarray(UP, int), np.asarray(DOWN, int)
        m = min(len(UP), len(DOWN))
        if m == 0:
            return []
        UP, DOWN = UP[:m], DOWN[:m]
        n = len(time_s)
        out = []
        for u, d in zip(UP[np.argsort(UP)], DOWN[np.argsort(UP)]):
            if 0 <= u < n and 0 < d <= n and d > u:
                out.append((float(time_s[u]), float(time_s[d - 1])))
        return out

    shapes = []
    for ivs, color in [
        (_mk_intervals(*(up_spont  or (None, None))), "rgba(46,204,113,0.18)"),
        (_mk_intervals(*(up_trig   or (None, None))), "rgba(31,119,180,0.18)"),
        (_mk_intervals(*(up_assoc  or (None, None))), "rgba(255,127,14,0.18)"),
    ]:
        for t0s, t1s in ivs:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=t0s, x1=t1s, y0=0, y1=1,
                fillcolor=color, line_width=0, layer="below",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        shapes=shapes,
        height=680,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        bargap=0.05,
    )
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text="Rate (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Zeit (s)", row=2, col=1)

    out_path = os.path.join(save_dir, f"{base_tag}__mua_interactive.html")
    _write_plotly_html(fig, out_path, title)
    print(f"[MUA-HTML] {out_path}  (spikes={spk.size:,})")


def export_mua_ap_raw_html(
    base_tag, save_dir, raw_signal, spike_times_s,
    fs_raw=32000.0, hp_hz=300.0, threshold_sigma=3.5,
    t0=0.0,
    title="MUA – HP-Signal + AP-Detektion (Rohsignal)",
    max_display_hz=8000.0,
    spont_counts=None,
    channel=None,
):
    """
    Interaktives HTML des HP-gefilterten Rohsignals mit Schwellenlinie und
    markierten Threshold-Crossings (Aktionspotenziale).

    Zeigt drei übereinanderliegende Panels:
      1. HP-gefiltertes Signal + Schwelle + AP-Marker (groß, für Überblick)
      2. Gleiche Daten – vorgezoomt auf die ersten ~2 Sekunden (für Detailcheck)
      3. Histogramm APs/Up-Zustand (spontan) – falls spont_counts übergeben;
         sonst Histogramm der AP-Amplituden

    Parameters
    ----------
    raw_signal      : 1D-Array, Rohsignal bei fs_raw Hz
    spike_times_s   : absolute Spike-Zeiten in Sekunden
    spont_counts    : 1D-Array mit AP-Anzahl pro spontanem Up-Zustand (optional)
    t0              : Zeitstempel des ersten Samples
    max_display_hz  : Abtastrate für die HTML-Darstellung (Downsample für Rendering)
    """
    from scipy.signal import butter, sosfiltfilt

    sig = np.asarray(raw_signal, dtype=float)
    n   = sig.size
    if n == 0:
        print("[MUA-AP-HTML] leeres Signal – übersprungen")
        return

    # ── 1. HP-filtern (identisch mit compute_mua_rate) ──────────────────────
    sos  = butter(3, hp_hz / (fs_raw / 2.0), btype="high", output="sos")
    filt = sosfiltfilt(sos, sig)

    noise = np.median(np.abs(filt)) / 0.6745
    thr   = -threshold_sigma * noise          # negative Schwelle

    # ── 2. Relative Zeit (0 = Aufnahme-Start) ───────────────────────────────
    dur_s = n / fs_raw
    t_rel = np.arange(n, dtype=float) / fs_raw   # 0 … dur_s

    # spike_times_s kommen immer 0-basiert (t0=0.0 wird aus Main übergeben)
    spk = np.sort(np.asarray(spike_times_s, dtype=float))

    # ── 3. Min-Max-Envelope-Downsampling für Panel 1 ────────────────────────
    # Einfaches Stride-Downsampling (filt[::4]) lässt 1ms-Spikes verschwinden:
    # ein Spike hat bei 32kHz ~32 Proben, liegt aber oft zwischen zwei
    # angezeigten 8kHz-Proben → im Display unsichtbar, Tick-Marker aber korrekt.
    # Mit Min-Max-Envelope wird pro 4-Proben-Fenster sowohl Minimum als auch
    # Maximum angezeigt → echte Schwellenkreuzungen sind immer sichtbar.
    ds = max(1, int(np.ceil(fs_raw / max_display_hz)))
    if ds > 1:
        n_win  = n // ds
        _trim  = filt[:n_win * ds].reshape(n_win, ds)
        _t_win = t_rel[:n_win * ds:ds]         # Fenster-Startzeit
        _emin  = _trim.min(axis=1)
        _emax  = _trim.max(axis=1)
        # Interleave: [max₀, min₀, max₁, min₁, …]  →  zeigt vollen Wertebereich
        t_ds    = np.empty(n_win * 2)
        t_ds[0::2] = _t_win
        t_ds[1::2] = _t_win + (ds - 1) / fs_raw   # Fenster-Endzeit
        filt_ds = np.empty(n_win * 2)
        filt_ds[0::2] = _emax
        filt_ds[1::2] = _emin
    else:
        t_ds    = t_rel
        filt_ds = filt

    # ── 4. AP-Marker: Amplitude am Crossing-Sample ──────────────────────────
    spk_valid = spk[(spk >= 0) & (spk <= dur_s)]
    spk_idx   = np.clip((spk_valid * fs_raw).astype(int), 0, n - 1)
    spk_amps  = filt[spk_idx]

    # ── 5. Statistik: ∅ APs pro spontanem Up-Zustand ────────────────────────
    _sc = np.asarray(spont_counts, dtype=float) if spont_counts is not None else np.array([])
    _sc_int = _sc[np.isfinite(_sc)].astype(int)
    if _sc_int.size > 0:
        _mean_c = float(np.mean(_sc_int))
        _std_c  = float(np.std(_sc_int))
        _med_c  = float(np.median(_sc_int))
        _stats_line = (
            f"∅ <b>{_mean_c:.1f}</b> APs/UP  ±{_std_c:.1f} SD  |  "
            f"Median {_med_c:.0f}  |  n={_sc_int.size} Up-Zustände"
        )
    else:
        _stats_line = f"Gesamt-Spikes: {spk_valid.size:,}  |  keine Up-Zustand-Daten"

    # ── 6. Plotly-Figure (1 Panel: Signalüberblick) ──────────────────────────
    fig = go.Figure()

    # HP-Signal (Min-Max-Envelope, downgesampelt)
    fig.add_trace(go.Scattergl(
        x=t_ds, y=filt_ds, mode="lines",
        line=dict(color="#1f77b4", width=0.6),
        name="HP-Signal",
    ))
    # Schwellenlinie
    fig.add_trace(go.Scatter(
        x=[float(t_ds[0]), float(t_ds[-1])],
        y=[thr, thr],
        mode="lines",
        line=dict(color="rgba(214,39,40,0.8)", width=1.5, dash="dash"),
        name=f"Schwelle (−{threshold_sigma}×MAD = {thr:.2f})",
    ))
    # Spike-Tick-Marker knapp unter der Schwelle
    if spk_valid.size > 0:
        _max_mk = 80_000
        _st = spk_valid[np.linspace(0, spk_valid.size-1, _max_mk).astype(int)] \
              if spk_valid.size > _max_mk else spk_valid
        _y_tick = thr - 0.15 * abs(thr)
        fig.add_trace(go.Scattergl(
            x=_st, y=np.full(_st.size, _y_tick), mode="markers",
            marker=dict(color="rgba(214,39,40,0.75)", size=9,
                        symbol="line-ns", line=dict(width=1.5)),
            name=f"Erkannte APs (n={spk_valid.size:,})",
            hovertemplate="AP bei t=%{x:.4f} s<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(title="Zeit ab Aufnahme-Start (s)"),
        yaxis=dict(title=(f"ch{channel} — HP-Signal (AU)" if channel is not None else "HP-Signal (AU)")),
        # Statistik als sichtbare Beschriftung unter dem Plot
        annotations=[dict(
            text=_stats_line,
            xref="paper", yref="paper",
            x=0.5, y=-0.10,
            showarrow=False,
            font=dict(size=14),
            xanchor="center",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(150,150,150,0.5)",
            borderwidth=1,
        )],
    )

    out_path = os.path.join(save_dir, f"{base_tag}__mua_ap_raw.html")
    _write_plotly_html(fig, out_path, title)
    print(
        f"[MUA-AP-HTML] {out_path}  "
        f"(n_spikes={spk_valid.size:,}, thr={thr:.3f}, noise={noise:.3f})"
    )


