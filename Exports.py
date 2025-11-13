import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot as plotly_offline_plot
import os



def export_interactive_lfp_html(
    base_tag, save_dir, time_s, y,
    pulse_times_1=None, pulse_times_2=None,
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
    if pulse_times_1 is not None and len(pulse_times_1):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(width=1, dash="dot", color="red"),
                                 name="Pulse 1"))
    if pulse_times_2 is not None and len(pulse_times_2):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(width=1, dash="dash", color="red"),
                                 name="Pulse 2"))

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
