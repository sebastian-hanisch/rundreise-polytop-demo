"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Die Knotenlagen sind fest je Instanz: dieselbe Instanz sieht in allen Ansichten gleich aus."""

from itertools import permutations
from math import cos, pi, sin, sqrt

import plotly.graph_objects as go

import rpt_constants as C
import rpt_evaluation as E
import rpt_instanzen as I
import rpt_lp as LP


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legende=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legende,
                      legend=dict(orientation="h", y=-0.15))
    return lock_axes(fig)


def _huelle_um(punkte, radius: float) -> list:
    """Konvexe Hülle um Kreise mit `radius` um die Punkte (Umrandung einer Knotenmenge)."""
    rand = [(x + radius * cos(2 * pi * k / 16), y + radius * sin(2 * pi * k / 16)) for x, y in punkte for k in range(16)]
    rand.sort()

    def kreuz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    unten, oben = [], []
    for p in rand:
        while len(unten) >= 2 and kreuz(unten[-2], unten[-1], p) <= 0:
            unten.pop()
        unten.append(p)
    for p in reversed(rand):
        while len(oben) >= 2 and kreuz(oben[-2], oben[-1], p) <= 0:
            oben.pop()
        oben.append(p)
    huelle = unten[:-1] + oben[:-1]
    return huelle + huelle[:1]


def _linien(inst, nummern, kanten):
    xs, ys = [], []
    for j in nummern:
        u, v = kanten[j]
        xs += [inst.punkte[u][0], inst.punkte[v][0], None]
        ys += [inst.punkte[u][1], inst.punkte[v][1], None]
    return xs, ys


def _umriss(fig, inst, menge, farbe, name, radius=28, fuellung=True):
    huelle = _huelle_um([inst.punkte[v] for v in sorted(menge)], radius)
    fig.add_trace(go.Scatter(x=[p[0] for p in huelle], y=[p[1] for p in huelle], mode="lines", name=name,
                             line=dict(color=farbe, width=2, dash="dash"), fill="toself" if fuellung else None,
                             fillcolor=_blass(farbe), hoverinfo="skip"))


def _blass(farbe: str, alpha: float = 0.07) -> str:
    farbe = farbe.lstrip("#")
    r, g, b = (int(farbe[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_karte(inst: I.Instanz, x, menge=None, griff=None, zahnkanten=(), zahnmengen=(), height: int = 480):
    """Lösung x auf der Karte: Wert 1 durchgezogen, gebrochen gestrichelt; markierte Subtourmenge, Griff, Zahnkanten und Zähne."""
    kanten = inst.kanten
    fig = go.Figure()
    if inst.netz:
        xs, ys = [], []
        for u, v, _ in inst.netz:
            xs += [inst.punkte[u][0], inst.punkte[v][0], None]
            ys += [inst.punkte[u][1], inst.punkte[v][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_GRUND, width=1), name="Prisma-Kanten", hoverinfo="skip"))
    if griff is not None:
        _umriss(fig, inst, griff, C.COLOR_GRIFF, "Griff H", radius=34)
    if menge is not None:
        _umriss(fig, inst, menge, C.COLOR_MENGE, "Menge S", radius=34)
    for i, t in enumerate(zahnmengen):
        _umriss(fig, inst, t, C.COLOR_ZAEHNE[i % len(C.COLOR_ZAEHNE)], f"Zahn {i + 1}", radius=22, fuellung=False)
    if zahnkanten:
        xs, ys = _linien(inst, zahnkanten, kanten)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_ZAHN, width=11), opacity=0.35, name="Zahnkanten",
                                 hoverinfo="skip"))
    eins = [j for j, w in enumerate(x) if w > 1 - LP.TOL_GANZ]
    gebrochen = [j for j, w in enumerate(x) if LP.TOL_GANZ <= w <= 1 - LP.TOL_GANZ]
    if eins:
        xs, ys = _linien(inst, eins, kanten)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EINS, width=3), name="x = 1", hoverinfo="skip"))
    if gebrochen:
        xs, ys = _linien(inst, gebrochen, kanten)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_BRUCH, width=2.5, dash="dash"), name="gebrochen",
                                 hoverinfo="skip"))
    mitten = eins + gebrochen
    if mitten:
        fig.add_trace(go.Scatter(
            x=[(inst.punkte[kanten[j][0]][0] + inst.punkte[kanten[j][1]][0]) / 2 for j in mitten],
            y=[(inst.punkte[kanten[j][0]][1] + inst.punkte[kanten[j][1]][1]) / 2 for j in mitten], mode="markers",
            marker=dict(size=9, opacity=0.0), showlegend=False,
            text=[f"{I.kante_name(inst, j, kanten)}: x = {E.bruch(x[j])}, Kosten {inst.kosten[j]}" for j in mitten],
            hovertemplate="%{text}<extra></extra>"))
    if inst.n <= C.BESCHRIFTEN_BIS:
        for j in gebrochen:
            u, v = kanten[j]
            # Beschriftung etwas neben der Mitte, damit Beschriftungen kreuzender Kanten sich seltener decken
            t = 0.42
            fig.add_annotation(x=inst.punkte[u][0] * (1 - t) + inst.punkte[v][0] * t, y=inst.punkte[u][1] * (1 - t) + inst.punkte[v][1] * t,
                               text=E.bruch(x[j]), showarrow=False, bgcolor="rgba(255,255,255,0.9)", font=dict(size=12, color="#8b1e2d"))
    namen = inst.n <= C.NAMEN_BIS
    markiert = set(menge or ()) | set(griff or ())
    farben = [C.COLOR_GRIFF if griff is not None and v in griff else (C.COLOR_MENGE if menge is not None and v in menge else C.COLOR_KNOTEN)
              for v in range(inst.n)]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in inst.punkte], y=[p[1] for p in inst.punkte], mode="markers+text" if namen else "markers",
        marker=dict(size=22 if namen else 9, color=farben, line=dict(color=C.COLOR_RAND, width=1.5)),
        text=list(inst.namen), textfont=dict(size=11, color=["#ffffff" if v in markiert else "#1b2631" for v in range(inst.n)]),
        textposition="middle center", hovertext=[f"Ort {nm}" for nm in inst.namen], hoverinfo="text", showlegend=False))
    xs = [p[0] for p in inst.punkte]
    ys = [p[1] for p in inst.punkte]
    rand = 60
    fig.add_trace(go.Scatter(x=[min(xs) - rand, max(xs) + rand], y=[min(ys) - rand, max(ys) + rand], mode="markers",
                             marker=dict(size=1, opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _base(fig, height)


def build_stufen(schritte: list, optimum, aktuell: int):
    """LP-Wert über die Schritte, gefärbt nach der Stufe; das Optimum ist waagrecht gepunktet eingezeichnet."""
    fig = go.Figure()
    xs = list(range(len(schritte)))
    fig.add_trace(go.Scatter(x=xs, y=[s.wert for s in schritte], mode="lines", line=dict(color="#aab7b8", width=1.5),
                             showlegend=False, hoverinfo="skip"))
    for stufe, farbe in C.STUFEN_FARBE.items():
        idx = [i for i, s in enumerate(schritte) if s.stufe == stufe]
        if idx:
            fig.add_trace(go.Scatter(x=idx, y=[schritte[i].wert for i in idx], mode="markers", marker=dict(size=9, color=farbe),
                                     name=C.STUFEN_KURZ[stufe], hovertemplate="Schritt %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[aktuell], y=[schritte[aktuell].wert], mode="markers",
                             marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#000000", width=2)), showlegend=False,
                             hoverinfo="skip"))
    if optimum is not None:
        fig.add_hline(y=optimum, line=dict(color="#117a65", width=2, dash="dot"), annotation_text=f"Optimum {E.zahl(optimum)}",
                      annotation_position="bottom right")
    fig.update_xaxes(title_text="Schritt", dtick=max(1, len(schritte) // 10))
    fig.update_yaxes(title_text="LP-Wert")
    return _base(fig, 280)


def build_stufenbalken(zeile: dict):
    """Anteil ganzzahlig und optimal je Stufe einer Messreihe."""
    stufen = ("grad", "subtour", "komp_alle", "exakt")
    namen = ["Grad-LP", "+ Subtour", "+ 2-M. (Komponentenregel)", "+ 2-M. (exakt)"]
    werte = [100.0 * zeile["geloest"][s] / zeile["instanzen"] for s in stufen]
    fig = go.Figure(go.Bar(x=namen, y=werte, marker_color=["#7f8c8d", "#e08e0b", "#85c1e9", "#2e86c1"],
                           text=[f"{zeile['geloest'][s]} von {zeile['instanzen']}" for s in stufen], textposition="outside",
                           hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="ganzzahlig und optimal (%)", range=[0, 112])
    return _base(fig, 320, legende=False)


def build_groesse(anteile: list):
    """Anteil ganzzahlig und optimal über die Größe, je Stufe eine Kurve."""
    fig = go.Figure()
    for s, name, farbe in (("grad", "Grad-LP", "#7f8c8d"), ("subtour", "+ Subtour", "#e08e0b"), ("exakt", "+ 2-Matching (exakt)", "#2e86c1")):
        fig.add_trace(go.Scatter(x=[str(n) for n, _, _ in anteile], y=[a[s] for _, a, _ in anteile], mode="lines+markers", name=name,
                                 line=dict(color=farbe, width=2.5), hovertemplate="%{x} Orte: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Orte n", type="category")
    fig.update_yaxes(title_text="ganzzahlig und optimal (%)", range=[-3, 103])
    return _base(fig, 320)


def build_hilfsgraph(inst: I.Instanz, kanten, teil, x, seite=None, height: int = 460):
    """Der geteilte Hilfsgraph: Mittelpunkte w als Quadrate; die Kapazität x liegt an der Hälfte beim kleineren Ort, 1 − x an der anderen.
    `seite`: Knotenmenge eines Schnitts (Knoten des Hilfsgraphen)."""
    lage = {v: inst.punkte[v] for v in range(inst.n)}
    for j, w in teil:
        u, v = kanten[j]
        lage[w] = ((lage[u][0] + lage[v][0]) / 2, (lage[u][1] + lage[v][1]) / 2)
    fig = go.Figure()
    halb = []
    for j, w in teil:
        u, v = kanten[j]
        halb += [(u, w, float(x[j])), (w, v, 1.0 - float(x[j]))]
    for gekreuzt, farbe, breite, name in ((False, "rgba(120,120,120,0.6)", 1.5, "Hälften"), (True, C.COLOR_BRUCH, 4, "vom Schnitt getrennt")):
        xs, ys = [], []
        for a, b, _ in halb:
            kreuzt = seite is not None and (a in seite) != (b in seite)
            if kreuzt == gekreuzt:
                xs += [lage[a][0], lage[b][0], None]
                ys += [lage[a][1], lage[b][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=farbe, width=breite), name=name, hoverinfo="skip"))
    for a, b, cap in halb:
        if cap > 1e-9:
            fig.add_annotation(x=(lage[a][0] + lage[b][0]) / 2, y=(lage[a][1] + lage[b][1]) / 2, text=E.bruch(cap), showarrow=False,
                               bgcolor="rgba(255,255,255,0.85)", font=dict(size=10, color="#424949"))
    orte = list(range(inst.n))
    innen = seite or set()
    fig.add_trace(go.Scatter(x=[lage[v][0] for v in orte], y=[lage[v][1] for v in orte], mode="markers+text",
                             marker=dict(size=22, color=[C.COLOR_GRIFF if v in innen else C.COLOR_KNOTEN for v in orte],
                                         line=dict(color=C.COLOR_RAND, width=1.5)),
                             text=[inst.namen[v] for v in orte], textfont=dict(size=11, color=["#ffffff" if v in innen else "#1b2631" for v in orte]),
                             showlegend=False, hoverinfo="skip"))
    ws = [w for _, w in teil]
    if ws:
        fig.add_trace(go.Scatter(x=[lage[w][0] for w in ws], y=[lage[w][1] for w in ws], mode="markers", name="Mittelknoten w",
                                 marker=dict(size=8, symbol="square", color=[C.COLOR_GRIFF if w in innen else "#d5d8dc" for w in ws],
                                             line=dict(color=C.COLOR_RAND, width=1)), hoverinfo="skip"))
    xs = [p[0] for p in inst.punkte]
    ys = [p[1] for p in inst.punkte]
    fig.add_trace(go.Scatter(x=[min(xs) - 60, max(xs) + 60], y=[min(ys) - 60, max(ys) + 60], mode="markers", marker=dict(size=1, opacity=0),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _base(fig, height)


def netz_schichten(komparatoren) -> list:
    """Spalte je Komparator für die Zeichnung: die erste Spalte, in der er keinen früheren Draht kreuzt."""
    belegt = {}
    spalten = []
    for i, j in komparatoren:
        s = max((belegt.get(d, -1) for d in range(min(i, j), max(i, j) + 1)), default=-1) + 1
        spalten.append(s)
        for d in range(min(i, j), max(i, j) + 1):
            belegt[d] = s
    return spalten


def build_netz(n: int, komparatoren, entfernt=None):
    """Netzbild: waagrechte Drähte, Komparatoren als senkrechte Stege; der gestrichene gestrichelt."""
    spalten = netz_schichten(komparatoren)
    breite = (max(spalten) + 1) if spalten else 1
    fig = go.Figure()
    for d in range(n):
        fig.add_trace(go.Scatter(x=[-0.8, breite - 0.2], y=[n - d, n - d], mode="lines", line=dict(color="#aab7b8", width=1.5),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=-1.2, y=n - d, text=f"z{d + 1}", showarrow=False, font=dict(size=11))
        fig.add_annotation(x=breite + 0.2, y=n - d, text=str(d + 1), showarrow=False, font=dict(size=11))
    for k, ((i, j), s) in enumerate(zip(komparatoren, spalten)):
        weg = k == entfernt
        fig.add_trace(go.Scatter(x=[s, s], y=[n - i, n - j], mode="lines+markers", showlegend=False,
                                 line=dict(color=C.COLOR_BRUCH if weg else C.COLOR_EINS, width=2.5, dash="dot" if weg else "solid"),
                                 marker=dict(size=7 if n <= 10 else 5, color=C.COLOR_BRUCH if weg else C.COLOR_EINS),
                                 hovertext=f"Komparator {k + 1}: Drähte {i + 1} und {j + 1}" + (" (gestrichen)" if weg else ""),
                                 hoverinfo="text"))
    fig.update_xaxes(visible=False, range=[-1.7, breite + 0.7])
    fig.update_yaxes(visible=False, range=[0.4, n + 0.6])
    return _base(fig, max(180, min(460, 40 + 28 * n)), legende=False)


def _basis3():
    """Drei orthonormale Richtungen senkrecht zu (1, 1, 1, 1)."""
    return [[v / sqrt(2) for v in (1, -1, 0, 0)], [v / sqrt(6) for v in (1, 1, -2, 0)], [v / sqrt(12) for v in (1, 1, 1, -3)]]


def permutaeder_lage():
    """{Permutation: (x, y, z)} der 24 Ecken des Permutaeders für n = 4 und seine 36 Kanten (zwei Einträge mit Werten k, k+1 getauscht)."""
    basis = _basis3()
    lage = {p: tuple(sum(b[i] * p[i] for i in range(4)) for b in basis) for p in permutations((1, 2, 3, 4))}
    kanten = set()
    for p in lage:
        for k in (1, 2, 3):
            q = tuple(k + 1 if v == k else (k if v == k + 1 else v) for v in p)
            kanten.add(tuple(sorted((p, q))))
    return lage, sorted(kanten)


def build_permutaeder(drin):
    """3D-Permutaeder: Ecken in der Projektion blau, verlorene rot."""
    lage, kanten = permutaeder_lage()
    drin = {tuple(p) for p in drin}
    xs, ys, zs = [], [], []
    for p, q in kanten:
        for achse, liste in enumerate((xs, ys, zs)):
            liste += [lage[p][achse], lage[q][achse], None]
    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=3), hoverinfo="skip",
                                 showlegend=False))
    for auswahl, farbe, name, symbol in ((sorted(p for p in lage if p in drin), C.COLOR_GRIFF, "in der Projektion", "circle"),
                                         (sorted(p for p in lage if p not in drin), C.COLOR_BRUCH, "verloren", "x")):
        if auswahl:
            fig.add_trace(go.Scatter3d(x=[lage[p][0] for p in auswahl], y=[lage[p][1] for p in auswahl], z=[lage[p][2] for p in auswahl],
                                       mode="markers", name=name, marker=dict(size=5, color=farbe, symbol=symbol),
                                       text=["".join(str(v) for v in p) for p in auswahl], hovertemplate="%{text}<extra></extra>"))
    leer = dict(visible=False)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=0.02),
                      scene=dict(xaxis=leer, yaxis=leer, zaxis=leer, aspectmode="data", camera=dict(eye=dict(x=1.4, y=1.2, z=0.8))))
    return fig
