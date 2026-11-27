"""Trennung der Subtourzeilen x(δ(S)) >= 2.

Zuerst die Zusammenhangskomponenten des Trägergraphen (Kanten mit x > 0): ist er nicht zusammenhängend, ist jede Komponente eine verletzte Menge.
Sonst alle Schnitte des Gomory-Hu-Baums unter 2 (rpt_trennung.baumschnitte mit Schwelle 2); die kleinste Baumkante ist der globale Mindestschnitt,
also ist nichts mehr verletzt, wenn kein Baumschnitt unter 2 liegt. S und das Komplement geben dieselbe Zeile, gespeichert wird die kleinere Seite."""

import rpt_lp as LP
import rpt_trennung as T

TOL = 1e-6
TRAEGER = 1e-7


def komponenten(n: int, kanten, x, unten: float = TRAEGER, oben: float = 2.0) -> list:
    """Zusammenhangskomponenten (frozensets, nach kleinstem Knoten sortiert) des Graphen der Kanten mit unten < x < oben."""
    kopf = list(range(n))

    def wurzel(a):
        while kopf[a] != a:
            kopf[a] = kopf[kopf[a]]
            a = kopf[a]
        return a

    for (u, v), w in zip(kanten, x):
        if unten < w < oben:
            kopf[wurzel(u)] = wurzel(v)
    teile = {}
    for v in range(n):
        teile.setdefault(wurzel(v), set()).add(v)
    return sorted((frozenset(t) for t in teile.values()), key=min)


def kleinere_seite(n: int, menge) -> frozenset:
    """Die kleinere der beiden Seiten; bei gleicher Größe die ohne Knoten 0 (feste Regel, damit jede Zeile nur eine Schreibweise hat)."""
    menge = frozenset(menge)
    andere = frozenset(range(n)) - menge
    if len(menge) > len(andere) or (len(menge) == len(andere) and 0 in menge):
        return andere
    return menge


def wert(kanten, x, menge) -> float:
    """x(δ(S))."""
    return float(sum(x[j] for j in LP.schnittkanten(kanten, menge)))


def verletzte_subtours(n: int, kanten, x, tol: float = TOL) -> list:
    """Verletzte Mengen als Liste (x(δ(S)), S), am stärksten verletzte zuerst (Gleichstand nach Rundung auf 7 Stellen: Größe, dann Knotennummern).
    Voraussetzung: die Gradgleichungen gelten."""
    teile = komponenten(n, kanten, x)
    if len(teile) > 1:
        mengen = {kleinere_seite(n, t) for t in teile}
        return [(0.0, s) for s in sorted(mengen, key=lambda s: (len(s), sorted(s)))]
    graph = [(u, v, float(w)) for (u, v), w in zip(kanten, x) if w > 1e-9]
    gefunden = {}
    for w, seite in T.baumschnitte(n, graph, grenze=2.0):
        if w < 2.0 - tol:
            gefunden[kleinere_seite(n, seite)] = w
    return sorted(((w, s) for s, w in gefunden.items()), key=T.reihenfolge_schluessel)


def zeile(kanten, menge):
    """Die Zeile x(δ(S)) >= 2 als (Spalten, Koeffizienten, rechte Seite)."""
    spalten = LP.schnittkanten(kanten, menge)
    return spalten, [1.0] * len(spalten), 2.0
