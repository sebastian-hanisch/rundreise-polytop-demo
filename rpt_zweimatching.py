"""2-Matching-Ungleichungen und ihre Trennung.

Form: Griff H (Knotenmenge) und eine ungerade Menge F von Kanten in δ(H):  x(δ(H) ohne F) + Σ_{e in F} (1 − x_e) >= 1.
Der Wert der linken Seite heißt hier „Wert“; verletzt ist die Ungleichung, wenn der Wert unter 1 liegt.

Drei Trennungen: exakt durch Aufzählen aller Griffe (nur klein), exakt nach Padberg und Rao (kleinste ungerade Schnitte im geteilten Hilfsgraphen,
Gomory-Hu-Baum aus rpt_trennung) und die Komponentenregel (Griffe nur aus den Komponenten der gebrochenen Kanten; eine Heuristik).

Hilfsgraph: jede Trägerkante uv wird in u – w (Kapazität x) und w – v (Kapazität 1 − x) geteilt. Ein Schnitt unter 1 trennt von jeder geteilten Kante
höchstens eine Hälfte; die Kanten, bei denen er die (1 − x)-Hälfte trennt, bilden F. Markiert sind alle w und die Orte, an denen ungerade viele
(1 − x)-Hälften hängen; damit hat die Zahl markierter Knoten einer Seite die Parität von |F|.

Zahnzerlegung (eigene Herleitung, nicht extern geprüft; die Tests prüfen die Identität mit exakten Brüchen an Punkten, die nur die
Gradgleichungen erfüllen): Liegen zwei Kanten aus F an einem Ort v und keine dritte, bringt man v auf die andere Seite von H und streicht beide
Kanten aus F. Mit x(δ(v)) = 2 ist der neue Wert gleich dem alten minus 2·S, wobei S die Summe der übrigen Kanten an v ist, die den alten Griff
kreuzen. Die verkürzte Ungleichung ist deshalb an jedem zulässigen Punkt mindestens so stark. Drei Kanten aus F an einem Ort sind nie verletzt
(ihre Beiträge (1 − x) sind zusammen mindestens 3 − 2 = 1), ebenso |F| = 1 unter den Subtourzeilen. So wird jede verletzte Zeile zu einer mit
paarweise disjunkten Kanten, also zu einem Kamm mit Zähnen aus je zwei Orten."""

from collections import namedtuple

import numpy as np

import rpt_lp as LP
import rpt_subtour as ST
import rpt_trennung as T

TOL = 1e-6
AUFZAEHL_MAX_N = 16

Zeile = namedtuple("Zeile", "wert griff zaehne")      # griff: frozenset, zaehne: sortiertes Tupel von Kantennummern


def sortierschluessel(z: Zeile):
    """Feste Gleichstandsregel: Wert (auf 7 Stellen), Größe des Griffs, Knotennummern, Zahnkanten."""
    return round(z.wert, 7), len(z.griff), sorted(z.griff), z.zaehne


def wert(kanten, x, griff, zaehne) -> float:
    """x(δ(H) ohne F) + Σ_F (1 − x_e)."""
    f = set(zaehne)
    return float(sum((1.0 - x[j]) if j in f else x[j] for j in LP.schnittkanten(kanten, griff)))


def beste_zaehne(kanten, x, griff):
    """Die beste ungerade Zahnmenge zu einem Griff: erst alle Kanten mit x > 1/2, bei gerader Zahl die Kante mit |1 − 2x| am kleinsten dazu oder weg
    (Gleichstand: kleinste Kantennummer). Rückgabe (Wert, F) oder None, wenn δ(H) leer ist."""
    rand = LP.schnittkanten(kanten, griff)
    if not rand:
        return None
    f = [j for j in rand if x[j] > 0.5 + 1e-9]
    w = sum(min(x[j], 1.0 - x[j]) for j in rand)
    if len(f) % 2 == 0:
        j = min(rand, key=lambda k: (round(abs(1.0 - 2.0 * x[k]), 9), k))
        w += abs(1.0 - 2.0 * x[j])
        f = sorted(set(f) ^ {j})
    return float(w), tuple(f)


def hilfsgraph(n: int, kanten, x, tol: float = 1e-9):
    """Der geteilte Hilfsgraph. Rückgabe (Knotenzahl, [(a, b, Kapazität)], Markierung, Teilkanten [(Kantennummer, Mittelknoten)]).
    Die Knoten 0 .. n−1 sind die Orte, die weiteren die Mittelknoten w."""
    graph, markiert, teil = [], [0] * n, []
    knoten = n
    for j, ((u, v), w) in enumerate(zip(kanten, x)):
        if w <= tol:
            continue
        w = min(1.0, float(w))
        graph.append((u, knoten, w))
        graph.append((knoten, v, 1.0 - w))
        markiert[v] ^= 1
        markiert.append(1)
        teil.append((j, knoten))
        knoten += 1
    return knoten, graph, markiert, teil


def trenne_padberg_rao(n: int, kanten, x, tol: float = TOL) -> list:
    """Exakte Trennung nach Padberg und Rao: alle verletzten ungeraden Baumschnitte als Zeilen, am stärksten verletzte zuerst.
    Leere Liste: keine 2-Matching-Ungleichung ist verletzt (bis auf die Toleranz). Die Schnittwerte werden mit der besten Zahnmenge des Griffs neu
    berechnet; Voraussetzung: x liegt in den Schranken 0 <= x <= 1."""
    knoten, graph, markiert, _ = hilfsgraph(n, kanten, x)
    gefunden = {}
    for _, seite in T.ungerade_schnitte(knoten, graph, markiert, grenze=1.0, tol=tol):
        griff = ST.kleinere_seite(n, (a for a in seite if a < n))
        if not griff:
            continue
        bestes = beste_zaehne(kanten, x, griff)
        if bestes is not None and bestes[0] < 1.0 - tol:
            gefunden[(griff, bestes[1])] = bestes[0]
    return sorted((Zeile(w, h, f) for (h, f), w in gefunden.items()), key=sortierschluessel)


def trenne_komponentenregel(n: int, kanten, x, tol: float = TOL) -> list:
    """Heuristik: Griffe sind nur die Zusammenhangskomponenten des Graphen der gebrochenen Kanten (1e-6 < x < 1 − 1e-6), die eine solche Kante
    enthalten; jede bekommt ihre beste ungerade Zahnmenge. Findet sie nichts, kann trotzdem etwas verletzt sein."""
    beruehrt = {v for (u, v), w in zip(kanten, x) if 1e-6 < w < 1.0 - 1e-6} | {u for (u, v), w in zip(kanten, x) if 1e-6 < w < 1.0 - 1e-6}
    zeilen = []
    for griff in ST.komponenten(n, kanten, x, 1e-6, 1.0 - 1e-6):
        if not griff & beruehrt:
            continue
        bestes = beste_zaehne(kanten, x, griff)
        if bestes is not None and bestes[0] < 1.0 - tol:
            zeilen.append(Zeile(bestes[0], griff, bestes[1]))
    # stabil nach dem Wert: bei Gleichstand bleibt die Reihenfolge der Komponenten (kleinster Knoten zuerst)
    return sorted(zeilen, key=lambda z: round(z.wert, 7))


def staerkste_durch_aufzaehlen(n: int, kanten, x):
    """Kleinster Wert über ALLE Griffe (Knoten n−1 nie im Griff) mit der jeweils besten ungeraden Zahnmenge: (Wert, Griff). Nur bis n = 16."""
    if n > AUFZAEHL_MAX_N:
        raise ValueError(f"Aufzählen nur bis {AUFZAEHL_MAX_N} Orte")
    masken = np.arange(1, 1 << (n - 1), dtype=np.int64)
    bits = ((masken[:, None] >> np.arange(n)) & 1).astype(bool)
    xa = np.asarray(x, float)
    rand = bits[:, [u for u, _ in kanten]] != bits[:, [v for _, v in kanten]]
    gesamt = (rand * np.minimum(xa, 1.0 - xa)).sum(1)
    ungerade = (rand & (xa > 0.5)).sum(1) % 2 == 1
    wechsel = np.where(rand, np.abs(1.0 - 2.0 * xa), np.inf).min(1)
    gesamt = gesamt + np.where(ungerade, 0.0, wechsel)
    i = int(np.argmin(gesamt))
    return float(gesamt[i]), frozenset(int(a) for a in np.flatnonzero(bits[i]))


def zeile(kanten, griff, zaehne):
    """Die 2-Matching-Ungleichung als LP-Zeile Σ a·x >= b: x(δ(H) ohne F) − x(F) >= 1 − |F|."""
    f = set(zaehne)
    spalten = LP.schnittkanten(kanten, griff)
    return spalten, [-1.0 if j in f else 1.0 for j in spalten], 1.0 - len(f)


def zaehne_disjunkt(kanten, zaehne) -> bool:
    enden = [a for j in zaehne for a in kanten[j]]
    return len(enden) == len(set(enden))


def reduziere_schritt(n: int, kanten, griff: frozenset, zaehne: tuple):
    """Ein Schritt der Zahnzerlegung: der kleinste Ort mit genau zwei Zahnkanten wechselt die Seite, beide Kanten verlassen F.
    Rückgabe (Griff, F) oder None, wenn die Zahnkanten schon disjunkt sind oder an jedem gemeinsamen Ort mindestens drei Kanten liegen."""
    an = {}
    for j in zaehne:
        for v in kanten[j]:
            an.setdefault(v, []).append(j)
    for v in sorted(an):
        if len(an[v]) == 2:
            neu = griff ^ {v}
            if not neu or len(neu) == n:
                return None
            return neu, tuple(j for j in zaehne if j not in an[v])
    return None


def kammform(n: int, kanten, griff, zaehne):
    """Verkürzt wiederholt, bis die Zahnkanten disjunkt sind (so weit möglich). Rückgabe (Griff in kleinerer Seite, F, disjunkt)."""
    griff, zaehne = frozenset(griff), tuple(zaehne)
    while True:
        schritt = reduziere_schritt(n, kanten, griff, zaehne)
        if schritt is None:
            return ST.kleinere_seite(n, griff), zaehne, zaehne_disjunkt(kanten, zaehne)
        griff, zaehne = schritt


def kammwert(kanten, x, griff, zaehne) -> tuple:
    """Dieselbe Ungleichung als Kamm: (x(δ(H)) + Σ x(δ(T)), 3p + 1) mit Zähnen T = Endpunkte der Kanten aus F. Bei x(δ(v)) = 2 gilt
    linke Seite − (3p + 1) = Wert − 1, weil x(δ(T)) = 4 − 2·x_e."""
    links = ST.wert(kanten, x, griff) + sum(ST.wert(kanten, x, frozenset(kanten[j])) for j in zaehne)
    return links, 3 * len(zaehne) + 1
