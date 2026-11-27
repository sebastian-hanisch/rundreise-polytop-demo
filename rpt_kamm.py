"""Kämme mit beliebig großen Zähnen, exakt getrennt bis 12 Orte.

Kamm (Schnittform): Griff H und p >= 3 (ungerade) paarweise disjunkte Zähne T_j, jeder mit einem Ort in H und einem außerhalb;
x(δ(H)) + Σ x(δ(T_j)) >= 3p + 1. Gelten die Subtourzeilen, ist jeder Zahnschnitt mindestens 2; verletzt ist ein Kamm dann genau, wenn
Σ (x(δ(T_j)) − 2) < p + 1 − x(δ(H)). Danach wird vollständig durchsucht: jeder Griff (H und Komplement geben denselben Kamm) und alle
Zahnfamilien innerhalb dieser Schranke. Bis 12 Orte sind das p = 3 oder 5."""

import numpy as np

import rpt_lp as LP
import rpt_subtour as ST

MAX_N = 12
TOL = 1e-6


def knotenzahlen(n: int) -> np.ndarray:
    """Zahl der Orte je Bitmaske 0 .. 2^n − 1."""
    c = np.zeros(1 << n, dtype=np.int8)
    for v in range(n):
        c[1 << v: 2 << v] = c[:1 << v] + 1
    return c


def innensummen(n: int, kanten, x) -> np.ndarray:
    """x(E[U]) für alle 2^n Knotenmengen U (Bitmaske), durch Verdoppeln."""
    w = np.zeros((n, n))
    for (u, v), wert_e in zip(kanten, x):
        w[u, v] = w[v, u] = wert_e
    f = np.zeros(1 << n)
    for v in range(n):
        g = np.zeros(1 << v)
        for u in range(v):
            g[1 << u: 2 << u] = g[:1 << u] + w[v, u]
        f[1 << v: 2 << v] = f[:1 << v] + g
    return f


def schnittwerte(n: int, kanten, x) -> np.ndarray:
    """x(δ(T)) für alle Teilmengen T; Voraussetzung: x(δ(v)) = 2, also x(δ(T)) = 2|T| − 2·x(E[T])."""
    return 2.0 * knotenzahlen(n) - 2.0 * innensummen(n, kanten, x)


def maske_zu_menge(maske: int) -> frozenset:
    return frozenset(i for i in range(maske.bit_length()) if maske >> i & 1)


def _suche_zaehne(kandidaten, schlupf, p: int, grenze: float):
    """Unter den nach Schlupf aufsteigend sortierten Kandidaten p disjunkte mit kleinster Schlupfsumme < grenze: (Masken, Summe) oder (None, grenze)."""
    gewaehlt, bestes = [], [None, grenze]

    def rekursion(start, benutzt, summe):
        fehlt = p - len(gewaehlt)
        if fehlt == 0:
            bestes[0], bestes[1] = tuple(gewaehlt), summe
            return
        for i in range(start, len(kandidaten) - fehlt + 1):
            if summe + schlupf[i] * fehlt >= bestes[1] - 1e-9:
                break
            if kandidaten[i] & benutzt:
                continue
            gewaehlt.append(kandidaten[i])
            rekursion(i + 1, benutzt | kandidaten[i], summe + schlupf[i])
            gewaehlt.pop()

    rekursion(0, 0, 0.0)
    return bestes[0], bestes[1]


def trenne_kamm(n: int, kanten, x, tol: float = TOL):
    """Am stärksten verletzter Kamm als (Verletzung, Griff, [Zähne]) oder None. Voraussetzung: x erfüllt Grad- und Subtourzeilen.
    Gleichstand: der erste Fund (kleineres p, kleinere Griffmaske, Zähne nach Schlupf und Maske)."""
    if n > MAX_N:
        raise ValueError(f"die Kammsuche ist auf {MAX_N} Orte begrenzt")
    c = schnittwerte(n, kanten, x)
    voll = (1 << n) - 1
    masken = np.arange(1 << n)
    anzahl = knotenzahlen(n)
    bestes = None
    for h in range(1, 1 << (n - 1)):
        a = c[h]
        for p in (3, 5):
            if 2 * p > n:
                break
            budget = p + 1 - a - tol
            if bestes is not None:
                budget = min(budget, p + 1 - a - bestes[0] - 1e-9)
            if budget <= 0:
                continue
            gueltig = ((masken & h) != 0) & ((masken & ~h & voll) != 0) & (c - 2.0 < budget) & (anzahl <= n - 2 * (p - 1))
            kandidaten = np.flatnonzero(gueltig)
            if len(kandidaten) < p:
                continue
            schlupf = np.maximum(c[kandidaten] - 2.0, 0.0)
            ordnung = np.lexsort((kandidaten, schlupf))
            zaehne, summe = _suche_zaehne([int(v) for v in kandidaten[ordnung]], [float(v) for v in schlupf[ordnung]], p, budget)
            if zaehne is not None:
                verletzung = p + 1 - a - summe
                if bestes is None or verletzung > bestes[0] + 1e-9:
                    bestes = (float(verletzung), h, zaehne)
    if bestes is None:
        return None
    return bestes[0], ST.kleinere_seite(n, maske_zu_menge(bestes[1])), [maske_zu_menge(t) for t in bestes[2]]


def zeile(kanten, griff, zaehne):
    """Die Kammungleichung als LP-Zeile: Σ über δ(H) und alle δ(T_j) >= 3p + 1."""
    koeffizient = {}
    for menge in (griff, *zaehne):
        for j in LP.schnittkanten(kanten, menge):
            koeffizient[j] = koeffizient.get(j, 0.0) + 1.0
    spalten = sorted(koeffizient)
    return spalten, [koeffizient[j] for j in spalten], 3.0 * len(zaehne) + 1.0


def ist_kamm(n: int, griff, zaehne) -> bool:
    """Unabhängiger Prüfer der Form: ungerade viele (>= 3) disjunkte Zähne, jeder mit Orten in und außerhalb des Griffs."""
    griff = set(griff)
    if len(zaehne) < 3 or len(zaehne) % 2 == 0 or not griff or len(griff) >= n:
        return False
    belegt = set()
    for t in zaehne:
        if belegt & set(t) or not set(t) & griff or not set(t) - griff:
            return False
        belegt |= set(t)
    return True
