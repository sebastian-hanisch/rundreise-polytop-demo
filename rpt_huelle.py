"""Die ganze Hülle aller Rundreisen bei kleinem n, mit der doppelten Beschreibung aus rpt_dd (ganze Zahlen, exakt) und der Einordnung jeder
Facette in eine Familie.

Facetten gelten modulo der Gradgleichungen: dieselbe Facette hat viele Schreibweisen. Zugeordnet wird deshalb über die Menge der straffen
Rundreisen. Familien: x_e >= 0, x_e <= 1, Subtour mit kleinerer Seite von 3 oder mehr Orten, Kämme mit 3 Zähnen aus je 2 Orten (das sind die
2-Matching-Ungleichungen mit 3 Zähnen), weitere Kämme mit 3 Zähnen (nach der Zahngröße der zuerst gefundenen Schreibweise), und der Rest."""

from itertools import combinations, permutations

import rpt_dd as DD
from rpt_instanzen import paare

LIVE_MAX_N = 6          # n = 7 rechnet tools/rpt_make_tables.py vor


def rundreisen(n: int) -> list:
    """Alle (n − 1)!/2 Rundreisen als 0/1-Tupel über die Kanten."""
    nummer = {e: j for j, e in enumerate(paare(n))}
    ergebnis = []
    for folge in permutations(range(1, n)):
        if folge[0] > folge[-1]:
            continue
        weg = (0,) + folge + (0,)
        x = [0] * len(nummer)
        for a, b in zip(weg, weg[1:]):
            x[nummer[(min(a, b), max(a, b))]] = 1
        ergebnis.append(tuple(x))
    return ergebnis


def kandidaten(n: int):
    """(Familienname, a, b) für Ungleichungen a·x <= b, in der Reihenfolge der Familien."""
    kanten = paare(n)
    for j in range(len(kanten)):
        yield "x ≥ 0", [-1 if k == j else 0 for k in range(len(kanten))], 0
    for j in range(len(kanten)):
        yield "x ≤ 1", [1 if k == j else 0 for k in range(len(kanten))], 1
    for groesse in range(3, n // 2 + 1):
        for s in combinations(range(n), groesse):
            yield "Subtour, kleinere Seite ab 3 Orten", [1 if (u in s and v in s) else 0 for u, v in kanten], groesse - 1
    for zahngroessen in ((2, 2, 2), (2, 2, 3)):
        name = "Kamm mit 3 Zähnen aus je 2 Orten (2-Matching)" if zahngroessen == (2, 2, 2) else "Kamm mit 3 Zähnen aus 2, 2 und 3 Orten"
        for a, b in kaemme(n, zahngroessen):
            yield name, a, b


def kaemme(n: int, zahngroessen) -> list:
    """Alle Kämme mit disjunkten Zähnen der genannten Größen in der Form x(E[H]) + Σ x(E[T]) <= |H| + Σ|T| − 5 (drei Zähne)."""
    kanten = paare(n)
    zaehne_je_groesse = {g: [frozenset(c) for c in combinations(range(n), g)] for g in set(zahngroessen)}
    ergebnis, gesehen = [], set()

    def waehle(i, benutzt, zaehne):
        if i == len(zahngroessen):
            schluessel = frozenset(zaehne)
            for h_groesse in range(1, n):
                for h in combinations(range(n), h_groesse):
                    h = frozenset(h)
                    if all(t & h and t - h for t in zaehne) and (schluessel, h) not in gesehen:
                        gesehen.add((schluessel, h))
                        a = [(1 if (u in h and v in h) else 0) + sum(1 for t in zaehne if u in t and v in t) for u, v in kanten]
                        ergebnis.append((a, len(h) + sum(len(t) for t in zaehne) - 5))
            return
        for t in zaehne_je_groesse[zahngroessen[i]]:
            if not t & benutzt:
                waehle(i + 1, benutzt | t, zaehne + [t])

    if sum(zahngroessen) <= n:
        waehle(0, frozenset(), [])
    return ergebnis


def straffe_menge(punkte, a, b) -> int:
    """Bitmaske der Rundreisen, die a·x <= b straff erfüllen; −1, wenn die Ungleichung eine Rundreise abschneidet."""
    maske = 0
    nz = [(i, c) for i, c in enumerate(a) if c]
    for k, p in enumerate(punkte):
        w = sum(c * p[i] for i, c in nz)
        if w > b:
            return -1
        if w == b:
            maske |= 1 << k
    return maske


def einordnen(n: int, huelle: DD.Huelle, punkte) -> dict:
    """Zahl der Facetten je Familie (die erste passende), dazu „weitere“."""
    offen = {straffe_menge(punkte, a, b) for a, b in huelle.facetten}
    zahlen = {}
    for name, a, b in kandidaten(n):
        zahlen.setdefault(name, 0)
        maske = straffe_menge(punkte, a, b)
        if maske in offen:
            offen.remove(maske)
            zahlen[name] += 1
    zahlen = {k: v for k, v in zahlen.items() if v}
    if offen:
        zahlen["weitere"] = len(offen)
    return zahlen


def huelle_rundreisen(n: int, max_strahlen: int = DD.MAX_STRAHLEN, max_paare: int = DD.MAX_PAARE):
    """Rückgabe (Hülle, Rundreisen); ZuGross, wenn die Obergrenzen nicht reichen."""
    punkte = rundreisen(n)
    return DD.huelle(punkte, max_strahlen, max_paare), punkte


def bericht(n: int) -> dict:
    """Dimension, Zahl der Ecken und Facetten und die Aufteilung nach Familien (nur für kleine n live)."""
    huelle, punkte = huelle_rundreisen(n)
    return {"n": n, "dimension": huelle.dimension, "ecken": len(punkte), "facetten": len(huelle.facetten),
            "familien": einordnen(n, huelle, punkte)}


def subtourfacetten(n: int) -> int:
    """Zahl verschiedener Subtourzeilen mit mindestens 2 Orten auf jeder Seite (S und Komplement gleich), eingeschlossen x <= 1: 2^(n−1) − n − 1.
    Sie sind für n >= 5 Facetten."""
    return 2 ** (n - 1) - n - 1
