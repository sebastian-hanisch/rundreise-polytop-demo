"""Instanzen der symmetrischen Rundreise: n Orte (Knoten des vollständigen Graphen) mit ganzzahligen Kosten je Paar.

Drei Zufallsfamilien und eine Beispielfamilie:
  Karte          n Orte mit Koordinaten (below(1001), below(1001)); Kosten = Abstand, auf eine ganze Zahl gerundet (durch das Runden nur nahezu metrisch)
  Haufen         erst 4 Mittelpunkte (150 + below(701) je Koordinate), dann je Ort ein Mittelpunkt below(4) und je Koordinate ein Versatz -100 + below(201)
  Zufallsmatrix  je Paar u < v Kosten 1 + below(1000), nicht metrisch
  Prisma         zwei Dreiecke, drei Wege aus k Kanten dazwischen; Dreieckskanten kosten c, Wegkanten v, alle übrigen Paare den kürzesten Weg

Zufall nur über SplitMix64 (rpt_rng.py): gleiche Zufalls-Seeds geben auf jeder Plattform dieselbe Instanz."""

from dataclasses import dataclass
from itertools import combinations
from math import cos, isqrt, pi, sin
from typing import Optional

from rpt_rng import SplitMix64

ZUFALLSFAMILIEN = ("karte", "haufen", "matrix")


@dataclass(frozen=True)
class Instanz:
    familie: str
    n: int
    kosten: tuple                  # ganzzahlig, in der Reihenfolge der Paare (u, v) mit u < v
    punkte: tuple                  # Lage je Knoten für die Zeichnung (bei der Zufallsmatrix ein Kreis ohne Bedeutung)
    namen: tuple
    netz: tuple = ()               # Prisma: die gezeichneten Grundkanten ((u, v, Länge), ...)

    @property
    def kanten(self) -> list:
        return paare(self.n)

    def kosten_von(self, u: int, v: int) -> int:
        a, b = min(u, v), max(u, v)
        return self.kosten[kantennummer(self.n, a, b)]


def paare(n: int) -> list:
    """Alle Paare (u, v) mit u < v in fester Reihenfolge: die Kanten des vollständigen Graphen."""
    return list(combinations(range(n), 2))


def kantennummer(n: int, u: int, v: int) -> int:
    """Nummer der Kante {u, v} in der Reihenfolge von `paare`."""
    a, b = min(u, v), max(u, v)
    return a * n - a * (a + 1) // 2 + (b - a - 1)


def gerundeter_abstand(d2: int) -> int:
    """Wurzel aus dem ganzzahligen Abstandsquadrat d2, kaufmännisch auf eine ganze Zahl gerundet (nur ganze Zahlen)."""
    return (isqrt(4 * d2) + 1) // 2


def _abstaende(punkte) -> tuple:
    return tuple(gerundeter_abstand((punkte[u][0] - punkte[v][0]) ** 2 + (punkte[u][1] - punkte[v][1]) ** 2)
                 for u, v in paare(len(punkte)))


def karte_punkte(rng: SplitMix64, n: int) -> list:
    return [(rng.below(1001), rng.below(1001)) for _ in range(n)]


def haufen_punkte(rng: SplitMix64, n: int) -> list:
    mitten = [(150 + rng.below(701), 150 + rng.below(701)) for _ in range(4)]
    punkte = []
    for _ in range(n):
        mx, my = mitten[rng.below(4)]
        dx = rng.below(201) - 100
        punkte.append((mx + dx, my + rng.below(201) - 100))
    return punkte


def kreis(n: int, radius: float = 450.0, mitte: float = 500.0) -> list:
    """n Punkte gleichmäßig auf einem Kreis (Zeichnung der Zufallsmatrix, deren Orte keine Lage haben)."""
    return [(mitte + radius * cos(pi / 2 - 2 * pi * i / n), mitte + radius * sin(pi / 2 - 2 * pi * i / n)) for i in range(n)]


def _namen_zahlen(n: int) -> tuple:
    return tuple(str(i + 1) for i in range(n))


def zufallsinstanz(familie: str, n: int, seed: int) -> Instanz:
    """Instanz einer Zufallsfamilie aus (Familie, n, Zufalls-Seed)."""
    if n < 3:
        raise ValueError("eine Rundreise braucht mindestens 3 Orte")
    rng = SplitMix64(seed)
    if familie == "matrix":
        kosten = tuple(1 + rng.below(1000) for _ in paare(n))
        punkte = kreis(n)
    elif familie == "karte":
        punkte = karte_punkte(rng, n)
        kosten = _abstaende(punkte)
    elif familie == "haufen":
        punkte = haufen_punkte(rng, n)
        kosten = _abstaende(punkte)
    else:
        raise ValueError(f"unbekannte Familie {familie!r}")
    return Instanz(familie, n, kosten, tuple(punkte), _namen_zahlen(n))


def prisma_grundkanten(k: int, dreieck: int, weg: int) -> tuple:
    """Grundkanten der Prisma-Familie: Dreiecke {0,1,2} und {3,4,5}; Weg i verbindet i und i + 3 über k - 1 Zwischenknoten.
    Rückgabe (Knotenzahl, [(u, v, Länge)], Wege als Knotenlisten)."""
    if k < 1:
        raise ValueError("jeder Weg braucht mindestens eine Kante")
    kanten = [(0, 1, dreieck), (1, 2, dreieck), (0, 2, dreieck), (3, 4, dreieck), (4, 5, dreieck), (3, 5, dreieck)]
    wege, n = [], 6
    for i in range(3):
        knoten = [i]
        for _ in range(k - 1):
            knoten.append(n)
            n += 1
        knoten.append(i + 3)
        kanten += [(a, b, weg) for a, b in zip(knoten, knoten[1:])]
        wege.append(knoten)
    return n, kanten, wege


def kuerzeste_wege(n: int, kanten) -> list:
    """Abstandsmatrix aller kürzesten Wege (Floyd-Warshall, ganzzahlig): macht die Kosten metrisch."""
    gross = 10 ** 9
    d = [[0 if i == j else gross for j in range(n)] for i in range(n)]
    for u, v, w in kanten:
        d[u][v] = d[v][u] = min(d[u][v], w)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def prisma_lage(k: int) -> list:
    """Ebene Zeichnung des Prismas: Dreieck 0-1-2 außen, 3-4-5 innen, die Zwischenknoten auf den Verbindungsstrecken."""
    aussen = [(500 + 470 * cos(w), 480 + 470 * sin(w)) for w in (pi / 2, pi / 2 + 2 * pi / 3, pi / 2 + 4 * pi / 3)]
    innen = [(500 + 150 * cos(w), 480 + 150 * sin(w)) for w in (pi / 2, pi / 2 + 2 * pi / 3, pi / 2 + 4 * pi / 3)]
    punkte = aussen + innen
    for i in range(3):
        (ax, ay), (bx, by) = aussen[i], innen[i]
        punkte += [(ax + (bx - ax) * s / k, ay + (by - ay) * s / k) for s in range(1, k)]
    return punkte


def prisma_namen(n: int) -> tuple:
    return tuple("abcdefghijklmnopqrstuvwxyz"[:n])


def prisma(k: int = 1, dreieck: int = 1, weg: int = 1) -> Instanz:
    """Prisma-Instanz mit 3k + 3 Orten; Kosten = kürzeste Wege im Prisma."""
    n, grund, _ = prisma_grundkanten(k, dreieck, weg)
    d = kuerzeste_wege(n, grund)
    kosten = tuple(d[u][v] for u, v in paare(n))
    return Instanz("prisma", n, kosten, tuple(prisma_lage(k)), prisma_namen(n), tuple(grund))


def erzeuge(familie: str, n: int = 0, seed: int = 0, k: int = 1, dreieck: int = 1, weg: int = 1) -> Instanz:
    """Dünne Komposition: Prisma aus (k, dreieck, weg), sonst Zufallsfamilie aus (n, seed)."""
    if familie == "prisma":
        return prisma(k, dreieck, weg)
    return zufallsinstanz(familie, n, seed)


def menge_name(inst: Instanz, menge, grenze: int = 14) -> str:
    """Lesbarer Name einer Knotenmenge, lange Mengen gekürzt."""
    teile = [inst.namen[v] for v in sorted(menge)]
    if len(teile) > grenze:
        return "{" + ", ".join(teile[:grenze]) + f", … ({len(teile)} Orte)}}"
    return "{" + ", ".join(teile) + "}"


def kante_name(inst: Instanz, nummer: int, kanten: Optional[list] = None) -> str:
    u, v = (kanten or inst.kanten)[nummer]
    return f"{inst.namen[u]}–{inst.namen[v]}"
