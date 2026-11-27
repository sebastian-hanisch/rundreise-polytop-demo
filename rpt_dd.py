"""Doppelte Beschreibung in ganzen Zahlen: aus Punkten die Ungleichungen der konvexen Hülle, aus Ungleichungen die Ecken.

Kopie von gze_dd.py aus ganzzahlige-ecken-demo (https://github.com/sebastian-hanisch/ganzzahlige-ecken-demo); ab der ersten Importzeile unverändert
(geprüft in tests/test_kopien.py). In dieser Demo berechnet es die ganze Hülle aller Rundreisen bei kleinen n.

Kern ist eine einzige Aufgabe: alle Extremstrahlen des Kegels {y : z·y >= 0 für alle Zeilen z} (Verfahren der doppelten Beschreibung: Start mit
einem simplizialen Kegel aus linear unabhängigen Zeilen, dann je Zeile ein Schritt; zwei Strahlen werden nur kombiniert, wenn sie benachbart sind,
geprüft über die gemeinsam straffen Zeilen). Gerechnet wird mit ganzen Zahlen und Brüchen, ohne Gleitkomma. Jeder Lauf zählt Strahlen und
Strahlenpaare; wird eine Obergrenze überschritten, endet er mit ZuGross (Ergebnis nicht bestimmt).

Nur Standardbibliothek, kein Zufall, kein Oberflächen-Code.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

MAX_STRAHLEN = 20_000
MAX_PAARE = 20_000_000


class ZuGross(Exception):
    """Eine Obergrenze wurde überschritten: Ergebnis nicht bestimmt."""


class Unbeschraenkt(Exception):
    """Die Ungleichungen beschreiben kein beschränktes Polytop."""


@dataclass(frozen=True)
class Huelle:
    dimension: int        # affine Dimension der Punktmenge
    gleichungen: tuple    # ((a, b), ...) mit a·x == b: die affine Hülle (leer, wenn volldimensional)
    facetten: tuple       # ((a, b), ...) mit a·x <= b, ganzzahlig und teilerfremd; in eliminierten Koordinaten steht 0


# ---------------------------------------------------------------- lineare Algebra mit Brüchen

def teilerfremd(vektor) -> tuple:
    """Ganzzahligen Vektor durch den größten gemeinsamen Teiler seiner Einträge teilen."""
    g = 0
    for c in vektor:
        g = gcd(g, c)
    return tuple(c // g for c in vektor) if g > 1 else tuple(vektor)


def _reduziere(zeile: tuple, basis: list) -> tuple:
    """Ganzzahlige Zeile gegen eine Stufenbasis [(pivotspalte, zeile)] reduzieren, ohne Brüche (danach teilerfremd)."""
    for spalte, b in basis:
        f = zeile[spalte]
        if f:
            zeile = teilerfremd([b[spalte] * x - f * y for x, y in zip(zeile, b)])
    return zeile


def unabhaengige_zeilen(zeilen, ziel: int) -> list:
    """Indizes linear unabhängiger ganzzahliger Zeilen, gierig in gegebener Reihenfolge, höchstens `ziel` viele."""
    basis, indizes = [], []
    for i, z in enumerate(zeilen):
        if len(indizes) == ziel:
            break
        rest = _reduziere(tuple(z), basis)
        spalte = next((s for s, c in enumerate(rest) if c), None)
        if spalte is None:
            continue
        basis.append((spalte, rest))
        indizes.append(i)
    return indizes


def rang(vektoren) -> int:
    """Rang einer Liste ganzzahliger Vektoren."""
    vektoren = list(vektoren)
    return len(unabhaengige_zeilen(vektoren, len(vektoren[0]))) if vektoren else 0


def affiner_rang(punkte) -> int:
    """Dimension der affinen Hülle einer nichtleeren Liste ganzzahliger Punkte."""
    punkte = list(punkte)
    p0 = punkte[0]
    return rang([[a - b for a, b in zip(p, p0)] for p in punkte[1:]]) if len(punkte) > 1 else 0


def startstrahlen(matrix) -> list:
    """Zu einer invertierbaren ganzzahligen d x d Matrix die Strahlen r_k mit z_j·r_k = 0 (j != k) und z_k·r_k > 0, ganzzahlig und teilerfremd."""
    d = len(matrix)
    m = [[Fraction(c) for c in zeile] + [Fraction(int(i == k)) for i in range(d)] for k, zeile in enumerate(matrix)]
    for s in range(d):
        p = next(z for z in range(s, d) if m[z][s])
        m[s], m[p] = m[p], m[s]
        pivot = m[s][s]
        m[s] = [c / pivot for c in m[s]]
        for z in range(d):
            if z != s and m[z][s]:
                f = m[z][s]
                m[z] = [a - f * b for a, b in zip(m[z], m[s])]
    strahlen = []
    for k in range(d):
        spalte = [m[z][d + k] for z in range(d)]
        nenner = 1
        for c in spalte:
            nenner = lcm(nenner, c.denominator)
        strahlen.append(teilerfremd([int(c * nenner) for c in spalte]))
    return strahlen


# ---------------------------------------------------------------- doppelte Beschreibung

def benachbart(gemeinsam: int, masken: list, dim: int) -> bool:
    """Nachbarschaftstest: Zwei Strahlen sind benachbart, wenn sie mindestens dim - 2 straffe Zeilen gemeinsam haben (`gemeinsam`, Bitmaske über
    Zeilen) und kein dritter Strahl alle diese Zeilen ebenfalls straff erfüllt. `masken` sind die Straffmasken aller Strahlen (die beiden
    geprüften eingeschlossen). Durchläuft die Strahlen; der Abbruch kommt beim dritten Treffer."""
    if gemeinsam.bit_count() < dim - 2:
        return False
    zahl = 0
    for t in masken:
        if gemeinsam & t == gemeinsam:
            zahl += 1
            if zahl > 2:
                return False
    return True


def spaltenmasken(masken: list, zeilenzahl: int) -> list:
    """Dieselbe Auskunft spaltenweise: spalten[j] ist die Bitmaske (über Strahlindizes) der Strahlen, die Zeile j straff erfüllen."""
    spalten = [0] * zeilenzahl
    for index, straff in enumerate(masken):
        bit = 1 << index
        while straff:
            b = straff & -straff
            spalten[b.bit_length() - 1] |= bit
            straff ^= b
    return spalten


def benachbart_spaltenweise(gemeinsam: int, spalten: list, paar: int, alle: int, dim: int) -> bool:
    """Derselbe Test über die Spaltenmasken: Der Schnitt aller Spalten der gemeinsamen Zeilen enthält genau die zwei geprüften Strahlen
    (`paar`, Bitmaske über Strahlindizes; `alle` ist die Maske aller Strahlen)."""
    if gemeinsam.bit_count() < dim - 2:
        return False
    rest = alle
    while gemeinsam and rest != paar:
        b = gemeinsam & -gemeinsam
        rest &= spalten[b.bit_length() - 1]
        gemeinsam ^= b
    return rest == paar


def zeile_einfuegen(strahlen: list, zeile, nummer: int, dim: int, budget: list, max_strahlen: int) -> list:
    """Ein Schritt: Die Zeile `nummer` kommt hinzu. strahlen ist eine Liste (Strahl, Straffmaske über die bisherigen Zeilen).
    Strahlen mit negativem Wert fallen weg; je benachbartem Paar (positiv, negativ) entsteht ein neuer Strahl auf der Hyperebene.
    budget = [verbleibende Paare] wird verringert."""
    nz = [(i, c) for i, c in enumerate(zeile) if c]
    bit = 1 << nummer
    pos, neg, null = [], [], []
    for index, (r, t) in enumerate(strahlen):
        v = 0
        for i, c in nz:
            v += c * r[i]
        if v > 0:
            pos.append((index, r, t, v))
        elif v < 0:
            neg.append((index, r, t, v))
        else:
            null.append((r, t | bit))
    neu = [(r, t) for _, r, t, _ in pos] + null
    if pos and neg:
        paare = len(pos) * len(neg)
        budget[0] -= paare
        if budget[0] < 0:
            raise ZuGross("zu viele Strahlenpaare")
        masken = [t for _, t in strahlen]
        # Zwei gleichwertige Tests; gewählt wird der billigere: das Aufbauen der Spalten kostet je straffem Eintrag einen Schritt,
        # der zeilenweise Test je Paar bis zu einen Schritt je Strahl.
        spalten = None
        if paare * len(masken) > 4 * sum(t.bit_count() for t in masken):
            spalten = spaltenmasken(masken, nummer)
            alle = (1 << len(masken)) - 1
        for ip, rp, tp, vp in pos:
            for im, rm, tm, vm in neg:
                gemeinsam = tp & tm
                if spalten is None:
                    nachbarn = benachbart(gemeinsam, masken, dim)
                else:
                    nachbarn = benachbart_spaltenweise(gemeinsam, spalten, (1 << ip) | (1 << im), alle, dim)
                if nachbarn:
                    neu.append((teilerfremd([vp * b - vm * a for a, b in zip(rp, rm)]), gemeinsam | bit))
            if len(neu) > max_strahlen:
                raise ZuGross("zu viele Strahlen")
    if len(neu) > max_strahlen:
        raise ZuGross("zu viele Strahlen")
    return neu


def extremstrahlen(zeilen, max_strahlen: int = MAX_STRAHLEN, max_paare: int = MAX_PAARE) -> list:
    """Alle Extremstrahlen von {y : z·y >= 0 für z in zeilen} als teilerfremde ganzzahlige Tupel. Die Zeilen müssen vollen Spaltenrang haben
    (spitzer Kegel). Reihenfolge der Zeilen: Sie bestimmt Rechenaufwand und Ausgabereihenfolge, nicht die Menge der Strahlen."""
    zeilen = [tuple(int(c) for c in z) for z in zeilen]
    if not zeilen:
        raise ValueError("keine Zeilen")
    dim = len(zeilen[0])
    basis = unabhaengige_zeilen(zeilen, dim)
    if len(basis) < dim:
        raise ValueError("Zeilen ohne vollen Spaltenrang: der Kegel enthält eine Gerade")
    voll = (1 << dim) - 1
    strahlen = [(r, voll & ~(1 << k)) for k, r in enumerate(startstrahlen([zeilen[i] for i in basis]))]
    gewaehlt = set(basis)
    budget = [max_paare]
    nummer = dim
    for i, z in enumerate(zeilen):
        if i in gewaehlt:
            continue
        strahlen = zeile_einfuegen(strahlen, z, nummer, dim, budget, max_strahlen)
        nummer += 1
    return [r for r, _ in strahlen]


# ---------------------------------------------------------------- Hülle und Ecken

def _affine_basis(punkte) -> tuple:
    """Zeilenstufenform der Differenzen zum ersten Punkt: (Pivotspalten, Zeilen aus Brüchen mit einer 1 auf dem eigenen Pivot und 0 auf den anderen)."""
    p0 = punkte[0]
    differenzen = [tuple(a - b for a, b in zip(p, p0)) for p in punkte[1:]]
    basis = []
    for i in unabhaengige_zeilen(differenzen, len(p0)):
        zeile = [Fraction(c) for c in differenzen[i]]
        for s, z in basis:
            f = zeile[s]
            if f:
                zeile = [x - f * y for x, y in zip(zeile, z)]
        spalte = next(s for s, c in enumerate(zeile) if c)
        pivot = zeile[spalte]
        zeile = [c / pivot for c in zeile]
        basis = [(s, [x - z[spalte] * y for x, y in zip(z, zeile)]) for s, z in basis]
        basis.append((spalte, zeile))
    basis.sort(key=lambda e: e[0])
    return [s for s, _ in basis], [z for _, z in basis]


def _gleichungen(p0, pivots, zeilen) -> tuple:
    """Gleichungen der affinen Hülle: je Nicht-Pivot-Spalte j gilt x_j - sum_k z_k[j] x_pivot(k) = p0_j - sum_k z_k[j] p0_pivot(k)."""
    out = []
    n = len(p0)
    for j in range(n):
        if j in pivots:
            continue
        a = [Fraction(0)] * n
        a[j] = Fraction(1)
        for s, z in zip(pivots, zeilen):
            a[s] -= z[j]
        b = sum((c * p for c, p in zip(a, p0)), Fraction(0))
        nenner = 1
        for c in a + [b]:
            nenner = lcm(nenner, c.denominator)
        ganz = teilerfremd([int(c * nenner) for c in a + [b]])
        out.append((ganz[:-1], ganz[-1]))
    return tuple(out)


def huelle(punkte, max_strahlen: int = MAX_STRAHLEN, max_paare: int = MAX_PAARE) -> Huelle:
    """Gleichungen und Facetten der konvexen Hülle ganzzahliger Punkte. Ist die Hülle nicht volldimensional, werden die Facetten in den
    Pivot-Koordinaten der affinen Hülle berechnet; in den übrigen Koordinaten steht der Koeffizient 0 (die Gleichungen stehen daneben)."""
    punkte = {tuple(int(c) for c in p) for p in punkte}
    if not punkte:
        raise ValueError("keine Punkte")
    n = len(next(iter(punkte)))
    if any(len(p) != n for p in punkte):
        raise ValueError("Punkte verschiedener Länge")
    # lexikografisch von der letzten Koordinate her: hält bei 0/1-Punkten die Zahl der Zwischenstrahlen klein (gemessen: Würfel, n = 12)
    punkte = sorted(punkte, key=lambda p: p[::-1])
    pivots, stufen = _affine_basis(punkte)
    d = len(pivots)
    gleich = _gleichungen(punkte[0], pivots, stufen)
    if d == 0:
        return Huelle(0, gleich, ())
    zeilen = [tuple(-p[s] for s in pivots) + (1,) for p in punkte]   # b - a·p >= 0, Unbekannte (a, b)
    facetten = []
    for r in extremstrahlen(zeilen, max_strahlen, max_paare):
        a = [0] * n
        for k, s in enumerate(pivots):
            a[s] = r[k]
        facetten.append((tuple(a), r[d]))
    return Huelle(d, gleich, tuple(sorted(facetten)))


def ecken(n: int, zeilen, max_strahlen: int = MAX_STRAHLEN, max_paare: int = MAX_PAARE) -> list:
    """Alle Ecken von {x >= 0, a·x <= b für (a, b) in zeilen} als Tupel von Brüchen (Länge n, sortiert).
    Leeres Polytop: leere Liste. Unbeschränkt: Unbeschraenkt."""
    kegel = [tuple(int(i == j) for j in range(n + 1)) for i in range(n + 1)]
    kegel += [tuple(-int(c) for c in a) + (int(b),) for a, b in zeilen]
    out = []
    unbeschraenkt = False
    for r in extremstrahlen(kegel, max_strahlen, max_paare):
        if r[n] == 0:
            unbeschraenkt = True
        else:
            out.append(tuple(Fraction(c, r[n]) for c in r[:n]))
    if unbeschraenkt and out:
        raise Unbeschraenkt("das System ist unbeschränkt")
    return sorted(out)


def wert(a, x):
    """Linke Seite a·x (ganzzahlig oder als Bruch, je nach x)."""
    return sum(c * v for c, v in zip(a, x) if c)


def pruefe_huelle(punkte, h: Huelle, max_strahlen: int = MAX_STRAHLEN, max_paare: int = MAX_PAARE) -> list:
    """Exakte Nachprüfung einer Hülle gegen ihre Punkte; Rückgabe: Liste der Verstöße (leer = bestanden).
      gültig       jeder Punkt erfüllt alle Gleichungen und Ungleichungen
      Facette      die straffen Punkte jeder Ungleichung haben die affine Dimension h.dimension - 1, keine doppelt
      Dimension    h.dimension und die Zahl der Gleichungen passen zu den Punkten
      vollständig  jede Ecke des beschriebenen Polytops ist einer der Punkte (Gegenrichtung der doppelten Beschreibung, ohne Vorzeichenannahme)"""
    punkte = list(dict.fromkeys(tuple(int(c) for c in p) for p in punkte))
    n = len(punkte[0])
    fehler = []
    if affiner_rang(punkte) != h.dimension or len(h.gleichungen) != n - h.dimension:
        fehler.append("Dimension oder Zahl der Gleichungen falsch")
    for a, b in h.gleichungen:
        if any(wert(a, p) != b for p in punkte):
            fehler.append(f"Gleichung {a} = {b} verletzt")
    if len(set(h.facetten)) != len(h.facetten):
        fehler.append("doppelte Facette")
    for a, b in h.facetten:
        werte = [wert(a, p) for p in punkte]
        if max(werte) > b:
            fehler.append(f"ungültig: {a} <= {b}")
            continue
        straff = [p for p, v in zip(punkte, werte) if v == b]
        # nicht alle Punkte straff: die straffen liegen in einer echten Hyperebene, ihr Rang ist höchstens dimension - 1;
        # es genügt also, bis dimension - 1 unabhängige Differenzen zu finden
        differenzen = [[x - y for x, y in zip(p, straff[0])] for p in straff[1:]]
        if not straff or len(straff) == len(punkte) or len(unabhaengige_zeilen(differenzen, h.dimension - 1)) != h.dimension - 1:
            fehler.append(f"keine Facette: {a} <= {b}")
    if fehler or h.dimension == 0:
        return fehler
    pivots, _ = _affine_basis(punkte)
    projiziert = {tuple(p[s] for s in pivots) for p in punkte}
    d = h.dimension
    kegel = [tuple([0] * d + [1])] + [tuple(-a[s] for s in pivots) + (b,) for a, b in h.facetten]
    try:
        strahlen = extremstrahlen(kegel, max_strahlen, max_paare)
    except ValueError:
        return ["unvollständig: die Ungleichungen beschränken nicht alle Richtungen"]
    for r in strahlen:
        if r[d] == 0:
            return ["unvollständig: das beschriebene Polyeder ist unbeschränkt"]
        if any(c % r[d] for c in r[:d]) or tuple(c // r[d] for c in r[:d]) not in projiziert:
            return ["unvollständig: das beschriebene Polytop hat eine Ecke außerhalb der Punktmenge"]
    return fehler
