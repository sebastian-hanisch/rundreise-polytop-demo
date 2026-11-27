"""Das LP der Rundreise: Gradgleichungen x(δ(v)) = 2, 0 <= x <= 1 und nachgeladene Zeilen der Form a·x >= b.

Gelöst mit HiGHS, dualer Simplex (scipy, method="highs-ds"); er liefert eine Ecke. Welche Ecke bei mehreren Optima erscheint, hängt vom Löser ab,
der LP-Wert nicht."""

from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

TOL_GANZ = 1e-5        # ganzzahlig / halbzahlig bis auf diese Toleranz


UNZULAESSIG = 2        # Status von scipy.optimize.linprog


class LPFehler(RuntimeError):
    """Der Löser hat kein Optimum gemeldet; `status` ist der Status von linprog."""

    def __init__(self, meldung: str, status: int = -1):
        super().__init__(meldung)
        self.status = status


def gradmatrix(n: int, kanten) -> csr_matrix:
    """Je Ort eine Zeile mit einer 1 an jeder Kante dieses Orts."""
    zeilen = [v for e in kanten for v in e]
    spalten = [j for j, _ in enumerate(kanten) for _ in (0, 1)]
    return csr_matrix((np.ones(len(zeilen)), (zeilen, spalten)), shape=(n, len(kanten)))


def schnittkanten(kanten, menge) -> list:
    """Nummern der Kanten, die die Menge verlassen (δ(S))."""
    return [j for j, (u, v) in enumerate(kanten) if (u in menge) != (v in menge)]


class Modell:
    """LP mit Gradgleichungen und einer wachsenden Liste von Zeilen a·x >= b (dünn gespeichert). Jede Zeile hat einen Schlüssel,
    doppelte werden nicht aufgenommen."""

    def __init__(self, n: int, kanten, kosten):
        self.n, self.kanten = n, list(kanten)
        self.kosten = np.asarray(kosten, float)
        self.grad = gradmatrix(n, self.kanten)
        self.spalten, self.werte, self.grenzen, self.schluessel = [], [], [], {}
        self.wert, self.x = None, None
        self.dual_grad, self.dual_zeilen = None, None

    @property
    def zeilen(self) -> int:
        return len(self.grenzen)

    def add(self, schluessel, spalten, werte, grenze: float) -> bool:
        """Zeile Σ werte·x[spalten] >= grenze aufnehmen; False, wenn der Schlüssel schon vorhanden ist."""
        if schluessel in self.schluessel:
            return False
        self.schluessel[schluessel] = len(self.grenzen)
        self.spalten.append(list(spalten))
        self.werte.append(list(werte))
        self.grenzen.append(float(grenze))
        return True

    def matrix(self) -> csr_matrix:
        """Alle nachgeladenen Zeilen als dünne Matrix (Zeilen × Kanten)."""
        zeilen = [i for i, s in enumerate(self.spalten) for _ in s]
        spalten = [j for s in self.spalten for j in s]
        werte = [w for s in self.werte for w in s]
        return csr_matrix((werte, (zeilen, spalten)), shape=(self.zeilen, len(self.kanten)))

    def kopie(self) -> "Modell":
        """Unabhängige Kopie mit denselben Zeilen und derselben Lösung."""
        neu = Modell.__new__(Modell)
        neu.__dict__.update(self.__dict__)
        neu.spalten, neu.werte, neu.grenzen = list(self.spalten), list(self.werte), list(self.grenzen)
        neu.schluessel = dict(self.schluessel)
        neu.x = None if self.x is None else self.x.copy()
        return neu

    def loese(self, unten=None, oben=None):
        """Löst das LP; liefert (Wert, x). `unten`/`oben`: eigene Schranken je Kante (sonst 0 und 1). Die Dualwerte bleiben gespeichert."""
        A_ub, b_ub = None, None
        if self.zeilen:
            A_ub, b_ub = -self.matrix(), -np.array(self.grenzen)
        unten = np.zeros(len(self.kanten)) if unten is None else np.asarray(unten, float)
        oben = np.ones(len(self.kanten)) if oben is None else np.asarray(oben, float)
        res = linprog(self.kosten, A_eq=self.grad, b_eq=np.full(self.n, 2.0), A_ub=A_ub, b_ub=b_ub,
                      bounds=np.column_stack([unten, oben]), method="highs-ds")
        if res.status != 0:
            raise LPFehler(res.message, res.status)
        self.wert, self.x = float(res.fun), np.clip(res.x, 0.0, 1.0)
        self.dual_grad = np.array(res.eqlin.marginals, float)
        self.dual_zeilen = -np.array(res.ineqlin.marginals, float) if self.zeilen else np.zeros(0)
        return self.wert, self.x

    def dualschranke(self, unten=None, oben=None) -> float:
        """Untere Schranke durch schwache Dualität, selbst nachgerechnet: Für beliebige Werte π der Gradgleichungen und beliebige y >= 0 der
        Zeilen G·x >= g gilt c·x >= 2·Σπ + g·y + Σ_e min(r_e·unten_e, r_e·oben_e) mit r = c − Aᵀπ − Gᵀy. Das hängt nicht davon ab, ob der
        Löser richtig gerechnet hat; benutzt werden nur seine Dualwerte (negative y werden auf 0 gesetzt). Voraussetzung: die Zeilen sind für die
        betrachteten Punkte gültig. Gerechnet in Gleitkommazahlen."""
        unten = np.zeros(len(self.kanten)) if unten is None else np.asarray(unten, float)
        oben = np.ones(len(self.kanten)) if oben is None else np.asarray(oben, float)
        pi = self.dual_grad
        y = np.maximum(self.dual_zeilen, 0.0)
        r = self.kosten - self.grad.T @ pi
        schranke = 2.0 * float(pi.sum())
        if len(y):
            r = r - self.matrix()[: len(y)].T @ y
            schranke += float(np.dot(self.grenzen[: len(y)], y))
        return schranke + float(np.minimum(r * unten, r * oben).sum())


def ist_ganzzahlig(x, tol: float = TOL_GANZ) -> bool:
    x = np.asarray(x, float)
    return bool(np.all(np.abs(x - np.round(x)) < tol))


def ist_halbzahlig(x, tol: float = TOL_GANZ) -> bool:
    """Nur die Werte 0, 1/2 und 1."""
    x = np.asarray(x, float)
    return bool(np.all(np.abs(2 * x - np.round(2 * x)) < 2 * tol))


def als_bruch(wert: float, nenner: int = 200, tol: float = TOL_GANZ):
    """Der Bruch mit Nenner <= `nenner`, der `wert` bis auf `tol` trifft, sonst None."""
    f = Fraction(float(wert)).limit_denominator(nenner)
    return f if abs(float(f) - float(wert)) < tol else None


def gradsummen(n: int, kanten, x) -> list:
    summe = [0.0] * n
    for (u, v), w in zip(kanten, x):
        summe[u] += w
        summe[v] += w
    return summe


def struktur(n: int, kanten, x):
    """Wie sieht eine LP-Lösung aus? Rückgabe (art, daten):
    ('ganzzahlig', None); ('halb', (nur_kreise, Kreislängen)) bei Werten aus {0, 1/2, 1}: bilden die 1/2-Kanten disjunkte Kreise?;
    ('anders', Menge der Nenner) bei anderen Brüchen; ('unbekannt', None), wenn ein Wert kein Bruch mit kleinem Nenner ist."""
    if ist_ganzzahlig(x):
        return "ganzzahlig", None
    brueche = [als_bruch(v) for v in x]
    if any(f is None for f in brueche):
        return "unbekannt", None
    if any(f.denominator > 2 for f in brueche):
        return "anders", tuple(sorted({f.denominator for f in brueche if f.denominator > 2}))
    nachbarn = {}
    for (u, v), f in zip(kanten, brueche):
        if f.denominator == 2:
            nachbarn.setdefault(u, []).append(v)
            nachbarn.setdefault(v, []).append(u)
    gesehen, laengen, nur_kreise = set(), [], True
    for start in sorted(nachbarn):
        if start in gesehen:
            continue
        gesehen.add(start)
        stapel, teil = [start], []
        while stapel:
            a = stapel.pop()
            teil.append(a)
            for b in nachbarn[a]:
                if b not in gesehen:
                    gesehen.add(b)
                    stapel.append(b)
        nur_kreise &= all(len(nachbarn[a]) == 2 for a in teil)
        laengen.append(len(teil))
    return "halb", (nur_kreise, tuple(sorted(laengen)))
