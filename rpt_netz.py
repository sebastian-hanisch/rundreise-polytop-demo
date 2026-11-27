"""Das Permutaeder mit Hilfsvariablen aus einem Komparatornetz (die Idee, aus Sortiernetzen eine erweiterte Formulierung zu bauen, stammt von
Goemans).

Permutaeder: konvexe Hülle aller Anordnungen von (1, …, n); nach Rado ist es die Menge mit Σ z = n(n+1)/2 und z(S) >= |S|(|S|+1)/2 für jede
echte nichtleere Teilmenge S, also 2^n − 2 Facetten.

Formulierung: Die Eingänge des Netzes sind z. Je Komparator mit Eingängen a, b gibt es neue Werte lo, hi mit a + b = lo + hi, lo <= a und
lo <= b; die Ausgänge am Ende sind fest 1, …, n und werden eingesetzt. Je Komparator also zwei Ungleichungen, eine Gleichung und nach dem
Einsetzen zusammen 2·(Zahl der Komparatoren) freie Variablen.

Satz (eigene Herleitung, nicht extern geprüft; die Tests prüfen ihn über alle Netze mit einem gestrichenen Komparator): Für jedes
Komparatornetz liegt die Projektion auf z im Permutaeder, und eine Permutation w liegt genau dann in der Projektion, wenn das Netz w sortiert.
Beweis: Aus lo <= a, lo <= b und lo + hi = a + b folgt hi >= max(a, b); (a, b) wird also von (lo, hi) majorisiert und z von (1, …, n), nach
Rado liegt z im Permutaeder. Sortiert das Netz w, ist sein Durchlauf selbst eine Belegung. Liegt umgekehrt eine Permutation w in der Projektion,
kann die Quadratsumme wegen lo <= min(a, b) und lo + hi = a + b an keinem Komparator fallen. Am Anfang und am Ende ist sie gleich, also wächst
sie auch an keinem; damit ist jedes Mal {lo, hi} = {a, b} mit lo = min, und der Durchlauf endet in der sortierten Folge."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

BATCHER, EINFUEGEN = "batcher", "einfuegen"
NETZE = (BATCHER, EINFUEGEN)


def einfuege_netz(n: int) -> list:
    """Sortieren durch Einfügen als Netz: n(n−1)/2 Komparatoren benachbarter Drähte."""
    return [(j - 1, j) for i in range(1, n) for j in range(i, 0, -1)]


def batcher_netz(n: int) -> list:
    """Batchers Merge-Exchange-Netz; dass es sortiert, prüfen die Tests mit dem 0-1-Prinzip für n = 2 bis 16."""
    netz = []
    if n < 2:
        return netz
    t = (n - 1).bit_length()
    p = 1 << (t - 1)
    while p > 0:
        q, r, d = 1 << (t - 1), 0, p
        while True:
            netz += [(i, i + d) for i in range(n - d) if i & p == r]
            if q == p:
                break
            d, q, r = q - p, q >> 1, p
        p >>= 1
    return netz


def netz(art: str, n: int) -> list:
    return batcher_netz(n) if art == BATCHER else einfuege_netz(n)


def lauf(komparatoren, folge) -> list:
    """Der Durchlauf: jeder Komparator (i, j) legt das kleinere Element auf Draht i."""
    w = list(folge)
    for i, j in komparatoren:
        if w[i] > w[j]:
            w[i], w[j] = w[j], w[i]
    return w


def sortiert_alles(komparatoren, n: int) -> bool:
    """0-1-Prinzip: ein Komparatornetz sortiert alle Folgen genau dann, wenn es alle 2^n 0/1-Folgen sortiert."""
    m = np.arange(1 << n)
    bits = ((m[:, None] >> np.arange(n)) & 1).astype(np.int8)
    for i, j in komparatoren:
        klein = np.minimum(bits[:, i], bits[:, j])
        gross = np.maximum(bits[:, i], bits[:, j])
        bits[:, i], bits[:, j] = klein, gross
    return bool(np.all(np.diff(bits, axis=1) >= 0))


def sortierte_permutationen(komparatoren, n: int) -> tuple:
    """(die Permutationen von 1..n, die das Netz sortiert, alle übrigen); nur für kleine n."""
    ziel = list(range(1, n + 1))
    gut, schlecht = [], []
    for w in permutations(ziel):
        (gut if lauf(komparatoren, w) == ziel else schlecht).append(w)
    return gut, schlecht


@dataclass(frozen=True)
class Formulierung:
    """LP-Daten mit eingesetzten Ausgängen. Variablen: die freien Drahtwerte; `eingang` je Position (Variablennummer oder fester Wert)."""
    variablen: int
    A_eq: object
    b_eq: np.ndarray
    A_ub: object
    b_ub: np.ndarray
    eingang: tuple                 # je Position ('v', Nummer) oder ('c', Wert)

    @property
    def ungleichungen(self) -> int:
        return len(self.b_ub)

    @property
    def gleichungen(self) -> int:
        return len(self.b_eq)


def add_komparator(modell: dict, a: int, b: int):
    """Hängt an den Drähten a, b (Wertnummern) einen Komparator an: neue Nummern (lo, hi), eine Gleichung, zwei Ungleichungen."""
    lo, hi = modell["werte"], modell["werte"] + 1
    modell["werte"] += 2
    modell["eq"].append(({a: 1, b: 1, lo: -1, hi: -1}, 0))
    modell["ub"].append(({lo: 1, a: -1}, 0))
    modell["ub"].append(({lo: 1, b: -1}, 0))
    return lo, hi


def netz_formulierung(komparatoren, n: int) -> Formulierung:
    """Komposition: je Komparator `add_komparator`, am Ende die Ausgänge 1..n einsetzen."""
    modell = {"werte": n, "eq": [], "ub": []}
    draht = list(range(n))
    for i, j in komparatoren:
        draht[i], draht[j] = add_komparator(modell, draht[i], draht[j])
    fest = {draht[pos]: pos + 1 for pos in range(n)}
    nummer, frei = {}, 0
    for w in range(modell["werte"]):
        if w not in fest:
            nummer[w] = frei
            frei += 1

    def duenn(zeilen):
        r, c, d, b = [], [], [], []
        for k, (koeffizienten, rechts) in enumerate(zeilen):
            for w, v in koeffizienten.items():
                if w in fest:
                    rechts -= v * fest[w]
                else:
                    r.append(k)
                    c.append(nummer[w])
                    d.append(v)
            b.append(rechts)
        return csr_matrix((d, (r, c)), shape=(len(zeilen), frei)), np.array(b, float)

    A_eq, b_eq = duenn(modell["eq"])
    A_ub, b_ub = duenn(modell["ub"])
    eingang = tuple(("c", fest[w]) if w in fest else ("v", nummer[w]) for w in range(n))
    return Formulierung(frei, A_eq, b_eq, A_ub, b_ub, eingang)


def netz_lp(komparatoren, n: int, ziel) -> tuple:
    """Kleinstes ziel·z über der Projektion: (Wert, z)."""
    f = netz_formulierung(komparatoren, n)
    c = np.zeros(f.variablen)
    konstante = 0.0
    for pos, (art, w) in enumerate(f.eingang):
        if art == "v":
            c[w] += ziel[pos]
        else:
            konstante += ziel[pos] * w
    if f.variablen == 0:
        z = [float(w) for _, w in f.eingang]
        return konstante, z
    res = linprog(c, A_ub=f.A_ub if f.ungleichungen else None, b_ub=f.b_ub if f.ungleichungen else None,
                  A_eq=f.A_eq if f.gleichungen else None, b_eq=f.b_eq if f.gleichungen else None,
                  bounds=[(None, None)] * f.variablen, method="highs-ds")
    if res.status != 0:
        raise RuntimeError(res.message)
    z = [float(res.x[w]) if art == "v" else float(w) for art, w in f.eingang]
    return float(res.fun + konstante), z


def sortier_optimum(ziel) -> tuple:
    """Kleinstes ziel·z über dem ganzen Permutaeder, nur durch Sortieren: der größte Koeffizient bekommt die 1. (Wert, z)."""
    ordnung = sorted(range(len(ziel)), key=lambda i: (-ziel[i], i))
    z = [0] * len(ziel)
    for wert, i in enumerate(ordnung, start=1):
        z[i] = wert
    return float(sum(c * v for c, v in zip(ziel, z))), z


def permutaeder_facetten(n: int) -> int:
    """2^n − 2 Facetten (für n >= 2)."""
    return 2 ** n - 2


# ---------------------------------------------------------------- Zahlen zum Spannbaum nach Martin (nur Zählung)

def spannbaum_facetten(n: int) -> int:
    """Facetten des Spannbaumpolytops des vollständigen Graphen K_n für n >= 4: x_e >= 0 und x(E[S]) <= |S| − 1: m + 2^n − n − 2."""
    return n * (n - 1) // 2 + 2 ** n - n - 2


def martin_groessen(n: int) -> dict:
    """Größe von Martins Formulierung für K_n: Richtungswerte y_k,ij >= 0 je Wurzel k ohne die Kanten, die aus k hinausführen würden.
    Rückgabe: Ungleichungen (y >= 0), Variablen (x und y), Gleichungen."""
    m = n * (n - 1) // 2
    return {"ungleichungen": n * (n - 1) ** 2, "variablen": m + n * (n - 1) ** 2, "gleichungen": 1 + n * (m + n - 1)}
