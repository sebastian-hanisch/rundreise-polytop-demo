"""Das ganzzahlige Optimum (die kürzeste Rundreise) und woran man es glaubt.

Gefunden wird die Rundreise mit HiGHS-MILP (scipy.optimize.milp) mit nachgeladenen Subtourzeilen und harter Zeitgrenze. Ein MILP-Status allein
ist kein Beleg (scipy.optimize.milp kann „optimal“ melden und danebenliegen). Deshalb wird jedes Ergebnis geprüft:

  1. unabhängiger Prüfer: es ist eine Rundreise, und ihre Länge ist die gemeldete;
  2. Dualschranke: die Kosten sind ganzzahlig. Erreicht eine selbst nachgerechnete Dualschranke des LP (Modell.dualschranke, schwache
     Dualität) einen Wert über L − 1, kostet jede Rundreise mindestens L, und die gefundene mit Länge L ist optimal;
  3. sonst bis 13 Orte eine Dynamik über Teilmengen (Held und Karp);
  4. sonst ein eigenes Branch-and-Bound mit denselben nachgerechneten Dualschranken (Subtourzeilen werden in den Knoten nachgeladen), mit
     Zeitgrenze; findet es eine kürzere Rundreise, war das MILP falsch;
  5. wenn keiner der Belege gelingt: Plausibilitätsprüfung mit 2-opt und Or-opt. Findet sie eine kürzere Rundreise oder liegt der LP-Wert über der
     gemeldeten Länge, ist das Optimum „nicht bestimmt“.

Die schon im LP nachgeladenen Zeilen (Subtour, 2-Matching, Kämme) sind für jede Rundreise gültig und werden dem MILP mitgegeben."""

import time
from dataclasses import dataclass
from math import ceil
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

import rpt_lp as LP
import rpt_subtour as ST

DUALSCHRANKE = "dualschranke"       # die LP-Dualschranke erreicht den Wert
HELD_KARP = "held-karp"             # Dynamik über Teilmengen liefert denselben Wert
ENUMERATION = "branch-and-bound"    # eigenes Branch-and-Bound mit nachgerechneten Dualschranken
NUR_MILP = "milp"                   # nur HiGHS-MILP; Plausibilitätsprüfung bestanden
UNBESTIMMT = "nicht bestimmt"
BELEGT = (DUALSCHRANKE, HELD_KARP, ENUMERATION)

HELD_KARP_BIS = 13
EPS = 1e-6


@dataclass(frozen=True)
class Optimum:
    status: str
    wert: Optional[int] = None
    tour: tuple = ()           # Kantennummern der Rundreise
    grund: str = ""            # bei UNBESTIMMT: warum
    knoten: int = 0            # Knoten des eigenen Branch-and-Bound

    @property
    def belegt(self) -> bool:
        return self.status in BELEGT


def ist_rundreise(n: int, kanten, nummern) -> bool:
    """Unabhängiger Prüfer: n Kanten, jeder Ort hat Grad 2, alles hängt zusammen."""
    grad = [0] * n
    for j in nummern:
        u, v = kanten[j]
        grad[u] += 1
        grad[v] += 1
    x = [1.0 if j in set(nummern) else 0.0 for j in range(len(kanten))]
    return len(set(nummern)) == n and all(g == 2 for g in grad) and len(ST.komponenten(n, kanten, x, 0.5)) == 1


def tourlaenge(kosten, nummern) -> int:
    return int(sum(kosten[j] for j in nummern))


def schranke_belegt(schranke: float, laenge: int) -> bool:
    """Bei ganzzahligen Kosten: eine Dualschranke über L − 1 zeigt, dass jede Rundreise mindestens L kostet."""
    return schranke > laenge - 1 + EPS


def held_karp(n: int, kanten, kosten) -> int:
    """Kleinste Rundreise durch Dynamik über Teilmengen (ohne LP und ohne MILP)."""
    d = np.full((n, n), np.inf)
    for (u, v), w in zip(kanten, kosten):
        d[u, v] = d[v, u] = w
    m = n - 1
    best = np.full((1 << m, m), np.inf)
    for j in range(m):
        best[1 << j, j] = d[0, j + 1]
    innen = d[1:, 1:]
    for maske in range(1, 1 << m):
        zeile_ = best[maske]
        if not np.isfinite(zeile_).any():
            continue
        weiter = (zeile_[:, None] + innen).min(0)
        for j in range(m):
            if not maske >> j & 1:
                neu = maske | 1 << j
                if weiter[j] < best[neu, j]:
                    best[neu, j] = weiter[j]
    return int(round((best[-1] + d[1:, 0]).min()))


# ---------------------------------------------------------------- Plausibilität: 2-opt und Or-opt

def _matrix(n: int, kanten, kosten) -> np.ndarray:
    d = np.zeros((n, n))
    for (u, v), w in zip(kanten, kosten):
        d[u, v] = d[v, u] = w
    return d


def ordnung(n: int, kanten, nummern) -> list:
    """Reihenfolge der Orte einer Rundreise (als Kantennummern gegeben)."""
    nachbarn = [[] for _ in range(n)]
    for j in nummern:
        u, v = kanten[j]
        nachbarn[u].append(v)
        nachbarn[v].append(u)
    folge, vorher = [0], None
    while len(folge) < n:
        a = folge[-1]
        weiter = nachbarn[a][0] if nachbarn[a][0] != vorher else nachbarn[a][1]
        vorher = a
        folge.append(weiter)
    return folge


def folgenlaenge(d: np.ndarray, folge) -> float:
    return float(sum(d[a, b] for a, b in zip(folge, folge[1:] + folge[:1])))


def verbessere(d: np.ndarray, folge: list) -> list:
    """2-opt (einen Abschnitt umdrehen) und Or-opt (einen Abschnitt aus 1 bis 3 Orten versetzen), bis keiner der beiden Züge mehr kürzt."""
    n = len(folge)
    besser = True
    while besser:
        besser = False
        for i in range(n - 1):
            for j in range(i + 2, n if i else n - 1):
                a, b, c, e = folge[i], folge[i + 1], folge[j], folge[(j + 1) % n]
                if d[a, c] + d[b, e] < d[a, b] + d[c, e] - EPS:
                    folge[i + 1: j + 1] = folge[i + 1: j + 1][::-1]
                    besser = True
        for laenge in (1, 2, 3):
            if n < laenge + 3:
                continue
            for i in range(n):
                gedreht = folge[i:] + folge[:i]
                abschnitt, rest = gedreht[:laenge], gedreht[laenge:]
                alt = folgenlaenge(d, gedreht)
                gewinn = d[rest[-1], abschnitt[0]] + d[abschnitt[-1], rest[0]] - d[rest[-1], rest[0]]
                for k in range(len(rest) - 1):
                    a, b = rest[k], rest[k + 1]
                    for s in (abschnitt, abschnitt[::-1]):
                        neu = d[a, s[0]] + d[s[-1], b] - d[a, b]
                        if neu < gewinn - EPS:
                            kandidat = rest[: k + 1] + s + rest[k + 1:]
                            if folgenlaenge(d, kandidat) < alt - EPS:
                                folge, besser = kandidat, True
                            break
                    if besser:
                        break
                if besser:
                    break
            if besser:
                break
    return folge


def heuristik_laenge(n: int, kanten, kosten, start: Optional[tuple] = None) -> int:
    """Beste Länge aus 2-opt/Or-opt, gestartet bei der gegebenen Rundreise und bei nächste-Nachbar-Touren von bis zu 4 Startorten."""
    d = _matrix(n, kanten, kosten)
    starts = [ordnung(n, kanten, start)] if start else []
    for s in range(0, n, max(1, n // 4)):
        folge, frei = [s], set(range(n)) - {s}
        while frei:
            a = folge[-1]
            b = min(sorted(frei), key=lambda v: d[a, v])
            folge.append(b)
            frei.remove(b)
        starts.append(folge)
    return int(round(min(folgenlaenge(d, verbessere(d, list(f))) for f in starts)))


# ---------------------------------------------------------------- eigenes Branch-and-Bound als zweiter Beleg

def zweitbeweis(modell: LP.Modell, laenge: int, zeitgrenze: float, max_knoten: int = 4000):
    """Zeigt mit nachgerechneten Dualschranken, dass keine Rundreise kürzer als `laenge` ist. Je Knoten: LP lösen, Subtourzeilen nachladen,
    abschneiden, wenn die Dualschranke über laenge − 1 liegt, sonst an der Kante mit dem Wert am nächsten bei 1/2 verzweigen.
    Rückgabe (Ergebnis, Kantennummern, Knoten): 'belegt', 'kuerzer' (mit der kürzeren Rundreise) oder 'offen' (Grenze erreicht).
    Ein Knoten, den der Löser als unzulässig meldet, gilt als erledigt."""
    ende = time.perf_counter() + zeitgrenze
    basis = modell.kopie()
    m = len(basis.kanten)
    stapel = [(np.zeros(m), np.ones(m))]
    knoten = 0
    while stapel:
        if knoten >= max_knoten or time.perf_counter() > ende:
            return "offen", (), knoten
        unten, oben = stapel.pop()
        knoten += 1
        while True:
            try:
                basis.loese(unten, oben)
            except LP.LPFehler as fehler:
                if fehler.status != LP.UNZULAESSIG:
                    return "offen", (), knoten             # anderer Löserfehler: nichts behaupten
                break                                      # unzulässig: keine Rundreise in diesem Knoten
            if schranke_belegt(basis.dualschranke(unten, oben), laenge):
                break
            neu = [s for _, s in ST.verletzte_subtours(basis.n, basis.kanten, basis.x)
                   if basis.add(("S", s), *ST.zeile(basis.kanten, s))]
            if neu:
                if time.perf_counter() > ende:
                    return "offen", (), knoten
                continue
            x = basis.x
            if LP.ist_ganzzahlig(x):
                nummern = tuple(int(j) for j in np.flatnonzero(x > 0.5))
                if ist_rundreise(basis.n, basis.kanten, nummern) and tourlaenge(basis.kosten, nummern) < laenge:
                    return "kuerzer", nummern, knoten
                return "offen", (), knoten                  # numerisch unklar: nichts behaupten
            j = int(np.argmin(np.abs(x - 0.5)))
            null_oben, eins_unten = oben.copy(), unten.copy()
            null_oben[j], eins_unten[j] = 0.0, 1.0
            stapel.append((unten, null_oben))
            stapel.append((eins_unten, oben))
            break
    return "belegt", (), knoten


# ---------------------------------------------------------------- Optimum

def milp_rundreise(n: int, kanten, kosten, modell: Optional[LP.Modell], zeitgrenze: float) -> Optional[tuple]:
    """Rundreise laut HiGHS-MILP als Kantennummern, None bei Zeitgrenze oder Fehler. Noch kein Beleg."""
    start = time.perf_counter()
    matrix = LP.gradmatrix(n, kanten)
    zusatz = []                       # (Spalten, Koeffizienten, Grenze)
    if modell is not None:
        zusatz = list(zip(modell.spalten, modell.werte, modell.grenzen))
    c = np.asarray(kosten, float)
    while True:
        rest = zeitgrenze - (time.perf_counter() - start)
        if rest <= 0:
            return None
        bedingungen = [LinearConstraint(matrix, 2, 2)]
        if zusatz:
            zeilen = [i for i, (s, _, _) in enumerate(zusatz) for _ in s]
            a = csr_matrix(([w for _, ws, _ in zusatz for w in ws], (zeilen, [j for s, _, _ in zusatz for j in s])),
                           shape=(len(zusatz), len(kanten)))
            bedingungen.append(LinearConstraint(a, np.array([g for _, _, g in zusatz]), np.inf))
        res = milp(c, constraints=bedingungen, integrality=np.ones(len(kanten)), bounds=Bounds(0, 1),
                   options={"mip_rel_gap": 0.0, "time_limit": max(rest, 0.05)})
        if res.status != 0 or res.x is None:
            return None
        x = np.round(res.x)
        teile = ST.komponenten(n, kanten, x, 0.5)
        if len(teile) == 1:
            return tuple(int(j) for j in np.flatnonzero(x > 0.5))
        zusatz += [ST.zeile(kanten, t) for t in teile]


def optimum(n: int, kanten, kosten, modell: Optional[LP.Modell] = None, zeitgrenze: float = 5.0,
            zeitgrenze_beleg: float = 1.0) -> Optimum:
    """Kürzeste Rundreise mit Angabe des Belegs (siehe Modulkopf). `modell`: ein gelöstes LP mit gültigen Zeilen oder None."""
    kosten_ganz = [int(k) for k in kosten]
    if modell is not None and modell.x is not None and LP.ist_ganzzahlig(modell.x):
        nummern = tuple(int(j) for j in np.flatnonzero(modell.x > 0.5))
        if ist_rundreise(n, kanten, nummern):
            laenge = tourlaenge(kosten_ganz, nummern)
            if schranke_belegt(modell.dualschranke(), laenge):
                return Optimum(DUALSCHRANKE, laenge, nummern)
    nummern = milp_rundreise(n, kanten, kosten_ganz, modell, zeitgrenze)
    if nummern is None:
        return Optimum(UNBESTIMMT, grund="Zeitgrenze")
    if not ist_rundreise(n, kanten, nummern):
        return Optimum(UNBESTIMMT, grund="das MILP lieferte keine Rundreise")
    laenge = tourlaenge(kosten_ganz, nummern)
    if modell is None:
        modell = LP.Modell(n, kanten, kosten_ganz)
        modell.loese()
    if modell.wert > laenge + 1e-4:
        return Optimum(UNBESTIMMT, grund="der LP-Wert liegt über der gemeldeten Länge")
    if schranke_belegt(modell.dualschranke(), laenge):
        return Optimum(DUALSCHRANKE, laenge, nummern)
    if n <= HELD_KARP_BIS:
        zweit = held_karp(n, kanten, kosten_ganz)
        if zweit != laenge:
            return Optimum(UNBESTIMMT, grund=f"die Dynamik über Teilmengen findet {zweit}, das MILP {laenge}")
        return Optimum(HELD_KARP, laenge, nummern)
    ergebnis, kuerzer, knoten = zweitbeweis(modell, laenge, zeitgrenze_beleg)
    if ergebnis == "belegt":
        return Optimum(ENUMERATION, laenge, nummern, knoten=knoten)
    if ergebnis == "kuerzer":
        return Optimum(UNBESTIMMT, grund=f"das eigene Branch-and-Bound findet eine kürzere Rundreise ({tourlaenge(kosten_ganz, kuerzer)})",
                       knoten=knoten)
    plausibel = heuristik_laenge(n, kanten, kosten_ganz, nummern)
    if plausibel < laenge:
        return Optimum(UNBESTIMMT, grund=f"2-opt findet eine kürzere Rundreise ({plausibel})", knoten=knoten)
    return Optimum(NUR_MILP, laenge, nummern, knoten=knoten)


def runde_hoch(schranke: float) -> int:
    """Kleinste ganze Zahl, die die Schranke (bis auf die Rechentoleranz) nicht unterschreitet."""
    return ceil(schranke - EPS)
