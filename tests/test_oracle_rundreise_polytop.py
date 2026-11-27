"""Orakel: jede Kernroutine gegen einen anderen Rechenweg (Aufzählen aller Ungleichungen, aller Rundreisen, aller Zahnfamilien, networkx).

Die Orakel benutzen aus der Demo nur Instanzen und die Paarliste, außer dort, wo die geprüfte Funktion selbst aufgerufen wird."""

from itertools import combinations, permutations

import numpy as np
import pytest
from scipy.optimize import linprog

import rpt_huelle as H
import rpt_instanzen as I
import rpt_kamm as K
import rpt_leiter as L
import rpt_lp as LP
import rpt_netz as NZ
import rpt_referenz as R
import rpt_subtour as ST
import rpt_zweimatching as ZM
from rpt_rng import SplitMix64


def alle_touren(n):
    kanten = I.paare(n)
    nummer = {e: j for j, e in enumerate(kanten)}
    out = []
    for p in permutations(range(1, n)):
        if p[0] > p[-1]:
            continue
        weg = (0,) + p + (0,)
        x = np.zeros(len(kanten))
        for a, b in zip(weg, weg[1:]):
            x[nummer[tuple(sorted((a, b)))]] = 1
        out.append(x)
    return np.array(out)


def rand(kanten, menge):
    return np.array([1.0 if (u in menge) != (v in menge) else 0.0 for u, v in kanten])


def lp_mit_zeilen(inst, zeilen_a, zeilen_b):
    """min c·x mit Gradgleichungen, 0 <= x <= 1 und zeilen_a·x >= zeilen_b, alle auf einmal."""
    kanten = inst.kanten
    grad = np.array([[1.0 if v in e else 0.0 for e in kanten] for v in range(inst.n)])
    res = linprog(inst.kosten, A_eq=grad, b_eq=[2] * inst.n, A_ub=-np.array(zeilen_a) if len(zeilen_a) else None,
                  b_ub=-np.array(zeilen_b) if len(zeilen_b) else None, bounds=(0, 1), method="highs")
    assert res.status == 0
    return res.fun


def alle_subtourzeilen(n):
    kanten = I.paare(n)
    zeilen = [rand(kanten, set(s)) for k in range(2, n // 2 + 1) for s in combinations(range(n), k)]
    return zeilen, [2.0] * len(zeilen)


def alle_zweimatchingzeilen(n):
    """Alle Griffe und alle ungeraden Teilmengen F von δ(H)."""
    kanten = I.paare(n)
    zeilen, rechts = [], []
    for k in range(1, n):
        for h in combinations(range(n), k):
            if 0 in h:
                continue
            d = [j for j, (u, v) in enumerate(kanten) if (u in h) != (v in h)]
            for g in range(1, len(d) + 1, 2):
                for f in combinations(d, g):
                    z = np.zeros(len(kanten))
                    for j in d:
                        z[j] = -1.0 if j in f else 1.0
                    zeilen.append(z)
                    rechts.append(1.0 - len(f))
    return zeilen, rechts


def instanzen(n, anzahl, familien=("karte", "matrix", "haufen")):
    for fam in familien:
        for s in range(anzahl):
            yield I.zufallsinstanz(fam, n, 500 + s)


@pytest.mark.parametrize("n", [5, 6, 7])
def test_subtourstufe_gleich_lp_mit_allen_subtourzeilen(n):
    a, b = alle_subtourzeilen(n)
    for inst in instanzen(n, 5):
        lt = L.leiter(inst)
        assert lt.wert_subtour == pytest.approx(lp_mit_zeilen(inst, a, b), abs=1e-6)


def test_exakte_2matching_stufe_gleich_lp_mit_allen_ungleichungen_bei_sechs_orten():
    sa, sb = alle_subtourzeilen(6)
    za, zb = alle_zweimatchingzeilen(6)
    assert len(za) > 1000
    anders = 0
    for inst in list(instanzen(6, 8, ("matrix", "karte"))) + [I.prisma(1, 2, 1), I.prisma(1, 3, 1), I.prisma(1, 5, 1)]:
        lt = L.leiter(inst)
        voll = lp_mit_zeilen(inst, sa + za, sb + zb)
        assert lt.wert_zweimatching == pytest.approx(voll, abs=1e-6)
        anders += lt.wert_zweimatching > lt.wert_subtour + 1e-6
    assert anders >= 3          # sonst prüft der Test die Stufe nicht


def test_padberg_rao_gegen_aufzaehlen_aller_griffe_auf_gebrochenen_punkten():
    """An LP-Ecken nach den Subtourzeilen: beide finden Verletzung oder keiner; die kleinsten Werte stimmen überein."""
    verletzt = geprueft = 0
    for n in (10, 12, 14):
        for inst in instanzen(n, 40, ("matrix",)):
            m = L.neues_modell(inst)
            L.grad_stufe(m)
            L.subtour_stufe(m)
            if LP.ist_ganzzahlig(m.x):
                continue
            geprueft += 1
            bestes, _ = ZM.staerkste_durch_aufzaehlen(n, inst.kanten, m.x)
            gefunden = ZM.trenne_padberg_rao(n, inst.kanten, m.x)
            assert bool(gefunden) == (bestes < 1 - 1e-6), inst
            if gefunden:
                verletzt += 1
                assert gefunden[0].wert == pytest.approx(bestes, abs=1e-6)
                for z in gefunden:
                    assert len(z.zaehne) % 2 == 1 and set(z.zaehne) <= set(LP.schnittkanten(inst.kanten, z.griff))
    assert geprueft >= 15 and verletzt >= 3


def test_optimum_gegen_alle_rundreisen():
    for n in (6, 7, 8):
        touren = alle_touren(n)
        for inst in instanzen(n, 6):
            soll = int(round((touren @ np.array(inst.kosten)).min()))
            lt = L.leiter(inst)
            for modell in (None, lt.modell_zm):
                o = R.optimum(inst.n, inst.kanten, inst.kosten, modell, 30.0)
                assert o.wert == soll
            assert lt.wert_zweimatching <= soll + 1e-6


def _zahnfamilien_zu_dritt(n):
    mengen = [m for m in range(1 << n) if bin(m).count("1") >= 2]
    familien = []
    for i, a in enumerate(mengen):
        for j in range(i + 1, len(mengen)):
            b = mengen[j]
            if a & b:
                continue
            for k in range(j + 1, len(mengen)):
                if not (a | b) & mengen[k]:
                    familien.append((a, b, mengen[k]))
    return np.array(familien)


def _staerkste_verletzung_aller_kaemme(n, x):
    """Alle Griffe und alle Familien aus 3 disjunkten Zähnen (n <= 7 reicht für drei Zähne aus mindestens 2 Orten)."""
    kanten = I.paare(n)
    c = np.array([float(rand(kanten, {i for i in range(n) if m >> i & 1}) @ x) for m in range(1 << n)])
    familien = _zahnfamilien_zu_dritt(n)
    voll = (1 << n) - 1
    beste = 0.0
    for h in range(1, 1 << (n - 1)):
        gueltig = (((familien & h) != 0) & ((familien & ~h & voll) != 0)).all(1)
        if gueltig.any():
            beste = max(beste, 10 - c[h] - c[familien[gueltig]].sum(1).min())
    return beste


def test_kammsuche_gegen_aufzaehlung_aller_zahnfamilien():
    punkte = verletzt = 0
    for i in range(300):
        inst = I.zufallsinstanz("matrix", 7, 2000 + i)
        kosten = [k // 251 + 1 for k in inst.kosten] if i % 2 else inst.kosten
        m = LP.Modell(7, inst.kanten, kosten)
        m.loese()
        L.subtour_stufe(m)
        if LP.ist_ganzzahlig(m.x):
            continue
        punkte += 1
        soll = _staerkste_verletzung_aller_kaemme(7, m.x)
        gefunden = K.trenne_kamm(7, inst.kanten, m.x)
        if soll > 1e-5:
            verletzt += 1
            assert gefunden is not None and gefunden[0] == pytest.approx(soll, abs=1e-5)
            assert K.ist_kamm(7, gefunden[1], gefunden[2])
        else:
            assert gefunden is None
        if punkte >= 20:
            break
    assert punkte >= 10 and verletzt >= 5


def test_an_rundreisen_wird_kein_kamm_gemeldet():
    kanten = I.paare(9)
    rng = SplitMix64(9)
    for _ in range(40):
        ordnung = list(range(9))
        rng.shuffle(ordnung)
        x = [0.0] * len(kanten)
        for a, b in zip(ordnung, ordnung[1:] + ordnung[:1]):
            x[I.kantennummer(9, a, b)] = 1.0
        assert K.trenne_kamm(9, kanten, x) is None
        assert ST.verletzte_subtours(9, kanten, x) == [] and ZM.trenne_padberg_rao(9, kanten, x) == []


def test_ganze_huelle_gibt_fuer_zufaellige_ziele_die_kuerzeste_rundreise():
    for n in (5, 6):
        huelle, punkte = H.huelle_rundreisen(n)
        kanten = I.paare(n)
        grad = np.array([[1.0 if v in e else 0.0 for e in kanten] for v in range(n)])
        a = np.array([f for f, _ in huelle.facetten], float)
        b = np.array([g for _, g in huelle.facetten], float)
        touren = alle_touren(n)
        assert len(punkte) == len(touren)
        rng = SplitMix64(40 + n)
        for _ in range(60):
            ziel = np.array([rng.below(41) - 20 for _ in kanten], float)
            res = linprog(ziel, A_ub=a, b_ub=b, A_eq=grad, b_eq=[2] * n, bounds=(None, None), method="highs")
            assert res.status == 0 and res.fun == pytest.approx((touren @ ziel).min(), abs=1e-6)


def test_bei_sechs_orten_reichen_die_kaemme_immer():
    """Die Hülle bei 6 Orten besteht aus Schranken, Subtourzeilen und 2-Matching-Ungleichungen: die Leiter endet immer am Optimum."""
    touren = alle_touren(6)
    for i in range(60):
        inst = I.zufallsinstanz("matrix", 6, 3000 + i)
        assert L.leiter(inst).wert_zweimatching == pytest.approx((touren @ np.array(inst.kosten)).min(), abs=1e-6)


def test_mindestschnitt_gegen_networkx():
    nx = pytest.importorskip("networkx")
    verletzt = 0
    for i in range(120):
        inst = I.zufallsinstanz("matrix", 14, 100 + i)
        m = L.neues_modell(inst)
        L.grad_stufe(m)
        for runde in range(3):
            g = nx.Graph()
            g.add_nodes_from(range(14))
            g.add_weighted_edges_from((u, v, w) for (u, v), w in zip(inst.kanten, m.x) if w > 1e-9)
            kleinster = 0.0 if not nx.is_connected(g) else nx.stoer_wagner(g)[0]
            gefunden = ST.verletzte_subtours(14, inst.kanten, m.x)
            assert bool(gefunden) == (kleinster < 2 - 1e-6)
            if not gefunden:
                break
            verletzt += 1
            assert gefunden[0][0] == pytest.approx(kleinster, abs=1e-6) or kleinster == 0.0
            m.add(("S", gefunden[0][1]), *ST.zeile(inst.kanten, gefunden[0][1]))
            m.loese()
    assert verletzt > 50


def test_netz_lp_gleich_sortieren_auf_zufaelligen_zielen():
    rng = SplitMix64(7)
    for art in NZ.NETZE:
        for n in range(1, 11):
            komparatoren = NZ.netz(art, n)
            for _ in range(4):
                ziel = [rng.below(201) - 100 for _ in range(n)]
                wert, z = NZ.netz_lp(komparatoren, n, ziel)
                assert wert == pytest.approx(NZ.sortier_optimum(ziel)[0], abs=1e-6)


def _in_projektion(komparatoren, n, w):
    """LP-Zulässigkeit mit festem z = w: ein anderer Rechenweg als der Durchlauf des Netzes."""
    f = NZ.netz_formulierung(komparatoren, n)
    schranken = [(None, None)] * f.variablen
    for pos, (art, wert) in enumerate(f.eingang):
        if art == "c":
            if wert != w[pos]:
                return False
        else:
            schranken[wert] = (w[pos], w[pos])
    if f.variablen == 0:
        return True
    res = linprog(np.zeros(f.variablen), A_ub=f.A_ub if f.ungleichungen else None, b_ub=f.b_ub if f.ungleichungen else None,
                  A_eq=f.A_eq if f.gleichungen else None, b_eq=f.b_eq if f.gleichungen else None, bounds=schranken, method="highs")
    return res.status == 0


@pytest.mark.parametrize("art", NZ.NETZE)
def test_satz_genau_die_sortierten_permutationen_liegen_in_der_projektion_bei_vier_werten(art):
    voll = NZ.netz(art, 4)
    varianten = [voll] + [voll[:i] + voll[i + 1:] for i in range(len(voll))]
    for komparatoren in varianten:
        for w in permutations((1, 2, 3, 4)):
            assert _in_projektion(komparatoren, 4, w) == (NZ.lauf(komparatoren, w) == [1, 2, 3, 4]), (komparatoren, w)


def test_projektion_liegt_fuer_zufaellige_netze_im_permutaeder():
    rng = SplitMix64(12)
    for _ in range(40):
        n = 4
        komparatoren = []
        for _ in range(rng.below(5)):
            i, j = sorted((rng.below(n), rng.below(n)))
            if i != j:
                komparatoren.append((i, j))
        ziel = [rng.below(101) - 50 for _ in range(n)]
        wert, z = NZ.netz_lp(komparatoren, n, ziel)
        assert wert >= NZ.sortier_optimum(ziel)[0] - 1e-6                          # nie unter dem Permutaeder: nicht zu groß
        assert sum(z) == pytest.approx(10)
        for k in range(1, 4):
            for s in combinations(range(n), k):
                assert sum(z[i] for i in s) >= k * (k + 1) / 2 - 1e-6              # Rados Ungleichungen


def _martin_lp(n, kosten):
    """Martins Formulierung für K_n (je Wurzel k Richtungswerte y >= 0, y_ij + y_ji = x_ij, jeder Ort außer k hat genau einen Elternort)."""
    kanten = I.paare(n)
    m = len(kanten)
    nv = m + 2 * m * n
    zeilen, rechts = [], []
    z = np.zeros(nv)
    z[:m] = 1
    zeilen.append(z)
    rechts.append(n - 1)
    for k in range(n):
        basis = m + 2 * m * k
        aus = [np.zeros(nv) for _ in range(n)]
        for e, (i, j) in enumerate(kanten):
            z = np.zeros(nv)
            z[basis + 2 * e] = z[basis + 2 * e + 1] = 1
            z[e] = -1
            zeilen.append(z)
            rechts.append(0)
            aus[i][basis + 2 * e] = 1
            aus[j][basis + 2 * e + 1] = 1
        for i in range(n):
            zeilen.append(aus[i])
            rechts.append(0 if i == k else 1)
    c = np.zeros(nv)
    c[:m] = kosten
    res = linprog(c, A_eq=np.array(zeilen), b_eq=rechts, bounds=(0, None), method="highs")
    assert res.status == 0
    return res.fun


def test_martins_formulierung_trifft_den_kleinsten_spannbaum():
    """Die Projektion ist das Spannbaumpolytop: auf zufälligen Zielen derselbe Wert wie networkx (Kruskal)."""
    nx = pytest.importorskip("networkx")
    for n in (4, 5, 6):
        rng = SplitMix64(60 + n)
        for _ in range(15):
            kosten = [rng.below(41) - 20 for _ in I.paare(n)]
            g = nx.Graph()
            g.add_weighted_edges_from((u, v, w) for (u, v), w in zip(I.paare(n), kosten))
            soll = sum(d["weight"] for _, _, d in nx.minimum_spanning_tree(g).edges(data=True))
            assert _martin_lp(n, kosten) == pytest.approx(soll, abs=1e-6)


# ---------------------------------------------------------------- Referenz: Belege unabhängig vom MILP-Status

def test_dualschranke_ist_fuer_beliebige_dualwerte_eine_untere_schranke():
    """Auch mit verfälschten Dualwerten darf die Schranke nie über der kürzesten Rundreise liegen (schwache Dualität)."""
    touren = alle_touren(7)
    rng = SplitMix64(71)
    for inst in instanzen(7, 6):
        soll = (touren @ np.array(inst.kosten)).min()
        m = L.leiter(inst).modell_zm
        assert m.dualschranke() <= soll + 1e-6 and m.dualschranke() == pytest.approx(m.wert, abs=1e-5)
        for _ in range(10):
            falsch = m.kopie()
            falsch.dual_grad = m.dual_grad + np.array([rng.below(201) - 100 for _ in range(7)], float)
            falsch.dual_zeilen = m.dual_zeilen * 0 + np.array([rng.below(101) - 20 for _ in range(len(m.dual_zeilen))], float)
            assert falsch.dualschranke() <= soll + 1e-6


def test_eigenes_branch_and_bound_gegen_die_dynamik_ueber_teilmengen():
    """Nur mit dem Grad-LP gestartet (viel zu verzweigen): Mit der richtigen Länge belegt es, mit einer zu großen findet es eine kürzere
    Rundreise. Referenz ist die Dynamik über Teilmengen, die weder LP noch MILP benutzt (selbst gegen alle Rundreisen geprüft)."""
    touren = alle_touren(8)
    for inst in instanzen(8, 5):
        assert R.held_karp(8, inst.kanten, inst.kosten) == int(round((touren @ np.array(inst.kosten)).min()))
    verzweigt = geprueft = 0
    for i in range(200):
        inst = I.zufallsinstanz("matrix", 13, 4000 + i)
        # nur Instanzen, bei denen das Subtour-LP mehr als 1 unter dem Optimum liegt: dort genügt der erste Knoten nicht
        sub = L.leiter(inst).wert_subtour
        soll = R.held_karp(13, inst.kanten, inst.kosten)
        if soll - sub < 1:
            continue
        m = L.neues_modell(inst)
        L.grad_stufe(m)
        ergebnis, _, knoten = R.zweitbeweis(m, soll, 60.0)
        assert ergebnis == "belegt", (i, ergebnis)
        verzweigt += knoten > 1
        ergebnis, kuerzer, _ = R.zweitbeweis(m, soll + 5, 60.0)
        assert ergebnis == "kuerzer", (i, ergebnis)
        assert R.ist_rundreise(13, inst.kanten, kuerzer) and soll <= R.tourlaenge(inst.kosten, kuerzer) < soll + 5
        geprueft += 1
        if geprueft == 8:
            break
    assert geprueft == 8 and verzweigt == 8
    assert R.zweitbeweis(m, soll, 0.0)[0] == "offen" and R.zweitbeweis(m, soll, 60.0, max_knoten=0)[0] == "offen"


@pytest.mark.parametrize("n", [9, 20, 30])
def test_ein_falsches_milp_ergebnis_wird_erkannt(monkeypatch, n):
    """Liefert das MILP eine zu lange Rundreise, darf sie nie als Optimum durchgehen (bis 13 Orte Dynamik, darüber Branch-and-Bound)."""
    erkannt = 0
    for i in range(6):
        inst = I.zufallsinstanz("matrix", n, 5000 + i)
        richtig = R.optimum(n, inst.kanten, inst.kosten, None, 60.0, 60.0)
        assert richtig.belegt
        folge = R.ordnung(n, inst.kanten, richtig.tour)
        folge[1], folge[n // 2] = folge[n // 2], folge[1]                 # zwei Orte vertauscht: eine andere, längere Rundreise
        falsch = tuple(I.kantennummer(n, a, b) for a, b in zip(folge, folge[1:] + folge[:1]))
        if R.tourlaenge(inst.kosten, falsch) == richtig.wert:
            continue
        monkeypatch.setattr(R, "milp_rundreise", lambda *a, falsch=falsch, **k: falsch)
        o = R.optimum(n, inst.kanten, inst.kosten, None, 60.0, 60.0)
        monkeypatch.undo()
        assert o.status == R.UNBESTIMMT and o.wert is None, o
        erkannt += 1
    assert erkannt >= 5


def test_nur_milp_wird_als_solches_gemeldet(monkeypatch):
    """Gelingt kein zweiter Beleg, heißt der Status „nur MILP“, nie belegt."""
    monkeypatch.setattr(R, "zweitbeweis", lambda *a, **k: ("offen", (), 0))
    gefunden = 0
    for i in range(30):
        inst = I.zufallsinstanz("matrix", 20, 5100 + i)
        lt = L.leiter(inst)
        o = R.optimum(20, inst.kanten, inst.kosten, lt.modell_zm, 60.0)
        if o.status == R.NUR_MILP:
            gefunden += 1
            assert not o.belegt and o.wert >= lt.wert_zweimatching - 1e-6
        else:
            assert o.status == R.DUALSCHRANKE
    assert gefunden >= 3


def test_referenz_gegen_cp_sat():
    pytest.importorskip("ortools")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from rpt_make_tables import cpsat_optimum

    stati = set()
    for familie, n, seed in (("karte", 20, 1), ("karte", 30, 34044), ("matrix", 30, 3), ("haufen", 30, 5), ("matrix", 12, 33398),
                             ("karte", 30, 34003)):
        inst = I.zufallsinstanz(familie, n, seed)
        o = R.optimum(n, inst.kanten, inst.kosten, L.leiter(inst).modell_zm, 60.0, 60.0)
        assert o.belegt and o.wert == cpsat_optimum(n, inst.kanten, inst.kosten), (familie, n, seed, o)
        stati.add(o.status)
    assert R.DUALSCHRANKE in stati and len(stati) >= 2


@pytest.mark.parametrize("status, erwartet", [(2, "belegt"), (4, "offen"), (1, "offen")])
def test_nur_unzulaessige_knoten_werden_abgeschnitten(monkeypatch, status, erwartet):
    """Meldet der Löser in jedem Knoten „unzulässig“ (2), gilt der Knoten als erledigt; bei jedem anderen Fehler bleibt der Beleg offen."""
    inst = I.zufallsinstanz("matrix", 9, 3)
    m = L.neues_modell(inst)
    L.grad_stufe(m)

    def fehler(self, unten=None, oben=None):
        raise LP.LPFehler("Probe", status)

    monkeypatch.setattr(LP.Modell, "loese", fehler)
    assert R.zweitbeweis(m, 100000, 10.0)[0] == erwartet
