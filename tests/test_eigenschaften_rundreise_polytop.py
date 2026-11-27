"""Eigenschaften, die gelten müssen, ohne das richtige Ergebnis zu kennen."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

import rpt_huelle as H
import rpt_instanzen as I
import rpt_leiter as L
import rpt_lp as LP
import rpt_referenz as R
import rpt_subtour as ST
import rpt_zweimatching as ZM
from rpt_rng import SplitMix64


def zufall(n, anzahl, familien=("karte", "matrix", "haufen")):
    return [I.zufallsinstanz(f, n, 900 + s) for f in familien for s in range(anzahl)]


@pytest.mark.parametrize("inst", zufall(7, 5) + [I.prisma(2), I.prisma(1, 3, 1)], ids=lambda i: f"{i.familie}-{i.n}-{i.kosten[:3]}")
def test_keine_nachgeladene_zeile_schneidet_eine_rundreise_ab(inst):
    if inst.n > 9:
        pytest.skip("alle Rundreisen nur klein")
    touren = H.rundreisen(inst.n)
    lt = L.leiter(inst)
    modell = lt.modell_zm.kopie()
    L.kamm_stufe(modell)
    for spalten, werte, grenze in zip(modell.spalten, modell.werte, modell.grenzen):
        for t in touren:
            assert sum(w * t[j] for j, w in zip(spalten, werte)) >= grenze - 1e-9


@pytest.mark.parametrize("n", [10, 20, 30])
def test_werte_steigen_und_enden_nicht_ueber_dem_optimum(n):
    for inst in zufall(n, 3):
        lt = L.leiter(inst)
        werte = [s.wert for s in lt.schritte()]
        assert all(b >= a - 1e-6 for a, b in zip(werte, werte[1:]))
        o = R.optimum(inst.n, inst.kanten, inst.kosten, lt.modell_zm, 60.0)
        assert werte[-1] <= o.wert + 1e-4


@pytest.mark.parametrize("n", [12, 20, 30])
def test_exakter_endwert_haengt_nicht_an_der_auswahlregel(n):
    for inst in zufall(n, 3):
        referenz = L.leiter(inst)
        for sub_modus in (L.ALLE, L.EINE):
            for modus in (L.ALLE, L.EINE):
                lt = L.leiter(inst, subtour_modus=sub_modus, modus=modus)
                assert lt.wert_subtour == pytest.approx(referenz.wert_subtour, abs=1e-5)
                assert lt.wert_zweimatching == pytest.approx(referenz.wert_zweimatching, abs=1e-5)
        # nach der exakten Stufe ist nichts mehr verletzt; die Komponentenregel liegt nie darüber
        assert ZM.trenne_padberg_rao(inst.n, inst.kanten, referenz.modell_zm.x) == []
        m = referenz.modell_sub.kopie()
        L.zweimatching_stufe(m, L.KOMPONENTEN, L.ALLE)
        assert m.wert <= referenz.wert_zweimatching + 1e-5


@pytest.mark.parametrize("faktor", [3, 7])
def test_kosten_mal_faktor_multipliziert_die_werte(faktor):
    for inst in zufall(14, 3):
        a = L.leiter(inst)
        b = L.leiter(I.Instanz(inst.familie, inst.n, tuple(faktor * k for k in inst.kosten), inst.punkte, inst.namen))
        for x, y in ((a.wert_grad, b.wert_grad), (a.wert_subtour, b.wert_subtour), (a.wert_zweimatching, b.wert_zweimatching)):
            assert y == pytest.approx(faktor * x, rel=1e-7)


def test_umnummerieren_der_orte_aendert_die_werte_nicht():
    rng = SplitMix64(3)
    for inst in zufall(13, 3):
        ordnung = list(range(inst.n))
        rng.shuffle(ordnung)
        kosten = tuple(inst.kosten_von(ordnung[u], ordnung[v]) for u, v in I.paare(inst.n))
        neu = I.Instanz(inst.familie, inst.n, kosten, inst.punkte, inst.namen)
        a, b = L.leiter(inst), L.leiter(neu)
        for x, y in ((a.wert_grad, b.wert_grad), (a.wert_subtour, b.wert_subtour), (a.wert_zweimatching, b.wert_zweimatching)):
            assert y == pytest.approx(x, abs=1e-5)


def _zwei_faktoren(m):
    """Alle Mengen von m Kanten mit Grad 2 an jedem Ort (Vereinigungen von Kreisen)."""
    kanten = I.paare(m)
    return [s for s in combinations(range(len(kanten)), m) if all(sum(v in kanten[j] for j in s) == 2 for v in range(m))]


def test_zahnzerlegung_identitaet_exakt_an_punkten_nur_mit_gradgleichungen():
    """Mit x(δ(v)) = 2 gilt exakt: Wert(neu) = Wert(alt) − 2·(Summe der übrigen Kanten an v, die den alten Griff kreuzen)."""
    n = 7
    kanten = I.paare(n)
    zwei = _zwei_faktoren(n)
    assert len(zwei) == 465
    rng = SplitMix64(5)
    faelle = 0
    for _ in range(8):
        wahl = [zwei[rng.below(len(zwei))] for _ in range(3)]
        x = [Fraction(sum(j in s for s in wahl), 3) for j in range(len(kanten))]
        assert [sum((w for (a, b), w in zip(kanten, x) if v in (a, b)), Fraction(0)) for v in range(n)] == [2] * n

        def wert(griff, zaehne):
            f = set(zaehne)
            return sum((1 - x[j]) if j in f else x[j] for j in LP.schnittkanten(kanten, griff))

        for k in (1, 2, 3):
            for h in combinations(range(n), k):
                h = frozenset(h)
                rand = LP.schnittkanten(kanten, h)
                for v in range(n):
                    an_v = [j for j in rand if v in kanten[j]]
                    for drei in combinations(an_v, 3):
                        assert wert(h, drei) >= 1
                    for paar in combinations(an_v, 2):
                        rest = tuple([j for j in rand if v not in kanten[j]][:1])
                        zaehne = tuple(paar) + rest
                        neu = h ^ {v}
                        if not neu:
                            continue
                        uebrige = sum(x[j] for j in an_v if j not in paar)
                        assert wert(neu, rest) == wert(h, zaehne) - 2 * uebrige
                        faelle += 1
    assert faelle > 500


def test_verkuerzte_zeilen_sind_disjunkt_und_mindestens_so_stark_auf_echten_trennungen():
    geteilt = geprueft = 0
    for inst in zufall(30, 4) + zufall(50, 3, ("karte",)):
        schritte = L.leiter(inst).schritte()
        for vorher, schritt in zip(schritte, schritte[1:]):
            x = vorher.x
            for z in schritt.neu:
                if not isinstance(z, ZM.Zeile):
                    continue
                griff, zaehne, disjunkt = ZM.kammform(inst.n, inst.kanten, z.griff, z.zaehne)
                assert disjunkt and len(zaehne) % 2 == 1 and len(zaehne) >= 3
                assert ZM.wert(inst.kanten, x, griff, zaehne) <= z.wert + 1e-6
                links, rechts = ZM.kammwert(inst.kanten, x, griff, zaehne)
                assert links - rechts == pytest.approx(ZM.wert(inst.kanten, x, griff, zaehne) - 1, abs=1e-6)
                geprueft += 1
                geteilt += not ZM.zaehne_disjunkt(inst.kanten, z.zaehne)
    assert geprueft > 40 and geteilt > 5       # die Zerlegung wird wirklich gebraucht


def test_ein_zahn_ist_unter_den_subtourzeilen_nie_verletzt():
    n = 8
    for inst in zufall(n, 4):
        m = L.neues_modell(inst)
        L.grad_stufe(m)
        L.subtour_stufe(m)
        for k in (2, 3, 4):
            for h in combinations(range(n), k):
                for j in LP.schnittkanten(inst.kanten, set(h)):
                    assert ZM.wert(inst.kanten, m.x, set(h), (j,)) >= 1 - 1e-6


# ---------------------------------------------------------------- Randfälle

@pytest.mark.parametrize("n", [4, 5])
def test_bis_fuenf_orte_ist_das_subtour_lp_schon_die_huelle(n):
    touren = np.array(H.rundreisen(n), float)
    rng = SplitMix64(n)
    for _ in range(100):
        kosten = [rng.below(21) - 10 for _ in I.paare(n)]
        inst = I.Instanz("matrix", n, tuple(kosten), tuple(I.kreis(n)), tuple(str(i) for i in range(n)))
        assert L.leiter(inst).wert_subtour == pytest.approx((touren @ np.array(kosten)).min(), abs=1e-6)


def test_alle_kosten_gleich():
    n = 9
    inst = I.Instanz("matrix", n, (4,) * 36, tuple(I.kreis(n)), tuple(str(i) for i in range(n)))
    lt = L.leiter(inst)
    assert lt.wert_grad == pytest.approx(36) and lt.wert_subtour == pytest.approx(36) and lt.wert_zweimatching == pytest.approx(36)
    assert R.optimum(n, inst.kanten, inst.kosten, lt.modell_zm, 10.0).wert == 36


def test_zwei_orte_am_selben_punkt():
    punkte = [(0, 0), (0, 0), (100, 0), (100, 100), (0, 100), (50, 50)]
    kosten = tuple(I.gerundeter_abstand((punkte[u][0] - punkte[v][0]) ** 2 + (punkte[u][1] - punkte[v][1]) ** 2) for u, v in I.paare(6))
    assert kosten[0] == 0
    inst = I.Instanz("karte", 6, kosten, tuple(punkte), tuple("123456"))
    lt = L.leiter(inst)
    soll = min(int(np.dot(t, kosten)) for t in H.rundreisen(6))
    assert lt.wert_zweimatching == pytest.approx(soll) and R.optimum(6, inst.kanten, kosten, lt.modell_zm, 10.0).wert == soll


def test_unzusammenhaengender_traeger_wird_zu_einer_rundreise():
    """Zwei weit entfernte Haufen: das Grad-LP bildet zwei Kreise, die Subtourzeile verbindet sie."""
    punkte = [(0, 0), (10, 0), (0, 10), (900, 900), (910, 900), (900, 910)]
    kosten = tuple(I.gerundeter_abstand((punkte[u][0] - punkte[v][0]) ** 2 + (punkte[u][1] - punkte[v][1]) ** 2) for u, v in I.paare(6))
    inst = I.Instanz("karte", 6, kosten, tuple(punkte), tuple("123456"))
    lt = L.leiter(inst)
    assert lt.subtour.zeilen >= 1 and ST.verletzte_subtours(6, inst.kanten, lt.modell_sub.x) == []
    assert lt.wert_subtour > lt.wert_grad + 100


def test_kleinste_zulaessige_groesse_drei_orte():
    inst = I.zufallsinstanz("karte", 3, 1)
    lt = L.leiter(inst)
    assert lt.wert_zweimatching == pytest.approx(sum(inst.kosten)) and lt.subtour.zeilen == 0
