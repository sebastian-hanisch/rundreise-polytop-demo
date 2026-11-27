"""Je Einheit ein kleiner Test mit von Hand gerechneten Werten (Prisma, siehe conftest.py)."""

from fractions import Fraction

import numpy as np
import pytest

import rpt_huelle as H
import rpt_instanzen as I
import rpt_kamm as K
import rpt_leiter as L
import rpt_lp as LP
import rpt_netz as NZ
import rpt_referenz as R
import rpt_subtour as ST
import rpt_zweimatching as ZM
from conftest import DREIECKE, VERBINDER, vektor

ABC, DEF = frozenset({0, 1, 2}), frozenset({3, 4, 5})


def kanten_nr(paare_liste, n=6):
    return [I.kantennummer(n, u, v) for u, v in paare_liste]


# ---------------------------------------------------------------- Instanzen

def test_gerundeter_abstand_rundet_kaufmaennisch_nur_mit_ganzen_zahlen():
    assert [I.gerundeter_abstand(d2) for d2 in (0, 2, 6, 7, 25)] == [0, 1, 2, 3, 5]


def test_paare_und_kantennummer_passen_zusammen():
    assert I.paare(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for n in (3, 6, 11):
        assert [I.kantennummer(n, u, v) for u, v in I.paare(n)] == list(range(len(I.paare(n))))
    assert I.kantennummer(6, 4, 1) == I.kantennummer(6, 1, 4)


def test_prisma_grundkanten_und_kosten(h1, h2):
    n, grund, wege = I.prisma_grundkanten(2, 1, 1)
    assert n == 9 and wege == [[0, 6, 3], [1, 7, 4], [2, 8, 5]] and len(grund) == 12
    # H1: Dreiecke 2, Verbinder 1, übrige Paare 3 (kürzester Weg über einen Verbinder)
    assert [h1.kosten_von(u, v) for u, v in DREIECKE] == [2] * 6
    assert [h1.kosten_von(u, v) for u, v in VERBINDER] == [1] * 3
    assert {h1.kosten_von(u, v) for u, v in I.paare(6) if (u, v) not in DREIECKE + VERBINDER} == {3}
    assert h2.kosten_von(0, 1) == 1 and h2.kosten_von(0, 3) == 4 and h2.kosten_von(0, 4) == 5
    assert h1.namen == ("a", "b", "c", "d", "e", "f")


def test_zufallsinstanzen_sind_deterministisch_und_im_bereich():
    for familie in I.ZUFALLSFAMILIEN:
        a, b = I.zufallsinstanz(familie, 12, 7), I.zufallsinstanz(familie, 12, 7)
        assert a == b and len(a.kosten) == 66 and all(isinstance(k, int) for k in a.kosten)
        assert a != I.zufallsinstanz(familie, 12, 8)
    matrix = I.zufallsinstanz("matrix", 20, 3)
    assert min(matrix.kosten) >= 1 and max(matrix.kosten) <= 1000
    karte = I.zufallsinstanz("karte", 20, 3)
    assert all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in karte.punkte)
    assert karte.kosten[0] == I.gerundeter_abstand((karte.punkte[0][0] - karte.punkte[1][0]) ** 2 + (karte.punkte[0][1] - karte.punkte[1][1]) ** 2)
    with pytest.raises(ValueError):
        I.zufallsinstanz("karte", 2, 0)
    with pytest.raises(ValueError):
        I.zufallsinstanz("quadrat", 5, 0)


# ---------------------------------------------------------------- LP

def test_grad_lp_auf_dem_dreieck_ist_die_rundreise():
    m = LP.Modell(3, I.paare(3), [4, 5, 6])
    wert, x = m.loese()
    assert wert == pytest.approx(15) and LP.ist_ganzzahlig(x)


def test_modell_nimmt_jede_zeile_nur_einmal_auf(h2):
    m = LP.Modell(6, h2.kanten, h2.kosten)
    zeile = ST.zeile(h2.kanten, ABC)
    assert m.add(("S", ABC), *zeile) and not m.add(("S", ABC), *zeile)
    assert m.zeilen == 1 and m.matrix().shape == (1, 15)
    kopie = m.kopie()
    assert kopie.add(("S", DEF), *ST.zeile(h2.kanten, DEF)) and m.zeilen == 1 and kopie.zeilen == 2


def test_gradwerte_von_hand(h1, h2, halbe_loesung, zwei_dreiecke):
    # H1: die halbe Lösung kostet 6·½·2 + 3·1 = 9; H2: zwei Dreiecke kosten 6
    assert sum(c * v for c, v in zip(h1.kosten, halbe_loesung)) == pytest.approx(9)
    assert sum(c * v for c, v in zip(h2.kosten, zwei_dreiecke)) == pytest.approx(6)
    assert LP.gradsummen(6, h1.kanten, halbe_loesung) == pytest.approx([2] * 6)
    assert L.grad_stufe(L.neues_modell(h1)).wert == pytest.approx(9)
    assert L.grad_stufe(L.neues_modell(h2)).wert == pytest.approx(6)


def test_struktur_erkennt_ganzzahlig_halbzahlig_und_andere_brueche(h1, halbe_loesung, zwei_dreiecke):
    assert LP.struktur(6, h1.kanten, zwei_dreiecke) == ("ganzzahlig", None)
    art, (nur_kreise, laengen) = LP.struktur(6, h1.kanten, halbe_loesung)
    assert art == "halb" and nur_kreise and laengen == (3, 3)
    # ein Weg aus zwei halben Kanten ist kein Kreis
    art, (nur_kreise, laengen) = LP.struktur(4, I.paare(4), vektor(4, {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 1.0}))
    assert art == "halb" and not nur_kreise and laengen == (3,)
    drittel = vektor(4, {(0, 1): 1 / 3, (1, 2): 2 / 3, (2, 3): 1.0})
    assert LP.struktur(4, I.paare(4), drittel) == ("anders", (3,))
    assert LP.als_bruch(0.2500001) == Fraction(1, 4) and LP.als_bruch(0.12345678, 20) is None


# ---------------------------------------------------------------- Subtour

def test_komponenten_und_kleinere_seite(h1, zwei_dreiecke):
    assert ST.komponenten(6, h1.kanten, zwei_dreiecke) == [ABC, DEF]
    assert ST.kleinere_seite(6, ABC) == DEF           # gleich groß: die Seite ohne Knoten 0
    assert ST.kleinere_seite(6, {0, 1}) == frozenset({0, 1})
    assert ST.kleinere_seite(6, {1, 2, 3, 4}) == frozenset({0, 5})


def test_verletzte_subtours_bei_unzusammenhaengendem_und_zusammenhaengendem_traeger(h1, zwei_dreiecke):
    assert ST.verletzte_subtours(6, h1.kanten, zwei_dreiecke) == [(0.0, DEF)]
    # Dreiecke mit 0,9 und Verbinder mit 0,2: Grad 1,8 + 0,2 = 2, Schnitt um abc nur 0,6
    x = vektor(6, {**{e: 0.9 for e in DREIECKE}, **{e: 0.2 for e in VERBINDER}})
    gefunden = ST.verletzte_subtours(6, h1.kanten, x)
    assert gefunden[0][1] == DEF and gefunden[0][0] == pytest.approx(0.6)
    assert ST.wert(h1.kanten, x, ABC) == pytest.approx(0.6)


def test_halbe_loesung_verletzt_keine_subtourzeile(h1, halbe_loesung):
    assert ST.verletzte_subtours(6, h1.kanten, halbe_loesung) == []


# ---------------------------------------------------------------- 2-Matching

def test_hilfsgraph_der_halben_loesung(h1, halbe_loesung):
    knoten, graph, markiert, teil = ZM.hilfsgraph(6, h1.kanten, halbe_loesung)
    assert knoten == 6 + 9 and len(graph) == 18 and len(teil) == 9
    assert sum(markiert) % 2 == 0 and all(markiert[w] == 1 for _, w in teil)
    # Kapazitäten je geteilter Kante summieren sich zu 1
    for i in range(0, 18, 2):
        assert graph[i][2] + graph[i + 1][2] == pytest.approx(1.0)


def test_beste_zaehne_von_hand(h1, halbe_loesung):
    wert, zaehne = ZM.beste_zaehne(h1.kanten, halbe_loesung, ABC)
    assert wert == pytest.approx(0) and set(zaehne) == set(kanten_nr(VERBINDER))
    # gerade viele Kanten über ½: es wechselt die mit dem kleinsten |1 − 2x| (hier 0,8): Wert 0,1 + 0,8 + 0,1 = 1,0
    x = vektor(4, {(0, 1): 0.9, (0, 2): 0.8, (0, 3): 0.1})
    wert, zaehne = ZM.beste_zaehne(I.paare(4), x, frozenset({0}))
    assert wert == pytest.approx(1.0) and zaehne == (I.kantennummer(4, 0, 1),)
    assert ZM.beste_zaehne(I.paare(4), x, frozenset()) is None


def test_wert_und_zeile_der_ungleichung(h1, halbe_loesung):
    zaehne = tuple(kanten_nr(VERBINDER))
    assert ZM.wert(h1.kanten, halbe_loesung, ABC, zaehne) == pytest.approx(0)
    spalten, werte, grenze = ZM.zeile(h1.kanten, ABC, zaehne)
    links = sum(w * halbe_loesung[j] for j, w in zip(spalten, werte))
    assert grenze == -2 and links == pytest.approx(-3)       # x(δ ohne F) − x(F) = 0 − 3 < 1 − 3
    assert ZM.zaehne_disjunkt(h1.kanten, zaehne)
    assert not ZM.zaehne_disjunkt(h1.kanten, tuple(kanten_nr([(0, 1), (0, 3)])))


def test_padberg_rao_findet_griff_abc_mit_den_verbindern(h1, halbe_loesung):
    gefunden = ZM.trenne_padberg_rao(6, h1.kanten, halbe_loesung)
    assert gefunden and gefunden[0].wert == pytest.approx(0)
    assert gefunden[0].griff == DEF and set(gefunden[0].zaehne) == set(kanten_nr(VERBINDER))


def test_komponentenregel_nimmt_die_beiden_dreiecke_in_knotenfolge(h1, halbe_loesung):
    gefunden = ZM.trenne_komponentenregel(6, h1.kanten, halbe_loesung)
    assert [z.griff for z in gefunden] == [ABC, DEF] and all(z.wert == pytest.approx(0) for z in gefunden)


def test_aufzaehlen_findet_den_wert_null(h1, halbe_loesung):
    wert, griff = ZM.staerkste_durch_aufzaehlen(6, h1.kanten, halbe_loesung)
    assert wert == pytest.approx(0) and ST.kleinere_seite(6, griff) == DEF
    with pytest.raises(ValueError):
        ZM.staerkste_durch_aufzaehlen(17, I.paare(17), [0.0] * 136)


def test_kammform_ist_die_kammungleichung(h1, halbe_loesung):
    zaehne = tuple(kanten_nr(VERBINDER))
    links, rechts = ZM.kammwert(h1.kanten, halbe_loesung, ABC, zaehne)
    assert links == pytest.approx(9) and rechts == 10                       # x(δ(H)) = 3, jeder Zahn 2
    # Umrechnung für einen Zahn aus einer Kante e: x(δ(T)) = 4 − 2·x_e
    for u, v in VERBINDER:
        assert ST.wert(h1.kanten, halbe_loesung, frozenset({u, v})) == pytest.approx(4 - 2 * halbe_loesung[I.kantennummer(6, u, v)])


def test_zahnzerlegung_ein_schritt_von_hand():
    # Griff {0}: die Zähne 0–1 und 0–2 treffen sich am Ort 0. Er wechselt die Seite, beide Zähne fallen weg; Zahn 3–4 bleibt.
    kanten = I.paare(5)
    zaehne = tuple(kanten_nr([(0, 1), (0, 2), (3, 4)], 5))
    griff, rest = ZM.reduziere_schritt(5, kanten, frozenset({0, 3}), zaehne)
    assert griff == frozenset({3}) and rest == (I.kantennummer(5, 3, 4),)
    assert ZM.reduziere_schritt(5, kanten, griff, rest) is None
    kurz, f, disjunkt = ZM.kammform(5, kanten, frozenset({0, 3}), zaehne)
    assert kurz == frozenset({3}) and disjunkt
    # drei Zähne an einem Ort werden nicht zerlegt
    drei = tuple(kanten_nr([(0, 1), (0, 2), (0, 3)], 5))
    assert ZM.reduziere_schritt(5, kanten, frozenset({0}), drei) is None


# ---------------------------------------------------------------- Kämme

def test_knotenzahlen_und_innensummen():
    assert list(K.knotenzahlen(3)) == [0, 1, 1, 2, 1, 2, 2, 3]
    f = K.innensummen(3, I.paare(3), [1.0, 2.0, 3.0])
    assert f[0b011] == 1 and f[0b101] == 2 and f[0b110] == 3 and f[0b111] == 6


def test_schnittwerte_aus_den_innensummen(h1, halbe_loesung):
    c = K.schnittwerte(6, h1.kanten, halbe_loesung)
    assert c[0b000111] == pytest.approx(3) and c[0b001001] == pytest.approx(2) and c[0b000001] == pytest.approx(2)


def test_kammsuche_findet_beim_prisma_mit_langen_wegen_den_kamm():
    inst = I.prisma(2)
    kanten = inst.kanten
    wege = [(0, 6), (3, 6), (1, 7), (4, 7), (2, 8), (5, 8)]
    x = vektor(9, {**{e: 0.5 for e in DREIECKE}, **{e: 1.0 for e in wege}})
    assert LP.gradsummen(9, kanten, x) == pytest.approx([2] * 9)
    zaehne = [frozenset({0, 6, 3}), frozenset({1, 7, 4}), frozenset({2, 8, 5})]
    links = ST.wert(kanten, x, ABC) + sum(ST.wert(kanten, x, t) for t in zaehne)
    assert links == pytest.approx(9)                                     # verlangt 10: um 1 verletzt
    verletzung, griff, gefunden = K.trenne_kamm(9, kanten, x)
    assert verletzung == pytest.approx(1) and K.ist_kamm(9, griff, gefunden)
    spalten, werte, grenze = K.zeile(kanten, griff, gefunden)
    assert grenze == 10 and sum(w * x[j] for j, w in zip(spalten, werte)) == pytest.approx(10 - verletzung)


def test_ist_kamm_prueft_die_form():
    assert K.ist_kamm(6, {0, 1, 2}, [{0, 3}, {1, 4}, {2, 5}])
    assert not K.ist_kamm(6, {0, 1, 2}, [{0, 3}, {1, 4}])                    # gerade Zahl
    assert not K.ist_kamm(6, {0, 1, 2}, [{0, 3}, {1, 3}, {2, 5}])            # nicht disjunkt
    assert not K.ist_kamm(6, {0, 1, 2}, [{0, 1}, {1, 4}, {2, 5}])            # ein Zahn ganz im Griff
    with pytest.raises(ValueError):
        K.trenne_kamm(13, I.paare(13), [0.0] * 78)


# ---------------------------------------------------------------- Referenz

def test_optimum_der_handbeispiele(h1, h2):
    for inst, soll in ((h1, 10), (h2, 12)):
        o = R.optimum(inst.n, inst.kanten, inst.kosten, None, 20.0)
        assert o.belegt and o.wert == soll and R.ist_rundreise(inst.n, inst.kanten, o.tour)


def test_ist_rundreise_lehnt_zwei_dreiecke_ab(h1):
    assert not R.ist_rundreise(6, h1.kanten, kanten_nr(DREIECKE))
    assert R.ist_rundreise(6, h1.kanten, kanten_nr([(0, 1), (1, 2), (2, 5), (4, 5), (3, 4), (0, 3)]))


def test_zeitgrenze_null_meldet_nicht_bestimmt():
    inst = I.zufallsinstanz("karte", 30, 1)
    o = R.optimum(inst.n, inst.kanten, inst.kosten, None, 0.0)
    assert o.status == R.UNBESTIMMT and o.wert is None and o.grund == "Zeitgrenze" and not o.belegt


def test_dualschranke_von_hand(h2):
    # Grad-LP des H2: Dualwerte ½ je Ort geben die Schranke 2·6·½ = 6; mit der Subtourzeile wird 12 erreicht
    m = L.neues_modell(h2)
    L.grad_stufe(m)
    assert m.dualschranke() == pytest.approx(6)
    L.subtour_stufe(m)
    assert m.dualschranke() == pytest.approx(12)
    assert R.schranke_belegt(11.5, 12) and not R.schranke_belegt(11.0, 12) and not R.schranke_belegt(6.0, 12)
    assert R.runde_hoch(11.0000000001) == 11 and R.runde_hoch(11.5) == 12


def test_held_karp_von_hand(h1, h2):
    assert R.held_karp(6, h1.kanten, h1.kosten) == 10 and R.held_karp(6, h2.kanten, h2.kosten) == 12
    assert R.held_karp(3, I.paare(3), [4, 5, 6]) == 15


def test_heuristik_findet_beim_prisma_die_kuerzeste_rundreise(h1):
    quadrat = [(0, 0), (0, 10), (10, 10), (10, 0)]
    kosten = [I.gerundeter_abstand((quadrat[u][0] - quadrat[v][0]) ** 2 + (quadrat[u][1] - quadrat[v][1]) ** 2) for u, v in I.paare(4)]
    gekreuzt = tuple(I.kantennummer(4, u, v) for u, v in ((0, 2), (2, 1), (1, 3), (3, 0)))
    d = R._matrix(4, I.paare(4), kosten)
    folge = R.verbessere(d, R.ordnung(4, I.paare(4), gekreuzt))
    assert R.folgenlaenge(d, folge) == 40                                     # 2-opt entkreuzt das Quadrat
    assert R.heuristik_laenge(6, h1.kanten, h1.kosten) == 10


@pytest.mark.parametrize("status, wort", [(R.DUALSCHRANKE, "Dualschranke"), (R.HELD_KARP, "Held und"), (R.ENUMERATION, "Branch-and-Bound"),
                                          (R.NUR_MILP, "**nur laut HiGHS-MILP.**")])
def test_referenz_text_nennt_den_beleg(status, wort):
    import rpt_evaluation as E
    text = E.referenz_text(R.Optimum(status, 123, (), knoten=7))
    assert wort in text and "123" in text and "bewiesen" not in text


def test_referenz_text_nicht_bestimmt():
    import rpt_evaluation as E
    assert E.referenz_text(R.Optimum(R.UNBESTIMMT, grund="Zeitgrenze")) == "Kürzeste Rundreise nicht bestimmt (Zeitgrenze)."


# ---------------------------------------------------------------- Leiter

def test_leiter_h2_laedt_genau_eine_subtourzeile(h2):
    lt = L.leiter(h2)
    assert lt.wert_grad == pytest.approx(6) and lt.wert_subtour == pytest.approx(12) and lt.wert_zweimatching == pytest.approx(12)
    assert lt.subtour.zeilen == 1 and lt.subtour.schritte[0].neu == (DEF,) and lt.zweimatching.zeilen == 0


def test_leiter_h1_braucht_das_2_matching(h1):
    lt = L.leiter(h1)
    assert lt.wert_subtour == pytest.approx(9) and lt.subtour.zeilen == 0
    assert lt.wert_zweimatching == pytest.approx(10) and lt.zweimatching.zeilen >= 1
    assert LP.ist_ganzzahlig(lt.modell_zm.x) and lt.status() == L.FERTIG
    assert [s.stufe for s in lt.schritte()][0] == "grad"


def test_leiter_prisma_mit_wegen_aus_drei_kanten_endet_bei_vierzig_drittel():
    inst = I.prisma(3)
    lt = L.leiter(inst)
    assert (lt.wert_subtour, lt.wert_zweimatching) == pytest.approx((12, 13))
    m = lt.modell_zm.kopie()
    v = L.kamm_stufe(m)
    assert v.status == L.FERTIG and m.wert == pytest.approx(40 / 3) and K.trenne_kamm(12, inst.kanten, m.x) is None


# ---------------------------------------------------------------- Netze

def test_netzgroessen_und_sortiereigenschaft():
    assert [len(NZ.batcher_netz(n)) for n in (2, 3, 4, 5, 16)] == [1, 3, 5, 9, 63]
    assert [len(NZ.einfuege_netz(n)) for n in (3, 5, 16)] == [3, 10, 120]
    for n in range(2, 17):
        assert NZ.sortiert_alles(NZ.batcher_netz(n), n), n
        assert NZ.sortiert_alles(NZ.einfuege_netz(n), n), n
    assert not NZ.sortiert_alles(NZ.batcher_netz(3)[:-1], 3)


def test_lauf_von_hand():
    assert NZ.lauf([(0, 2), (0, 1), (1, 2)], (3, 1, 2)) == [1, 2, 3]
    gut, schlecht = NZ.sortierte_permutationen([(0, 1)], 3)
    assert len(gut) + len(schlecht) == 6 and (2, 1, 3) in gut and (3, 2, 1) in schlecht


def test_add_komparator_laesst_den_richtigen_ausgang_zu_und_lehnt_den_falschen_ab():
    modell = {"werte": 2, "eq": [], "ub": []}
    lo, hi = NZ.add_komparator(modell, 0, 1)

    def zulaessig(werte):
        return (all(sum(k * werte[w] for w, k in z.items()) == r for z, r in modell["eq"])
                and all(sum(k * werte[w] for w, k in z.items()) <= r for z, r in modell["ub"]))

    assert zulaessig({0: 3, 1: 1, lo: 1, hi: 3})
    assert zulaessig({0: 3, 1: 1, lo: 0, hi: 4})      # zulässig: die Projektion liegt trotzdem im Permutaeder
    assert not zulaessig({0: 3, 1: 1, lo: 2, hi: 2})  # lo > min(a, b)
    assert not zulaessig({0: 3, 1: 1, lo: 1, hi: 2})  # Summe falsch


def test_netz_formulierung_zaehlt_wie_beschrieben():
    f = NZ.netz_formulierung(NZ.batcher_netz(3), 3)
    assert (f.ungleichungen, f.variablen, f.gleichungen) == (6, 6, 3)
    f = NZ.netz_formulierung(NZ.batcher_netz(16), 16)
    assert (f.ungleichungen, f.variablen, f.gleichungen) == (126, 126, 63)
    assert NZ.netz_formulierung([], 3).eingang == (("c", 1), ("c", 2), ("c", 3))   # leeres Netz: nur (1, 2, 3)


def test_netz_lp_und_sortieren_von_hand():
    ziel = (5, 1, 3)
    assert NZ.sortier_optimum(ziel) == (5 * 1 + 1 * 3 + 3 * 2, [1, 3, 2])
    assert NZ.netz_lp(NZ.batcher_netz(3), 3, ziel)[0] == pytest.approx(14)
    assert NZ.netz_lp([], 3, ziel)[0] == pytest.approx(5 + 2 + 9)
    assert NZ.netz_lp([], 1, (7,))[0] == pytest.approx(7)


def test_zaehlungen_zum_permutaeder_und_zum_spannbaum():
    assert [NZ.permutaeder_facetten(n) for n in (3, 4, 16)] == [6, 14, 65534]
    assert [NZ.spannbaum_facetten(n) for n in (4, 8, 10)] == [16, 274, 1057]
    assert NZ.martin_groessen(4) == {"ungleichungen": 36, "variablen": 42, "gleichungen": 37}
    assert NZ.martin_groessen(10) == {"ungleichungen": 810, "variablen": 855, "gleichungen": 541}
    erst = next(n for n in range(4, 40) if NZ.martin_groessen(n)["ungleichungen"] < NZ.spannbaum_facetten(n))
    assert erst == 10


# ---------------------------------------------------------------- Hülle

def test_rundreisen_und_facetten_bei_fuenf_und_sechs_orten():
    assert len(H.rundreisen(4)) == 3 and len(H.rundreisen(5)) == 12 and len(H.rundreisen(6)) == 60
    b5, b6 = H.bericht(5), H.bericht(6)
    assert (b5["dimension"], b5["facetten"], b5["familien"]) == (5, 20, {"x ≥ 0": 10, "x ≤ 1": 10})
    assert (b6["dimension"], b6["facetten"]) == (9, 100)
    assert list(b6["familien"].values()) == [15, 15, 10, 60]
    assert H.bericht(4)["facetten"] == 3
    assert [H.subtourfacetten(n) for n in (5, 6, 7, 8, 9)] == [10, 25, 56, 119, 246]
    assert np.isclose(2 ** 49 - 51, H.subtourfacetten(50))
