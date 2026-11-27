"""Verdikte an von Hand gesetzten Werten: jede Art, jede Schwelle und jede Zahl im Text."""

import pytest

import rpt_constants as C
import rpt_evaluation as E
import rpt_instanzen as I
import rpt_leiter as L
import rpt_netz as NZ
import rpt_zweimatching as ZM

F = L.FERTIG


# ---------------------------------------------------------------- Leiter

def test_grad_stufe_geloest_gebrochen_oder_mit_luecke():
    assert E.verdikt_stufenwerte("grad", None, 100.0, True, 0, F, 100)[0] == "success"
    art, text = E.verdikt_stufenwerte("grad", None, 100.0, False, 0, F, 100)
    assert art == "warning" and "gebrochen" in text
    art, text = E.verdikt_stufenwerte("grad", None, 90.0, True, 0, F, 100)
    assert art == "warning" and "**10,00 %**" in text and "90" in text and "100" in text


def test_ohne_bestimmtes_optimum_gibt_es_keine_luecke():
    art, text = E.verdikt_stufenwerte("zm", (90.0, False), 95.5, False, 3, F, None)
    assert art == "info" and "95,50" in text and "%" not in text


def test_abbruch_ist_nur_eine_untere_schranke():
    art, text = E.verdikt_stufenwerte("zm", (90.0, False), 95.0, False, 3, L.ABGEBROCHEN, 100)
    assert art == "warning" and "untere Schranke" in text


def test_vorige_stufe_schon_optimal_meldet_keine_wirkung():
    art, text = E.verdikt_stufenwerte("zm", (100.0, True), 100.0, True, 0, F, 100)
    assert (art, text) == ("info", "Schon die vorige Stufe war ganzzahlig und optimal; diese Stufe lädt nichts.")
    # Wert vorher gleich, aber gebrochen: das ist nicht „schon gelöst“
    assert E.verdikt_stufenwerte("zm", (100.0, False), 100.0, True, 2, F, 100)[0] == "success"


def test_stufe_loest_mit_einer_oder_mehreren_zeilen():
    assert E.verdikt_stufenwerte("subtour", (6.0, False), 12.0, True, 1, F, 12)[1] == \
        "Grad-LP 6, mit einer Subtourzeile 12 = Optimum, und die Lösung ist eine Rundreise."
    art, text = E.verdikt_stufenwerte("zm", (9.0, False), 10.0, True, 4, F, 10)
    assert art == "success" and "Subtour-LP 9" in text and "4 2-Matching-Ungleichungen 10 = Optimum" in text


def test_wert_stimmt_aber_gebrochen():
    art, text = E.verdikt_stufenwerte("subtour", (90.0, False), 100.0, False, 3, F, 100)
    assert art == "info" and "gebrochen" in text and "3 Subtourzeilen" in text


def test_offen_mit_und_ohne_zeilen():
    art, text = E.verdikt_stufenwerte("zm", (95.0, False), 97.0, False, 2, F, 100)
    assert art == "warning" and "**3,00 %**" in text and "(2 2-Matching-Ungleichungen nachgeladen)" in text
    art, text = E.verdikt_stufenwerte("zm", (95.0, False), 95.0, False, 0, F, 100)
    assert art == "warning" and "findet keine verletzte Ungleichung" in text and "**5,00 %**" in text


# ---------------------------------------------------------------- gebrochene Lösung

def test_gebrochen_alle_zweige():
    assert E.verdict_gebrochen("ganzzahlig", None) == ("success", "Nichts ist gebrochen: alle Werte sind 0 oder 1.")
    art, text = E.verdict_gebrochen("halb", (True, (3, 3)))
    assert art == "info" and "2 getrennte Kreise (2 × Länge 3)" in text
    assert "einen Kreis durch 8 Orte" in E.verdict_gebrochen("halb", (True, (8,)))[1]
    art, text = E.verdict_gebrochen("halb", (False, (3, 5)))
    assert art == "info" and "nicht nur getrennte Kreise" in text and "Zahl der Orte: 3, 5" in text
    assert E.anzahl_wort(1, "Zeile", "Zeilen") == "1 Zeile" and E.anzahl_wort(0, "Zeile", "Zeilen") == "0 Zeilen"
    art, text = E.verdict_gebrochen("anders", (3, 6))
    assert art == "warning" and "Nenner 3, 6" in text
    assert E.verdict_gebrochen("unbekannt", None)[0] == "warning"


# ---------------------------------------------------------------- Falle

def test_falle_mit_uebersehener_ungleichung():
    art, text = E.verdict_falle(L.KOMPONENTEN, 1 / 3, 4677.0, 4689.0, 4689, F, 4)
    assert art == "warning" and "Wert ⅓ < 1" in text and "0,256 %" in text


def test_falle_ohne_wirkung_und_mit_wirkung():
    assert E.verdict_falle(L.KOMPONENTEN, None, 10.0, 10.0, 10, F, 0)[0] == "info"
    art, text = E.verdict_falle(L.KOMPONENTEN, None, 10.0, 10.0, 10, F, 2)
    assert art == "success" and "genügt die einfache Regel" in text and "2 Ungleichungen" in text
    assert E.verdict_falle(L.PADBERG_RAO, None, 10.0, 10.0, 10, F, 0)[0] == "info"
    assert E.verdict_falle(L.PADBERG_RAO, None, 10.0, 10.0, 10, F, 3)[0] == "success"
    assert E.verdict_falle(L.PADBERG_RAO, None, 10.0, 10.0, 10, L.ABGEBROCHEN, 3)[0] == "warning"
    art, text = E.verdict_falle(L.KOMPONENTEN, 0.0, 9.0, 10.0, None, F, 1)
    assert art == "warning" and "des Optimums" not in text
    # gleicher Wert trotz verletzter Ungleichung: kein „0 % darunter“
    art, text = E.verdict_falle(L.KOMPONENTEN, 1 / 3, 13.0, 13.0, 14, F, 8)
    assert art == "warning" and "stimmt hier trotzdem mit dem exakten überein (13)" in text and "%" not in text


# ---------------------------------------------------------------- Kämme

def test_kamm_alle_zweige():
    assert E.verdict_kamm(10.0, True, 10.0, True, 0, F, 10)[1] == \
        "Schon die 2-Matching-Ungleichungen genügten: 10 = Optimum; es wird kein Kamm nachgeladen."
    art, text = E.verdict_kamm(9.0, False, 10.0, True, 1, F, 10)
    assert art == "success" and "Mit 1 Kamm gelöst" in text
    assert "Mit 3 Kämmen gelöst" in E.verdict_kamm(9.0, False, 10.0, True, 3, F, 10)[1]
    assert E.verdict_kamm(9.0, False, 10.0, False, 3, F, 10)[0] == "info"
    art, text = E.verdict_kamm(13.0, False, 40 / 3, False, 8, F, 14)
    assert art == "warning" and "13,33" in text and "14" in text and "8 nachgeladen" in text
    assert E.verdict_kamm(9.0, False, 9.5, False, 3, L.ABGEBROCHEN, 10)[0] == "warning"
    assert E.verdict_kamm(9.0, False, 9.5, False, 3, F, None)[0] == "info"


# ---------------------------------------------------------------- Tabellen

def _zeile(geloest, luecken, familie="karte", n=50, instanzen=60):
    return {"familie": familie, "n": n, "instanzen": instanzen,
            "geloest": {"grad": geloest[0], "subtour": geloest[1], "exakt": geloest[2]},
            "luecke_mittel": {"grad": luecken[0], "subtour": luecken[1], "exakt": luecken[2]}}


def test_stufentabelle_nennt_zahlen_und_bei_grosser_grad_luecke_die_andere_form():
    art, text = E.verdict_stufentabelle(_zeile((0, 9, 32), (6.656, 0.826, 0.219)))
    assert art == "info" and "Karte, 50 Orte" in text and "0 / 9 / 32 von 60" in text and "6,66 % / 0,83 % / 0,219 %" in text
    assert "28 Instanzen" in text
    art, text = E.verdict_stufentabelle(_zeile((0, 28, 44), (20.0, 0.18, 0.058), "haufen"))
    assert "20,0 % unter dem Optimum" in text and "0,18 %" in text
    # Schwelle 10 % ist inklusive
    assert "unter dem Optimum" in E.verdict_stufentabelle(_zeile((0, 1, 2), (10.0, 1.0, 0.5)))[1]
    assert "unter dem Optimum" not in E.verdict_stufentabelle(_zeile((0, 1, 2), (9.99, 1.0, 0.5)))[1]
    art, text = E.verdict_stufentabelle(_zeile((44, 99, 100), (4.0, 0.004, 0.0), n=8, instanzen=100))
    assert art == "success" and "genügen hier auf allen Instanzen" in text


def test_groesse_monoton_oder_nicht():
    fallend = [(8, {"exakt": 100.0}, 100), (12, {"exakt": 98.5}, 200), (16, {"exakt": 96.0}, 100)]
    art, text = E.verdict_groesse("karte", fallend)
    assert art == "info" and "fällt in jedem gemessenen Schritt (8: 100,0 %, 12: 98,5 %, 16: 96,0 %)" in text
    nicht = [(12, {"exakt": 89.5}, 200), (16, {"exakt": 90.0}, 100), (30, {"exakt": 69.0}, 100)]
    art, text = E.verdict_groesse("matrix", nicht)
    assert art == "warning" and "von 12 auf 16 Orte nicht (89,5 % → 90,0 %)" in text and "von 16 auf 30" not in text
    # gleich viel ist kein Fallen
    assert E.verdict_groesse("karte", [(8, {"exakt": 90.0}, 1), (12, {"exakt": 90.0}, 1)])[0] == "warning"


def test_komponenten_verdikt_zaehlt_und_nennt_die_schlimmste_reihe():
    def z(familie, n, uebersehen, abstand):
        return {"familie": familie, "n": n, "instanzen": 100, "komponentenregel": {"alle": {"uebersehen": uebersehen},
                                                                                  "eine": {"abstand_max": abstand}}}
    art, text = E.verdict_komponenten([z("karte", 30, 15, 0.59), z("matrix", 30, 5, 0.872)])
    assert art == "info" and "2 Messreihen" in text and "20 von 200" in text and "0,87 %" in text and "Zufallsmatrix, 30 Orte" in text


def test_facetten_verdikt():
    zeile = {"familie": "karte", "n": 50, "instanzen": 60,
             "subtourzeilen": {"alle_median": 24.0, "alle_max": 61, "eine_median": 10.0}}
    art, text = E.verdict_facetten(zeile)
    assert art == "info" and "2^49 − 51" in text and E.ganzzahl(2 ** 49 - 51) in text and "Median 24" in text and "größte 61" in text
    assert "Median 10)" in text


# ---------------------------------------------------------------- Netz

def test_netz_alle_zweige():
    art, text = E.verdict_netz(16, 126, 126, 63, 65534, False, True, None, None, 10.0, 10.0)
    assert art == "success"
    assert "126 Ungleichungen in 126 Variablen (63 Gleichungen) statt 65.534 Facetten" in text
    art, text = E.verdict_netz(3, 6, 6, 3, 6, False, True, 6, 6, 1.0, 1.0)
    assert art == "warning" and "6 Ungleichungen gegen 6 Facetten: Hier spart die Erweiterung nichts." in text
    art, text = E.verdict_netz(4, 8, 8, 4, 14, True, False, 16, 24, 5.0, 5.0)
    assert art == "warning" and "**16 von 24**" in text and "zu klein, nie zu groß" in text and "fällt es nicht auf" in text
    art, text = E.verdict_netz(4, 8, 8, 4, 14, True, False, 12, 24, 7.0, 5.0)
    assert "größer als das Minimum 5" in text
    art, text = E.verdict_netz(8, 36, 36, 18, 254, True, False, None, None, 7.0, 5.0)
    assert art == "warning" and "echt kleiner" in text
    art, text = E.verdict_netz(5, 20, 20, 10, 30, True, True, 120, 120, 5.0, 5.0)
    assert art == "success" and "überflüssig" in text
    assert E.verdict_netz(16, 126, 126, 63, 65534, False, True, None, None, 10.0, 11.0)[0] == "error"


def test_netzbild_n4_fuenfter_komparator_gestrichen():
    nb = E.netzbild(4, NZ.BATCHER, 4, 1)
    assert (nb.ungleichungen, nb.facetten, nb.sortiert, len(nb.drin), len(nb.draussen)) == (8, 14, False, 16, 8)
    voll = E.netzbild(4, NZ.BATCHER, None, 1)
    assert voll.sortiert and len(voll.drin) == 24 and voll.lp_wert == pytest.approx(voll.sort_wert)
    assert E.netzbild(4, NZ.BATCHER, 99, 1).entfernt is None


# ---------------------------------------------------------------- Beschreibungen und Format

def test_zeile_beschreiben_fuer_alle_drei_arten(h1, halbe_loesung, zwei_dreiecke):
    text = E.zeile_beschreiben(h1, zwei_dreiecke, frozenset({0, 1, 2}))
    assert "Subtourzeile" in text and "{a, b, c}" in text and "x(δ(S)) = 0" in text
    zeile = ZM.trenne_padberg_rao(6, h1.kanten, halbe_loesung)[0]
    text = E.zeile_beschreiben(h1, halbe_loesung, zeile)
    assert "3 Zahnkanten" in text and "= 0 < 1" in text and "x(δ(H)) + Σ x(δ(T)) = 9 < 10" in text
    text = E.zeile_beschreiben(h1, halbe_loesung, (1.0, frozenset({0, 1, 2}), (frozenset({0, 3}), frozenset({1, 4}), frozenset({2, 5}))))
    assert "Kamm" in text and "= 9 < 10" in text


def test_zahlformate():
    assert [E.zahl(v) for v in (6.0, 13.3333, 4689.0, 12345.0, None)] == ["6", "13,33", "4689", "12.345", "–"]
    assert E.ganzzahl(65534) == "65.534" and E.ganzzahl(100) == "100"
    assert [E.bruch(v) for v in (0.5, 1 / 3, 1.0, 0.0, 5 / 6)] == ["½", "⅓", "1", "0", "5/6"]
    assert E.prozent(None) == "–" and E.prozent(2.5, 1) == "2,5 %"
    assert E.luecke(9.0, 10) == pytest.approx(10.0) and E.luecke(10.0000001, 10) == 0.0 and E.luecke(1.0, None) is None
    assert E.instanz_text(E.Settings("prisma", 0, 0, 1, 2, 1)) == "Prisma, Wege aus 1 Kante (6 Orte), Dreieckskosten 2, Wegkosten 1"
    assert E.instanz_text(E.Settings("karte", 30, 5)) == "Karte, 30 Orte, Zufalls-Seed 5"


def test_bereinigte_einstellungen_haben_einen_schluessel_je_instanz():
    assert E.Settings("prisma", 17, 4, 2, 3, 1).bereinigt() == E.Settings("prisma", 0, 0, 2, 3, 1)
    assert E.Settings("karte", 17, 4, 2, 3, 1).bereinigt() == E.Settings("karte", 17, 4, 1, 1, 1)


def test_jede_verfahrensbeschreibung_ist_gesetzt():
    assert set(C.REGEL_BESCHREIBUNG) == set(C.REGELN) and set(C.NETZ_BESCHREIBUNG) == set(NZ.NETZE) == set(C.NETZ_NAMEN)
    assert set(C.FAMILIEN_HILFE) == set(C.FAMILIEN) == set(C.FAMILIEN_NAMEN)
