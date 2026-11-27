"""Jede Zahl der README: die Tabellen gegen die eingefrorene Datei, dazu Nachrechnungen mit denselben Funktionen.

Plattformfest sind die LP-Werte (Optima, Werte nach Subtour- und exakter 2-Matching-Stufe, „Wert gleich Optimum“, Lücken). „Ganzzahlig und
optimal“, Zahlen nachgeladener Zeilen, gebrochene Lösungen und die Komponentenregel hängen an der Ecke, die der Löser liefert; sie werden
beim Nachrechnen nur als Band geprüft."""

import math
import re
from pathlib import Path

import pytest

import rpt_evaluation as E
import rpt_huelle as H
import rpt_instanzen as I
import rpt_messreihen as M
import rpt_netz as NZ

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FLACH = re.sub(r"\s+", " ", README)
T = E.tabellen()


def komma(wert, stellen):
    return f"{wert:.{stellen}f}".replace(".", ",")


def tausender(wert):
    return f"{wert:,}".replace(",", ".")


def tabellenzeile(anfang: str) -> list:
    zeile = next(z for z in README.splitlines() if z.startswith("| " + anfang))
    return [s.strip() for s in zeile.strip("|").split("|")]


def steht(text: str):
    assert re.sub(r"\s+", " ", text) in FLACH, text


NAMEN = {"karte": "Karte", "haufen": "Haufen", "matrix": "Zufallsmatrix"}


# ---------------------------------------------------------------- Leiter

@pytest.mark.parametrize("familie, n, anzahl", M.LEITER_REIHEN)
def test_leitertabelle(familie, n, anzahl):
    z = T["leiter"][f"{familie}-{n}"]
    assert z["instanzen"] == anzahl and z["basis"] == M.seed_von(familie, n, 0)
    spalten = tabellenzeile(f"{NAMEN[familie]} {n} ({anzahl})")
    g = z["geloest"]
    assert spalten[1:5] == [str(g["grad"]), str(g["subtour"]), f"{g['komp_alle']} / {g['komp_eine']}", str(g["exakt"])]
    gap = z["luecke_mittel"]
    assert spalten[5] == f"{komma(gap['grad'], 2)} / {komma(gap['subtour'], 3)} / {komma(gap['exakt'], 3)} %"


def test_saetze_zur_leiter():
    anteil = {n: 100.0 * z["geloest"]["exakt"] / z["instanzen"] for n, z in
              ((int(k.split("-")[1]), v) for k, v in T["leiter"].items() if k.startswith("karte-"))}
    assert [anteil[n] for n in (8, 12, 16, 30, 50)] == pytest.approx([100, 98.5, 96, 76, 53.33], abs=0.01)
    assert all(anteil[b] < anteil[a] for a, b in ((8, 12), (12, 16), (16, 30), (30, 50)))
    steht("(100 % bei 8 Orten, 98,5 % bei 12, 96 % bei 16, 76 % bei 30, 53,3 % bei 50)")
    mz = {n: 100.0 * T["leiter"][f"matrix-{n}"]["geloest"]["exakt"] / T["leiter"][f"matrix-{n}"]["instanzen"] for n in (12, 16)}
    assert mz[12] == pytest.approx(89.5) and mz[16] == pytest.approx(90.0) and mz[16] > mz[12]
    steht("89,5 % bei 12 Orten, 90 % bei 16")
    haufen = [z for k, z in T["leiter"].items() if k.startswith("haufen-")]
    grad = [z["luecke_mittel"]["grad"] for z in haufen]
    assert (komma(min(grad), 1), komma(max(grad), 1)) == ("20,0", "23,3") and all(z["luecke_mittel"]["subtour"] <= 0.18 for z in haufen)
    steht("im Mittel 20,0 bis 23,3 % unter dem Optimum")
    for k, soll in (("haufen-30", (93, 94)), ("matrix-30", (69, 70)), ("matrix-50", (31, 32))):
        assert (T["leiter"][k]["geloest"]["exakt"], T["leiter"][k]["wert_optimal"]["exakt"]) == soll
    anders = [k for k, z in T["leiter"].items() if z["wert_optimal"]["exakt"] != z["geloest"]["exakt"]]
    assert sorted(anders) == ["haufen-30", "matrix-30", "matrix-50"]
    steht("Haufen 30: 94 statt 93, Zufallsmatrix 30: 70 statt 69, Zufallsmatrix 50: 32 statt 31")


def test_gegenproben_zaehlungen():
    """Die Zählungen schreibt das Werkzeug in die Datei; bei einer Abweichung eines zweiten Lösers wäre die Erzeugung abgebrochen."""
    meta = T["meta"]
    summe = {name: sum(meta[t]["gegenproben"].get(name, 0) for t in ("leiter", "kaemme", "prisma")) for name in ("cp-sat", "held-karp")}
    leiter = sum(z["instanzen"] for z in T["leiter"].values())
    kaemme = sum(z["instanzen"] for z in T["kaemme"].values())
    prisma = len(T["prisma"]["wege"]) + len(T["prisma"]["kosten"])
    assert (leiter, kaemme, prisma) == (1480, 1600, 10) and summe["cp-sat"] == leiter + kaemme + prisma == 3090
    n13 = sum(z["instanzen"] for z in T["leiter"].values() if z["n"] <= 13)
    assert n13 == 700 and summe["held-karp"] == n13 + kaemme + prisma == 2310             # darin einmal das Prisma mit 15 Orten
    assert meta["kaemme"]["gegenproben"]["kamm-dynamik"] == 9
    n16 = sum(z["instanzen"] for z in T["leiter"].values() if z["n"] <= 16)
    assert n16 == 1000 == meta["leiter"]["gegenproben"]["aufzaehlen"]
    assert all(sum(z["beleg"].values()) == z["instanzen"] for z in T["leiter"].values())
    belege = {}
    for z in T["leiter"].values():
        for art, zahl_ in z["beleg"].items():
            belege[art] = belege.get(art, 0) + zahl_
    assert belege == {"dualschranke": 1311, "held-karp": 25, "branch-and-bound": 144}          # nirgends „nur MILP“
    steht("1.311 durch die nachgerechnete Dualschranke, 25 durch die Dynamik über Teilmengen, 144 durch das eigene Branch-and-Bound")
    steht("3.090 Instanzen")
    steht("(2.309 Instanzen, dazu das Prisma mit 15 Orten)")
    steht("Auf allen 1.000 Instanzen der Leiter bis 16 Orte")


def test_falle_tabelle_und_saetze():
    for familie, n in (("karte", 12), ("karte", 30), ("karte", 50), ("matrix", 50), ("haufen", 50)):
        z = T["leiter"][f"{familie}-{n}"]
        k = z["komponentenregel"]
        spalten = tabellenzeile(f"{NAMEN[familie]} {n} ({z['instanzen']}) | {k['alle']['uebersehen']} / {k['eine']['uebersehen']}")
        assert spalten[2] == str(k["endwert_verschieden"]) and spalten[3] == f"{komma(k['eine']['abstand_max'], 2)} %"
    alle = sum(z["komponentenregel"]["alle"]["uebersehen"] for z in T["leiter"].values())
    eine = sum(z["komponentenregel"]["eine"]["uebersehen"] for z in T["leiter"].values())
    assert (alle, eine, sum(z["instanzen"] for z in T["leiter"].values())) == (92, 123, 1480)
    steht("auf 92 von 1.480 Instanzen etwas, mit „eine je Runde“ auf 123")
    assert all(z["komponentenregel"]["eine"]["uebersehen"] == z["komponentenregel"]["eine"]["unter_exakt"] for z in T["leiter"].values())
    assert any(z["komponentenregel"]["alle"]["uebersehen"] != z["komponentenregel"]["alle"]["unter_exakt"] for z in T["leiter"].values())
    assert all(z["komponentenregel"][m]["abgebrochen"] == 0 for z in T["leiter"].values() for m in ("alle", "eine"))
    schlimmste = {m: max(T["leiter"].values(), key=lambda z: z["komponentenregel"][m]["abstand_max"]) for m in ("alle", "eine")}
    for m, z in schlimmste.items():
        assert (z["familie"], z["n"]) == ("matrix", 30) and z["komponentenregel"][m]["abstand_max"] == pytest.approx(0.872, abs=1e-3)
    steht("höchstens 0,87 % des Optimums (Zufallsmatrix, 30 Orte)")
    assert T["leiter"]["karte-8"]["komponentenregel"]["alle"]["uebersehen"] == 0
    assert max(T["leiter"][f"{f}-12"]["komponentenregel"]["eine"]["uebersehen"] for f in ("karte", "matrix", "haufen")) <= 5


def test_nachgeladene_zeilen_und_zahnmengen():
    z = T["leiter"]["karte-50"]
    s, p = z["subtourzeilen"], z["zweimatchingzeilen"]
    assert tabellenzeile("Subtourzeilen bis zur Subtourschranke")[1:] == [
        f"{s['alle_median']:.0f} / {s['alle_max']}", f"{s['eine_median']:.0f} / {s['eine_max']}"]
    assert tabellenzeile("2-Matching-Zeilen, exakt")[1:] == [f"{komma(p['alle_median'], 1)} / {p['alle_max']}",
                                                             f"{p['eine_median']:.0f} / {p['eine_max']}"]
    steht(f"(Mittel {komma(p['alle_mittel'], 2)} bzw. {komma(p['eine_mittel'], 2)})")
    geteilt, gesamt = z["geteilte_zaehne"]
    assert (geteilt, gesamt) == (654, 1855)
    steht("bei Karte 50 in 654 von 1.855 Zeilen")
    alle = [sum(v["geteilte_zaehne"][i] for v in T["leiter"].values()) for i in (0, 1)]
    assert alle == [1039, 3627]
    steht("1.039 von 3.627")


def test_tabelle_gebrochene_loesungen():
    for familie in ("karte", "haufen", "matrix"):
        a, b = T["leiter"][f"{familie}-30"]["subtour_gebrochen"], T["leiter"][f"{familie}-50"]["subtour_gebrochen"]
        spalten = tabellenzeile(f"{NAMEN[familie]} 30 / 50")
        assert spalten[1] == f"{a['gebrochen']} von 100 / {b['gebrochen']} von 60"
        assert spalten[2] == f"{a['halb']} / {b['halb']}" and spalten[3] == f"{a['halb_kreise']} / {b['halb_kreise']}"
        assert a["unbekannt"] == b["unbekannt"] == 0
    m50 = T["leiter"]["matrix-50"]["subtour_gebrochen"]
    assert 0.5 < m50["halb"] / m50["gebrochen"] < 0.55


# ---------------------------------------------------------------- Kämme, Prisma, Hülle

def test_kammtabelle():
    for familie, n in (("matrix", 8), ("matrix", 10), ("matrix", 12), ("karte", 12), ("haufen", 12)):
        z = T["kaemme"][f"{familie}-{n}"]
        spalten = tabellenzeile(f"{NAMEN[familie]} {n} | {z['offen']} | {z['mit_kaemmen_geloest']}")
        assert spalten[3] == f"{komma(z['luecke_vorher_mittel'], 2)} → {komma(max(z['luecke_nachher_mittel'], 0), 2)} %"
        assert spalten[4] == f"{komma(max(z['luecke_nachher_max'], 0), 2)} %"
    assert all(T["kaemme"][k]["offen"] == 0 for k in ("karte-8", "karte-10", "haufen-10"))
    assert all(z["instanzen"] == 200 for z in T["kaemme"].values())
    ungeloest = {k: len(z["ungeloeste_seeds"]) for k, z in T["kaemme"].items() if z["ungeloeste_seeds"]}
    assert ungeloest == {"matrix-10": 2, "matrix-12": 7}
    # löserunabhängig: dort liegt der LP-Wert nach allen Kämmen unter dem Optimum
    assert all(T["kaemme"][k]["wertluecke_nach_kaemmen"] == v for k, v in ungeloest.items())
    steht("Auf allen 9 Instanzen, die offen bleiben (2 bei 10, 7 bei 12 Orten)")


def test_offene_kammfaelle_neu_gerechnet():
    """Endwerte nach allen Kämmen sind eindeutige LP-Werte: für jede gemeldete offene Instanz neu gerechnet."""
    for familie, n in (("matrix", 10), ("matrix", 12)):
        z = T["kaemme"][f"{familie}-{n}"]
        for seed in z["ungeloeste_seeds"]:
            e = M.kamm_instanz(I.zufallsinstanz(familie, n, seed))
            soll_wert, soll_opt = z["werte_ungeloest"][str(seed)]
            assert e is not None and e["wert_nachher"] == pytest.approx(soll_wert, abs=1e-4) and e["opt"] == soll_opt


def test_prisma_tabellen_neu_gerechnet():
    for k, zeile in zip((1, 2, 3, 4), T["prisma"]["wege"]):
        neu = M.prisma_zeile(k)
        for key in ("subtour", "zweimatching", "optimum"):
            assert neu[key] == pytest.approx(zeile[key], abs=1e-4)
        assert neu["optimum"] == 4 * k + 2
        if k >= 2:
            assert neu["subtour"] == pytest.approx(3 * k + 3)
        spalten = tabellenzeile(f"{k} ({neu['orte']})")
        assert spalten[1] == E.zahl(neu["subtour"]).replace(",00", "") and spalten[4] == str(neu["optimum"])
        assert spalten[5] == komma(neu["optimum"] / neu["subtour"], 2)
    assert T["prisma"]["wege"][2]["kaemme"] == pytest.approx(40 / 3)
    steht("c = 2 ergibt 9 gegen 10, c = 3 12 gegen 13, c = 5 18 gegen 19")
    werte = {(z["dreieck"], z["verbinder"]): (z["grad"], z["subtour"], z["zweimatching"], z["optimum"]) for z in T["prisma"]["kosten"]}
    for c, (a, b) in {2: (9, 10), 3: (12, 13), 5: (18, 19)}.items():
        assert werte[(c, 1)] == (pytest.approx(a), pytest.approx(a), pytest.approx(b), b)
    assert werte[(1, 4)] == (pytest.approx(6), pytest.approx(12), pytest.approx(12), 12)


def test_facettentabelle():
    h = T["huelle"]
    assert [h[n]["facetten"] for n in ("5", "6", "7")] == [20, 100, 3437]
    for n in (5, 6):
        neu = H.bericht(n)
        assert (neu["facetten"], neu["familien"], neu["dimension"], neu["ecken"]) == (
            h[str(n)]["facetten"], h[str(n)]["familien"], h[str(n)]["dimension"], h[str(n)]["ecken"])
    familien7 = h["7"]["familien"]
    assert sorted(familien7.values()) == sorted([21, 21, 35, 840, 1260, 1260]) and familien7["weitere"] == 1260
    assert sum(familien7.values()) == 3437
    for n, facetten in ((5, 20), (6, 100), (7, 3437), (8, 194187), (9, 42104442)):
        spalten = tabellenzeile(f"{n} | {n * (n - 3) // 2} | ")
        assert spalten[2] == tausender(math.factorial(n - 1) // 2) and spalten[3] == tausender(facetten)
        assert spalten[4] == str(H.subtourfacetten(n))
    assert sorted(h["6"]["familien"].values()) == [10, 15, 15, 60]
    assert sorted(h["5"]["familien"].values()) == [10, 10]
    steht("1.260 Kämme mit Zähnen aus 2, 2 und 3 Orten und **1.260 weitere**")
    steht("2⁴⁹ − 51 Subtourfacetten")
    assert H.subtourfacetten(50) == 2 ** 49 - 51
    assert T["leiter"]["karte-50"]["subtourzeilen"]["alle_median"] == 24


# ---------------------------------------------------------------- Netze und Spannbaum

def test_netztabelle():
    for n in (3, 4, 5, 16):
        b = NZ.netz_formulierung(NZ.batcher_netz(n), n)
        e = NZ.netz_formulierung(NZ.einfuege_netz(n), n)
        spalten = tabellenzeile(f"{n} | {tausender(NZ.permutaeder_facetten(n))} |")
        assert spalten[2] == f"{b.ungleichungen} / {b.variablen} / {b.gleichungen}" and spalten[3] == str(e.ungleichungen)


def test_gestrichener_komparator():
    voll = NZ.batcher_netz(4)
    assert [len(NZ.sortierte_permutationen(voll[:i] + voll[i + 1:], 4)[0]) for i in range(5)] == [14, 14, 12, 12, 16]
    steht("14, 14, 12, 12 bzw. 16 von 24 Permutationen (Komparator 1 bis 5)")
    e = NZ.einfuege_netz(4)
    zahlen = [len(NZ.sortierte_permutationen(e[:i] + e[i + 1:], 4)[0]) for i in range(len(e))]
    assert (min(zahlen), max(zahlen)) == (6, 20)
    steht("beim Einfüge-Netz mit 4 Werten sind es je nach Komparator zwischen 6 und 20")


def test_spannbaumzahlen():
    assert (NZ.spannbaum_facetten(10), NZ.martin_groessen(10)) == (1057, {"ungleichungen": 810, "variablen": 855, "gleichungen": 541})
    assert (NZ.spannbaum_facetten(4), NZ.martin_groessen(4)["ungleichungen"]) == (16, 36)
    assert all(NZ.martin_groessen(n)["ungleichungen"] >= NZ.spannbaum_facetten(n) for n in range(4, 10))
    steht("(810 statt 1.057, bei 855 statt 45 Variablen und 541 Gleichungen)")
    steht("bei n = 4 sind es 36 statt 16")


# ---------------------------------------------------------------- Nachrechnen mit Band

@pytest.mark.parametrize("familie, n, anzahl", [("karte", 8, 100), ("matrix", 12, 200)])
def test_reihe_neu_gerechnet(familie, n, anzahl):
    neu = M.leiter_reihe(familie, n, anzahl)
    alt = T["leiter"][f"{familie}-{n}"]
    for s, soll in alt["wert_optimal"].items():
        assert abs(neu["wert_optimal"][s] - soll) <= (3 if s.startswith("komp") else 0), s      # exakt außer bei der Heuristik
    for s in ("grad", "subtour", "exakt"):
        assert neu["luecke_mittel"][s] == pytest.approx(alt["luecke_mittel"][s], abs=1e-6)
    for s in ("grad", "subtour", "exakt"):
        assert abs(neu["geloest"][s] - alt["geloest"][s]) <= 3
    assert abs(neu["subtourzeilen"]["alle_median"] - alt["subtourzeilen"]["alle_median"]) <= 2


def test_handbeispiel_prisma_dreiecke_zwei():
    zeile = M.prisma_kosten_zeile(2, 1)
    assert (zeile["subtour"], zeile["zweimatching"], zeile["optimum"]) == (pytest.approx(9), pytest.approx(10), 10)
    assert len(H.rundreisen(6)) == 60
    steht("60 Rundreisen, Optimum 10")


def test_testzahl_steht_in_der_readme():
    import subprocess
    import sys
    lauf = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "--collect-only", "-q", "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)
    treffer = re.search(r"(\d+) tests? collected", lauf.stdout)
    assert treffer, lauf.stdout[-2000:]
    steht(f"({treffer.group(1)} Tests")


def test_jede_stufe_loest_mindestens_so_viele_wie_die_vorige_und_nur_karte_acht_ist_ganz_geloest():
    for k, z in T["leiter"].items():
        g = z["geloest"]
        assert g["grad"] <= g["subtour"] <= min(g["komp_alle"], g["komp_eine"]) and max(g["komp_alle"], g["komp_eine"]) <= g["exakt"], k
        assert (g["exakt"] == z["instanzen"]) == (k == "karte-8"), k
    steht("außer auf den Karten mit 8 Orten bleibt auch nach exaktem 2-Matching")


@pytest.mark.parametrize("satz", ["1 Sekunde", "15 Sekunden", "n von 5 bis 50", "Prisma mit 1 bis 4 Kanten je Weg", "Kämme bis 12 Orte"])
def test_app_grenzen_stehen_wie_in_den_konstanten(satz):
    import rpt_constants as C
    import rpt_kamm as K
    assert (C.N_MIN, C.N_MAX, C.K_MIN, C.K_MAX, K.MAX_N, C.ZEITGRENZE_OPT, C.ZEITGRENZE_BELEG, C.ZEITGRENZE_OPT_LANG) == (
        5, 50, 1, 4, 12, 1.0, 1.0, 15.0)
    steht(satz)
