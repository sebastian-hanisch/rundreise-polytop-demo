"""Die Messreihen der README. Jede Reihe ist durch Familie, n, Instanzzahl und Start-Seed festgelegt (Instanz i bekommt Basis + i).

Gerechnet wird mit denselben Funktionen wie in der App; tools/rpt_make_tables.py schreibt die Ergebnisse nach rpt_tabellen.json und lässt jedes
Optimum von einem zweiten Löser prüfen (`pruefer(inst, wert)`)."""

import numpy as np

import rpt_instanzen as I
import rpt_kamm as K
import rpt_leiter as L
import rpt_lp as LP
import rpt_referenz as R
import rpt_zweimatching as ZM

# Basis der Zufalls-Seeds je Familie; die Reihe (Familie, n) benutzt Basis + 100·n, Basis + 100·n + 1, …
BASIS = {"karte": 31000, "matrix": 32000, "haufen": 33000}
LEITER_REIHEN = (("karte", 8, 100), ("karte", 12, 200), ("matrix", 12, 200), ("haufen", 12, 200), ("karte", 16, 100), ("matrix", 16, 100),
                 ("haufen", 16, 100), ("karte", 30, 100), ("matrix", 30, 100), ("haufen", 30, 100), ("karte", 50, 60), ("matrix", 50, 60),
                 ("haufen", 50, 60))
KAMM_REIHEN = (("karte", 8, 200), ("matrix", 8, 200), ("karte", 10, 200), ("matrix", 10, 200), ("haufen", 10, 200), ("karte", 12, 200),
               ("matrix", 12, 200), ("haufen", 12, 200))
OPT_TOL = 1e-4
REFERENZ_ZEITGRENZE = 120.0
BELEG_ZEITGRENZE = 30.0


def referenz(inst: I.Instanz, modell, pruefer=None) -> R.Optimum:
    """Kürzeste Rundreise der Messreihen; ohne Wert wird abgebrochen. `pruefer(inst, wert)` ist der zweite, unabhängige Löser."""
    opt = R.optimum(inst.n, inst.kanten, inst.kosten, modell, REFERENZ_ZEITGRENZE, BELEG_ZEITGRENZE)
    if opt.wert is None:
        raise RuntimeError(f"Referenz nicht bestimmt ({opt.grund}): {inst.familie} {inst.n}")
    if pruefer is not None:
        pruefer(inst, opt.wert)
    return opt


def seed_von(familie: str, n: int, i: int) -> int:
    return BASIS[familie] + 100 * n + i


def instanz(familie: str, n: int, i: int) -> I.Instanz:
    return I.zufallsinstanz(familie, n, seed_von(familie, n, i))


def reihen_schluessel(familie: str, n: int) -> str:
    return f"{familie}-{n}"


def ganz_und_optimal(wert: float, x, opt: int) -> bool:
    return bool(LP.ist_ganzzahlig(x) and abs(wert - opt) < OPT_TOL)


def luecke_prozent(wert: float, opt: int) -> float:
    """(Optimum − LP-Wert) / Optimum in Prozent."""
    return 100.0 * (opt - wert) / opt


def leiter_instanz(inst: I.Instanz, pruefer=None) -> dict:
    """Alle Stufen und Regeln auf einer Instanz."""
    haupt = L.leiter(inst)                                      # Subtour alle je Runde, Padberg-Rao alle je Runde
    eine_sub = L.neues_modell(inst)
    L.grad_stufe(eine_sub)
    sub_eine = L.subtour_stufe(eine_sub, L.EINE)
    pr_eine_modell = haupt.modell_sub.kopie()
    pr_eine = L.zweimatching_stufe(pr_eine_modell, L.PADBERG_RAO, L.EINE)
    komp = {}
    for modus in (L.ALLE, L.EINE):
        m = haupt.modell_sub.kopie()
        v = L.zweimatching_stufe(m, L.KOMPONENTEN, modus)
        komp[modus] = (m, v)
    opt = referenz(inst, haupt.modell_zm, pruefer)
    if haupt.status() != L.FERTIG or pr_eine.status != L.FERTIG:
        raise RuntimeError(f"Leiter nicht fertig: {inst}")
    if abs(eine_sub.wert - haupt.wert_subtour) > OPT_TOL:
        raise RuntimeError("der Subtourwert hängt an der Regel")
    if abs(pr_eine_modell.wert - haupt.wert_zweimatching) > OPT_TOL:
        raise RuntimeError("der Endwert nach exaktem 2-Matching hängt an der Regel")
    stufen = {
        "grad": (haupt.wert_grad, haupt.start.x),
        "subtour": (haupt.wert_subtour, haupt.modell_sub.x),
        "komp_alle": (komp[L.ALLE][0].wert, komp[L.ALLE][0].x),
        "komp_eine": (komp[L.EINE][0].wert, komp[L.EINE][0].x),
        "exakt": (haupt.wert_zweimatching, haupt.modell_zm.x),
    }
    ergebnis = {"opt": opt.wert, "beleg": opt.status, "geloest": {k: ganz_und_optimal(w, x, opt.wert) for k, (w, x) in stufen.items()},
                "wert_optimal": {k: abs(w - opt.wert) < OPT_TOL for k, (w, x) in stufen.items()},
                "luecke": {k: luecke_prozent(w, opt.wert) for k, (w, x) in stufen.items()},
                "wert": {k: w for k, (w, x) in stufen.items()}}
    for k, (w, _) in stufen.items():
        if w > opt.wert + OPT_TOL:
            raise RuntimeError(f"LP-Wert über dem Optimum ({k})")
    ergebnis["subtourzeilen"] = {"alle": haupt.subtour.zeilen, "eine": sub_eine.zeilen, "runden_alle": haupt.subtour.runden}
    ergebnis["pr_zeilen"] = {"alle": haupt.zweimatching.zeilen, "eine": pr_eine.zeilen, "runden_alle": haupt.zweimatching.runden}
    # Zahnzahlen und Disjunktheit der nachgeladenen 2-Matching-Zeilen (beide Padberg-Rao-Läufe)
    zaehne, geteilt, gesamt = [], 0, 0
    for verlauf in (haupt.zweimatching, pr_eine):
        for schritt in verlauf.schritte:
            for z in schritt.neu:
                if isinstance(z, ZM.Zeile):
                    gesamt += 1
                    zaehne.append(len(z.zaehne))
                    geteilt += not ZM.zaehne_disjunkt(inst.kanten, z.zaehne)
    ergebnis["zaehne"] = zaehne
    ergebnis["geteilt"] = (geteilt, gesamt)
    # Komponentenregel: endet sie, obwohl noch etwas verletzt ist?
    ergebnis["komp"] = {}
    for modus, (m, v) in komp.items():
        rest = ZM.trenne_padberg_rao(inst.n, inst.kanten, m.x)
        ergebnis["komp"][modus] = {"fertig": v.status == L.FERTIG, "uebersehen": bool(rest),
                                   "unter_exakt": m.wert < haupt.wert_zweimatching - OPT_TOL,
                                   "abstand": 100.0 * (haupt.wert_zweimatching - m.wert) / opt.wert}
    ergebnis["komp_endwert_verschieden"] = abs(komp[L.ALLE][0].wert - komp[L.EINE][0].wert) > OPT_TOL
    x1 = haupt.modell_sub.x
    ergebnis["subtour_struktur"] = LP.struktur(inst.n, inst.kanten, x1)
    ergebnis["zm_struktur"] = LP.struktur(inst.n, inst.kanten, haupt.modell_zm.x)
    return ergebnis


def _median(werte) -> float:
    return float(np.median(werte))


def fasse_leiter_zusammen(familie: str, n: int, ergebnisse: list) -> dict:
    """Die Tabellenzeile einer Reihe: Stufen, Lücken, nachgeladene Zeilen, Komponentenregel und gebrochene Lösungen."""
    anzahl = len(ergebnisse)
    stufen = ("grad", "subtour", "komp_alle", "komp_eine", "exakt")
    belege = [e["beleg"] for e in ergebnisse]
    zeile = {"familie": familie, "n": n, "instanzen": anzahl, "basis": seed_von(familie, n, 0),
             "beleg": {b: belege.count(b) for b in sorted(set(belege))},
             "geloest": {s: sum(e["geloest"][s] for e in ergebnisse) for s in stufen},
             "wert_optimal": {s: sum(e["wert_optimal"][s] for e in ergebnisse) for s in stufen},
             "luecke_mittel": {s: float(np.mean([e["luecke"][s] for e in ergebnisse])) for s in stufen},
             "luecke_max": {s: float(max(e["luecke"][s] for e in ergebnisse)) for s in stufen}}
    sub = [e["subtourzeilen"] for e in ergebnisse]
    zeile["subtourzeilen"] = {"alle_median": _median([s["alle"] for s in sub]), "alle_max": max(s["alle"] for s in sub),
                              "eine_median": _median([s["eine"] for s in sub]), "eine_max": max(s["eine"] for s in sub),
                              "runden_median": _median([s["runden_alle"] for s in sub])}
    pr = [e["pr_zeilen"] for e in ergebnisse]
    zeile["zweimatchingzeilen"] = {"eine_median": _median([p["eine"] for p in pr]), "eine_max": max(p["eine"] for p in pr),
                                   "alle_median": _median([p["alle"] for p in pr]), "alle_max": max(p["alle"] for p in pr),
                                   "eine_mittel": float(np.mean([p["eine"] for p in pr])),
                                   "alle_mittel": float(np.mean([p["alle"] for p in pr]))}
    zaehne = [z for e in ergebnisse for z in e["zaehne"]]
    zeile["zahnzahlen"] = {str(z): zaehne.count(z) for z in sorted(set(zaehne))}
    zeile["geteilte_zaehne"] = [sum(e["geteilt"][0] for e in ergebnisse), sum(e["geteilt"][1] for e in ergebnisse)]
    zeile["komponentenregel"] = {}
    for modus in ("alle", "eine"):
        k = [e["komp"][modus] for e in ergebnisse]
        zeile["komponentenregel"][modus] = {
            "geloest": zeile["geloest"]["komp_" + modus], "uebersehen": sum(x["uebersehen"] for x in k),
            "unter_exakt": sum(x["unter_exakt"] for x in k), "abstand_max": float(max(x["abstand"] for x in k)),
            "abgebrochen": sum(not x["fertig"] for x in k)}
    zeile["komponentenregel"]["endwert_verschieden"] = sum(e["komp_endwert_verschieden"] for e in ergebnisse)
    struktur = [e["subtour_struktur"] for e in ergebnisse]
    gebrochen = [s for s in struktur if s[0] != "ganzzahlig"]
    laengen = [l for s in gebrochen if s[0] == "halb" for l in s[1][1]]
    zeile["subtour_gebrochen"] = {
        "gebrochen": len(gebrochen), "halb": sum(s[0] == "halb" for s in gebrochen),
        "halb_kreise": sum(s[0] == "halb" and s[1][0] for s in gebrochen),
        "anders": sum(s[0] == "anders" for s in gebrochen), "unbekannt": sum(s[0] == "unbekannt" for s in gebrochen),
        "kreislaengen": {str(l): laengen.count(l) for l in sorted(set(laengen))}}
    zm = [e["zm_struktur"] for e in ergebnisse]
    zeile["zweimatching_gebrochen"] = {"gebrochen": sum(s[0] != "ganzzahlig" for s in zm), "halb": sum(s[0] == "halb" for s in zm)}
    return zeile


def leiter_reihe(familie: str, n: int, anzahl: int, pruefer=None) -> dict:
    return fasse_leiter_zusammen(familie, n, [leiter_instanz(instanz(familie, n, i), pruefer) for i in range(anzahl)])


def kamm_instanz(inst: I.Instanz, pruefer=None):
    """None, wenn exaktes 2-Matching schon alles löst; sonst die Lücke davor und danach und ob alle Kämme lösen."""
    modell = L.neues_modell(inst)
    L.grad_stufe(modell)
    L.subtour_stufe(modell)
    zm = L.zweimatching_stufe(modell, L.PADBERG_RAO, L.EINE)
    opt = referenz(inst, modell, pruefer)
    if zm.status != L.FERTIG:
        raise RuntimeError("2-Matching nicht fertig")
    if ganz_und_optimal(modell.wert, modell.x, opt.wert):
        return None
    vorher = modell.wert
    kv = L.kamm_stufe(modell)
    if kv.status != L.FERTIG or modell.wert > opt.wert + OPT_TOL:
        raise RuntimeError("Kammstufe nicht fertig oder LP-Wert über dem Optimum")
    zahngroessen = [tuple(sorted(len(t) for t in k[2])) for s in kv.schritte for k in s.neu if isinstance(k, tuple) and len(k) == 3
                    and not isinstance(k, ZM.Zeile)]
    return {"opt": opt.wert, "wert_vorher": vorher, "luecke_vorher": luecke_prozent(vorher, opt.wert), "wert_nachher": modell.wert,
            "luecke_nachher": luecke_prozent(modell.wert, opt.wert), "geloest": ganz_und_optimal(modell.wert, modell.x, opt.wert),
            "kaemme": kv.zeilen, "zahngroessen": zahngroessen, "x": tuple(modell.x)}


def kamm_reihe(familie: str, n: int, anzahl: int, pruefer=None) -> dict:
    offen = {}
    for i in range(anzahl):
        e = kamm_instanz(instanz(familie, n, i), pruefer)
        if e is not None:
            offen[seed_von(familie, n, i)] = e
    ungeloest = [s for s, e in offen.items() if not e["geloest"]]
    # löserunabhängig: nur die LP-Werte, nicht die gelieferte Ecke
    wertluecke_2m = [s for s, e in offen.items() if e["opt"] - e["wert_vorher"] > OPT_TOL]
    wertluecke_kamm = [s for s, e in offen.items() if e["opt"] - e["wert_nachher"] > OPT_TOL]
    return {"familie": familie, "n": n, "instanzen": anzahl, "basis": seed_von(familie, n, 0), "offen": len(offen),
            "wertluecke_nach_zweimatching": len(wertluecke_2m), "wertluecke_nach_kaemmen": len(wertluecke_kamm),
            "mit_kaemmen_geloest": len(offen) - len(ungeloest), "offene_seeds": sorted(offen), "ungeloeste_seeds": sorted(ungeloest),
            "luecke_vorher_mittel": float(np.mean([e["luecke_vorher"] for e in offen.values()])) if offen else None,
            "luecke_nachher_mittel": float(np.mean([e["luecke_nachher"] for e in offen.values()])) if offen else None,
            "luecke_nachher_max": float(max(e["luecke_nachher"] for e in offen.values())) if offen else None,
            "kaemme_median": _median([e["kaemme"] for e in offen.values()]) if offen else None,
            "kaemme_max": max((e["kaemme"] for e in offen.values()), default=None),
            "werte_ungeloest": {str(s): [offen[s]["wert_nachher"], offen[s]["opt"]] for s in ungeloest}}


def prisma_zeile(k: int, pruefer=None) -> dict:
    """Prisma mit Wegen aus k Kanten, alle Längen 1."""
    inst = I.prisma(k)
    lt = L.leiter(inst)
    opt = referenz(inst, lt.modell_zm, pruefer)
    kamm_wert = None
    if inst.n <= K.MAX_N:
        m = lt.modell_zm.kopie()
        L.kamm_stufe(m)
        kamm_wert = m.wert
    return {"k": k, "orte": inst.n, "grad": lt.wert_grad, "subtour": lt.wert_subtour, "zweimatching": lt.wert_zweimatching,
            "kaemme": kamm_wert, "optimum": opt.wert}


def prisma_kosten_zeile(dreieck: int, weg: int, pruefer=None) -> dict:
    """6 Orte, Dreieckskanten `dreieck`, Verbinder `weg`."""
    inst = I.prisma(1, dreieck, weg)
    lt = L.leiter(inst)
    opt = referenz(inst, lt.modell_zm, pruefer)
    return {"dreieck": dreieck, "verbinder": weg, "grad": lt.wert_grad, "subtour": lt.wert_subtour,
            "zweimatching": lt.wert_zweimatching, "optimum": opt.wert}


PRISMA_KOSTEN = ((1, 1), (2, 1), (3, 1), (5, 1), (1, 2), (1, 4))
