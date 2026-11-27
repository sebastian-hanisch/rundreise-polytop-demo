"""Rechnet die Messreihen und schreibt rpt_tabellen.json; mit --check wird neu gerechnet und verglichen.

Aufruf (vom Repo-Ordner):
  python tools/rpt_make_tables.py                    alle Teile rechnen und schreiben
  python tools/rpt_make_tables.py --check            alle Teile neu rechnen und mit der Datei vergleichen
  python tools/rpt_make_tables.py --teil leiter      nur ein Teil (leiter, kaemme, prisma, huelle), mit oder ohne --check
  … --ohne-cpsat                                     ohne die CP-SAT-Gegenprobe (deutlich schneller; nur zum Nachrechnen mit --check)
Teile: leiter (Stufen und Regeln, 13 Reihen), kaemme (alle Kämme bis 12 Orte, 8 Reihen), prisma (Prisma-Familie),
huelle (ganze Hülle bei 5, 6 und 7 Orten; n = 7 braucht etwa zwei Minuten). Mit CP-SAT dauert alles zusammen rund eine halbe Stunde.

Gegenproben während der Rechnung: JEDES Referenzoptimum gegen einen zweiten Löser (CP-SAT mit AddCircuit aus ortools, dazu die für jede
Rundreise gültigen Subtourzeilen; Status „optimal“),
bis 13 Orte zusätzlich gegen eine Dynamik über Teilmengen (Held und Karp), bis 16 Orte der Endwert nach exaktem 2-Matching gegen eine Trennung
durch Aufzählen aller Griffe, und an jeder Instanz, die auch nach allen Kämmen offen ist, eine zweite Kammsuche ohne Beschneidung (Dynamik über
Teilmengen). Weicht ein Löser ab, bricht der Lauf mit „ABWEICHUNG“ ab. Zahlen nachgeladener Zeilen, gebrochene Ecken und die Komponentenregel
hängen an der Ecke, die HiGHS liefert; auf einem anderen System können sie abweichen."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import ortools
import scipy
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rpt_huelle as H           # noqa: E402
import rpt_kamm as K             # noqa: E402
import rpt_leiter as L           # noqa: E402
import rpt_messreihen as M       # noqa: E402
import rpt_referenz as R         # noqa: E402
import rpt_zweimatching as ZM    # noqa: E402

DATEI = ROOT / "rpt_tabellen.json"
TEILE = ("leiter", "kaemme", "prisma", "huelle")
HELD_KARP_BIS = 13
CPSAT_WORKER = 8
MIT_CPSAT = {"an": True}
AUFZAEHLEN_BIS = 16


def subtourmengen(inst) -> list:
    """Die Ortsmengen der Subtourzeilen bis zur Subtourschranke (als Hilfe für CP-SAT)."""
    m = L.neues_modell(inst)
    L.grad_stufe(m)
    L.subtour_stufe(m)
    return [schluessel[1] for schluessel in m.schluessel if schluessel[0] == "S"]


def cpsat_optimum(n, kanten, kosten, mengen=(), zeitgrenze: float = 900.0) -> int:
    """Zweiter Löser: CP-SAT mit AddCircuit (je Verbindung zwei Bögen). `mengen`: Ortsmengen S, für die zusätzlich x(δ(S)) >= 2 gefordert wird.
    Das gilt für jede Rundreise, wenn S eine echte nichtleere Teilmenge ist (genau das wird geprüft); die Zeilen beschleunigen nur den Beweis.
    Bricht ab, wenn CP-SAT nicht „optimal“ meldet."""
    modell = cp_model.CpModel()
    boegen, kosten_terme, je_kante = [], [], []
    for j, (u, v) in enumerate(kanten):
        hin, her = modell.new_bool_var(f"h{j}"), modell.new_bool_var(f"z{j}")
        boegen += [(u, v, hin), (v, u, her)]
        kosten_terme += [int(kosten[j]) * hin, int(kosten[j]) * her]
        je_kante.append((hin, her))
    modell.add_circuit(boegen)
    for s in mengen:
        s = set(s)
        if not 0 < len(s) < n or not s <= set(range(n)):
            raise ValueError(f"keine echte Teilmenge: {sorted(s)}")
        modell.add(sum(a + b for (u, v), (a, b) in zip(kanten, je_kante) if (u in s) != (v in s)) >= 2)
    modell.minimize(sum(kosten_terme))
    loeser = cp_model.CpSolver()
    loeser.parameters.max_time_in_seconds = zeitgrenze
    loeser.parameters.num_workers = CPSAT_WORKER
    loeser.parameters.linearization_level = 2       # stärkere LP-Relaxierung; ohne sie fand CP-SAT bei einzelnen 50er-Karten lange keinen Beweis
    status = loeser.solve(modell)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT nicht fertig: {loeser.status_name(status)}")
    return int(round(loeser.objective_value))


def pruefer_fuer(pruefungen: list):
    """Prüft jedes gemeldete Optimum mit CP-SAT, bis HELD_KARP_BIS Orte zusätzlich mit der Dynamik über Teilmengen."""
    def pruefen(inst, wert):
        if MIT_CPSAT["an"]:
            zweit = cpsat_optimum(inst.n, inst.kanten, inst.kosten, subtourmengen(inst))
            if zweit != wert:
                raise AssertionError(f"ABWEICHUNG CP-SAT {zweit} gegen Referenz {wert}: {inst.familie} {inst.n} {inst.kosten[:5]}")
            pruefungen.append("cp-sat")
        if inst.n <= HELD_KARP_BIS:
            dritt = R.held_karp(inst.n, inst.kanten, inst.kosten)
            if dritt != wert:
                raise AssertionError(f"ABWEICHUNG Held-Karp {dritt} gegen Referenz {wert}: {inst.familie} {inst.n}")
            pruefungen.append("held-karp")
    return pruefen


def aufzaehl_endwert(inst) -> float:
    """Endwert nach 2-Matching, wenn die verletzte Zeile durch Aufzählen aller Griffe gefunden wird."""
    m = L.neues_modell(inst)
    L.grad_stufe(m)
    L.subtour_stufe(m)
    for _ in range(500):
        L.subtour_stufe(m)
        w, griff = ZM.staerkste_durch_aufzaehlen(inst.n, inst.kanten, m.x)
        if w >= 1.0 - ZM.TOL:
            return m.wert
        zaehne = ZM.beste_zaehne(inst.kanten, m.x, griff)[1]
        if not m.add(("B", griff, zaehne), *ZM.zeile(inst.kanten, griff, zaehne)):
            raise RuntimeError("verletzte Zeile schon im LP")
        m.loese()
    raise RuntimeError("Aufzählen nicht fertig")


def teil_leiter(pruefungen: list) -> dict:
    ergebnis = {}
    for familie, n, anzahl in M.LEITER_REIHEN:
        t0 = time.time()
        einzeln = []
        pruefen = pruefer_fuer(pruefungen)
        for i in range(anzahl):
            inst = M.instanz(familie, n, i)
            e = M.leiter_instanz(inst, pruefen)
            if n <= AUFZAEHLEN_BIS:
                assert abs(aufzaehl_endwert(inst) - e["wert"]["exakt"]) < 1e-4, (familie, n, i)
                pruefungen.append("aufzaehlen")
            einzeln.append(e)
        schluessel = M.reihen_schluessel(familie, n)
        ergebnis[schluessel] = M.fasse_leiter_zusammen(familie, n, einzeln)
        g = ergebnis[schluessel]["geloest"]
        print(f"leiter {schluessel}: {g} ({time.time() - t0:.0f} s)", flush=True)
    return ergebnis


def staerkste_kammverletzung_dp(n, kanten, x) -> float:
    """Zweiter Rechenweg ohne Beschneidung und ohne Subtour-Voraussetzung: je Griff die kleinste Summe von p disjunkten zulässigen
    Zahnschnitten (Dynamik über Teilmengen, 3^n je Stufe). Rückgabe: die größte Verletzung (<= 0: kein Kamm ist verletzt)."""
    c = K.schnittwerte(n, kanten, x)
    voll = (1 << n) - 1
    mengen, zaehne = [], []
    for s in range(1 << n):
        t = s
        while t:
            mengen.append(s)
            zaehne.append(t)
            t = (t - 1) & s
    mengen, zaehne = np.array(mengen), np.array(zaehne)
    anfaenge = np.flatnonzero(np.r_[True, mengen[1:] != mengen[:-1]])
    alle = np.arange(1 << n)
    rest = mengen ^ zaehne
    beste = -np.inf
    for h in range(1, 1 << (n - 1)):
        zulaessig = ((alle & h) != 0) & ((alle & ~h & voll) != 0)
        kosten = np.where(zulaessig, c, np.inf)
        f = np.zeros(1 << n)
        for p in range(1, n // 2 + 1):
            g = np.full(1 << n, np.inf)
            g[mengen[anfaenge]] = np.minimum.reduceat(kosten[zaehne] + f[rest], anfaenge)
            f = g
            if p >= 3 and p % 2 == 1:
                beste = max(beste, 3 * p + 1 - c[h] - f[voll])
    return float(beste)


def teil_kaemme(pruefungen: list) -> dict:
    ergebnis = {}
    for familie, n, anzahl in M.KAMM_REIHEN:
        t0 = time.time()
        schluessel = M.reihen_schluessel(familie, n)
        ergebnis[schluessel] = M.kamm_reihe(familie, n, anzahl, pruefer_fuer(pruefungen))
        r = ergebnis[schluessel]
        # an jeder offenen Instanz: die Endlösung nach allen Kämmen verletzt auch nach der Dynamik keinen Kamm
        for seed in r["ungeloeste_seeds"]:
            e = M.kamm_instanz(M.I.zufallsinstanz(familie, n, seed))
            assert staerkste_kammverletzung_dp(n, M.I.paare(n), e["x"]) <= 1e-5, (familie, n, seed)
            pruefungen.append("kamm-dynamik")
        print(f"kaemme {schluessel}: offen {r['offen']}, mit Kämmen gelöst {r['mit_kaemmen_geloest']} ({time.time() - t0:.0f} s)", flush=True)
    return ergebnis


def teil_prisma(pruefungen: list) -> dict:
    wege = []
    for k in (1, 2, 3, 4):
        zeile = M.prisma_zeile(k, pruefer_fuer(pruefungen))
        inst = M.I.prisma(k)
        if inst.n > HELD_KARP_BIS:
            assert R.held_karp(inst.n, inst.kanten, inst.kosten) == zeile["optimum"]
            pruefungen.append("held-karp")
        wege.append(zeile)
    kosten = [M.prisma_kosten_zeile(c, v, pruefer_fuer(pruefungen)) for c, v in M.PRISMA_KOSTEN]
    print(f"prisma: {[(z['k'], z['subtour'], z['zweimatching'], z['optimum']) for z in wege]}", flush=True)
    return {"wege": wege, "kosten": kosten}


def teil_huelle(pruefungen: list) -> dict:
    ergebnis = {}
    for n in (5, 6, 7):
        t0 = time.time()
        huelle, punkte = H.huelle_rundreisen(n, max_strahlen=200_000, max_paare=2_000_000_000)
        ergebnis[str(n)] = {"n": n, "dimension": huelle.dimension, "ecken": len(punkte), "facetten": len(huelle.facetten),
                            "familien": H.einordnen(n, huelle, punkte)}
        print(f"huelle n={n}: {ergebnis[str(n)]['facetten']} Facetten ({time.time() - t0:.0f} s)", flush=True)
    return ergebnis


RECHNER = {"leiter": teil_leiter, "kaemme": teil_kaemme, "prisma": teil_prisma, "huelle": teil_huelle}


def vergleiche(alt, neu, pfad: str, abweichungen: list):
    """Struktur, Zeichenketten und ganze Zahlen exakt, Gleitkommazahlen mit relativer Toleranz 1e-6."""
    if isinstance(alt, dict) and isinstance(neu, dict):
        if set(alt) != set(neu):
            abweichungen.append(f"{pfad}: Schlüssel {sorted(set(alt) ^ set(neu))}")
        for k in set(alt) & set(neu):
            vergleiche(alt[k], neu[k], f"{pfad}.{k}", abweichungen)
    elif isinstance(alt, list) and isinstance(neu, list):
        if len(alt) != len(neu):
            abweichungen.append(f"{pfad}: Länge {len(alt)} / {len(neu)}")
        for i, (a, b) in enumerate(zip(alt, neu)):
            vergleiche(a, b, f"{pfad}[{i}]", abweichungen)
    elif isinstance(alt, float) or isinstance(neu, float):
        if alt is None or neu is None or abs(alt - neu) > 1e-6 * max(1.0, abs(alt)):
            abweichungen.append(f"{pfad}: {alt} / {neu}")
    elif alt != neu:
        abweichungen.append(f"{pfad}: {alt} / {neu}")


def main(argv):
    teile = [argv[argv.index("--teil") + 1]] if "--teil" in argv else list(TEILE)
    pruefen = "--check" in argv
    MIT_CPSAT["an"] = "--ohne-cpsat" not in argv
    if not MIT_CPSAT["an"] and not pruefen:
        print("--ohne-cpsat ist nur mit --check erlaubt: die geschriebene Datei braucht die Gegenprobe")
        return 2
    daten = json.loads(DATEI.read_text(encoding="utf-8")) if DATEI.exists() else {}
    pruefungen = []
    abweichungen = []
    for teil in teile:
        t0 = time.time()
        vorher = len(pruefungen)
        neu = json.loads(json.dumps(RECHNER[teil](pruefungen)))
        if pruefen:
            vergleiche(daten.get(teil), neu, teil, abweichungen)
        else:
            daten[teil] = neu
            zaehlung_teil = {name: pruefungen[vorher:].count(name) for name in sorted(set(pruefungen[vorher:]))}
            daten.setdefault("meta", {})[teil] = {"scipy": scipy.__version__, "numpy": np.__version__, "ortools": ortools.__version__,
                                                  "loeser": "HiGHS (scipy linprog highs-ds, milp), Gegenprobe CP-SAT",
                                                  "gegenproben": zaehlung_teil}
            DATEI.write_text(json.dumps(daten, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"Teil {teil} fertig in {time.time() - t0:.0f} s", flush=True)
    zaehlung = {name: pruefungen.count(name) for name in sorted(set(pruefungen))}
    if pruefen:
        for a in abweichungen[:40]:
            print("ABWEICHUNG", a)
        print(f"--check: {len(abweichungen)} Abweichungen in {', '.join(teile)}; Gegenproben: {zaehlung}")
        return 1 if abweichungen else 0
    print(f"geschrieben: {DATEI.name}; Gegenproben: {zaehlung}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
