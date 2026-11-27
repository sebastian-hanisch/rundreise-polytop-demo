"""Konstanten: Regler-Grenzen, Live-Grenzen, Farben, Namen, Beschreibungen und Presets."""

import rpt_leiter as L

# Instanzregler. Laufzeit am Maximum n = 50 (Runtime-venv, 150 Karten): Subtour- und 2-Matching-Stufe im Median unter 0,2 s, in seltenen Fällen
# knapp 2 s; für das Optimum bekommt in der App das MILP höchstens ZEITGRENZE_OPT und ein zweiter Beleg höchstens ZEITGRENZE_BELEG Sekunden.
N_MIN, N_MAX, DEFAULT_N = 5, 50, 30
SEED_MAX, DEFAULT_SEED = 999999, 1
K_MIN, K_MAX = 1, 4                     # Wege im Prisma: 6, 9, 12, 15 Orte
KOSTEN_MIN, KOSTEN_MAX = 1, 5
ZEITGRENZE_OPT = 1.0                    # Sekunden für das MILP beim ersten Anzeigen
ZEITGRENZE_BELEG = 1.0                  # Sekunden für das eigene Branch-and-Bound als zweiten Beleg
ZEITGRENZE_OPT_LANG = 15.0              # Sekunden nach Klick auf „länger suchen“

BESCHRIFTEN_BIS = 16                    # Brüche an die Kanten schreiben bis zu so vielen Orten
NAMEN_BIS = 30                          # Ortsnamen in die Zeichnung bis zu so vielen Orten
HILFSGRAPH_MAX_N = 12

FAMILIEN = ("karte", "haufen", "matrix", "prisma")
ZUFALLSFAMILIEN = ("karte", "haufen", "matrix")
FAMILIEN_NAMEN = {"karte": "Karte", "haufen": "Haufen", "matrix": "Zufallsmatrix", "prisma": "Prisma (Beispiele)"}
FAMILIEN_HILFE = {
    "karte": "Orte mit ganzzahligen Koordinaten von 0 bis 1000, Kosten = gerundeter Abstand.",
    "haufen": "Vier zufällige Mittelpunkte; jeder Ort liegt bis zu 100 je Koordinate neben einem davon. Kosten = gerundeter Abstand.",
    "matrix": "Jedes Paar bekommt zufällige Kosten von 1 bis 1000 (nicht metrisch). Die Lage in der Zeichnung hat keine Bedeutung.",
    "prisma": "Zwei Dreiecke, drei Wege dazwischen; übrige Paare kosten den kürzesten Weg im Prisma.",
}

STUFEN = ("grad", "subtour", "zm")
STUFEN_NAMEN = {"grad": "Nur Gradgleichungen", "subtour": "+ Subtourzeilen", "zm": "+ 2-Matching-Ungleichungen"}
TABELLEN_STUFEN = ("grad", "subtour", "komp_alle", "komp_eine", "exakt")
TABELLEN_STUFEN_NAMEN = {"grad": "Grad-LP", "subtour": "+ Subtour", "komp_alle": "+ 2-Matching, Komponentenregel (alle je Runde)",
                         "komp_eine": "+ 2-Matching, Komponentenregel (eine je Runde)", "exakt": "+ 2-Matching, exakt (Padberg–Rao)"}

MODI = (L.ALLE, L.EINE)
MODUS_NAMEN = {L.ALLE: "alle je Runde", L.EINE: "eine je Runde"}
REGELN = (L.KOMPONENTEN, L.PADBERG_RAO)
REGEL_NAMEN = {L.KOMPONENTEN: "Komponentenregel", L.PADBERG_RAO: "Padberg–Rao"}

VERFAHREN_TABELLE = (
    ("Aufzählen", "alle Griffe H prüfen, zu jedem die beste ungerade Zahnmenge", "exakt, nur klein (hier als Gegenprobe bis 16 Orte)"),
    ("Padberg–Rao", "kleinster ungerader Schnitt unter 1 im geteilten Hilfsgraphen, gesucht unter den Schnitten eines Gomory-Hu-Baums",
     "exakt, polynomiell"),
    ("Komponentenregel", "Griffe sind nur die Zusammenhangskomponenten der gebrochenen Kanten, je mit der besten Zahnmenge",
     "Heuristik: findet sie nichts, kann trotzdem etwas verletzt sein"),
    ("alle je Runde / eine je Runde", "alle gefundenen verletzten Ungleichungen nachladen oder nur die am stärksten verletzte (Gleichstand: kleinerer "
     "Griff, dann Knotennummern; bei der Komponentenregel die erste Komponente)", "Auswahlregel, gilt für beide Trennungen und für Subtour"),
)
REGEL_BESCHREIBUNG = {
    L.KOMPONENTEN: "Griffe sind nur die Zusammenhangskomponenten der Kanten mit gebrochenem Wert; jede bekommt ihre beste ungerade Zahnmenge. "
                   "Eine Heuristik: findet sie nichts, kann trotzdem eine 2-Matching-Ungleichung verletzt sein.",
    L.PADBERG_RAO: "Jede Trägerkante wird im Hilfsgraphen geteilt, und gesucht ist ein ungerader Schnitt unter 1 unter den Schnitten eines "
                   "Gomory-Hu-Baums. Exakt: findet es nichts, ist keine 2-Matching-Ungleichung verletzt (bis auf die Toleranz).",
}

COLOR_EINS = "#33475b"
COLOR_BRUCH = "#d1495b"
COLOR_GRUND = "rgba(140,140,140,0.35)"
COLOR_KNOTEN = "#ffffff"
COLOR_RAND = "#33475b"
COLOR_GRIFF = "#2e86c1"
COLOR_MENGE = "#e08e0b"
COLOR_ZAHN = "#7d3c98"
COLOR_ZAEHNE = ("#7d3c98", "#117864", "#a04000", "#5d6d7e", "#9a7d0a")
STUFEN_FARBE = {"grad": "#7f8c8d", "subtour": "#e08e0b", "zweimatching": "#2e86c1", "kamm": "#7d3c98"}
STUFEN_KURZ = {"grad": "Grad-LP", "subtour": "Subtour", "zweimatching": "2-Matching", "kamm": "Kämme"}

KAMM_BEISPIELE = ("aktuell", "prisma2", "prisma3", "offen")
KAMM_BEISPIEL_NAMEN = {"aktuell": "die Instanz links", "prisma2": "Prisma mit Wegen aus 2 Kanten (9 Orte)",
                       "prisma3": "Prisma mit Wegen aus 3 Kanten (12 Orte)",
                       "offen": "vorgemerkter Fall, den auch alle Kämme nicht lösen (Zufallsmatrix, 12 Orte)"}
OFFENES_BEISPIEL = ("matrix", 12, 33398)      # Familie, n, Zufalls-Seed; aus der Messreihe der Kämme

NETZ_N_MAX = 16
NETZ_NAMEN = {"batcher": "Batcher (Merge-Exchange)", "einfuegen": "Sortieren durch Einfügen"}
NETZ_BESCHREIBUNG = {
    "batcher": "Batchers Merge-Exchange-Netz; es braucht bei 16 Drähten 63 Komparatoren.",
    "einfuegen": "Jeder neue Wert wandert durch benachbarte Vergleiche an seinen Platz: n(n−1)/2 Komparatoren.",
}

PRESET_KEYS_ORDER = ("familie", "n", "seed", "k", "dreieck", "weg")


def _preset(familie, n=DEFAULT_N, seed=DEFAULT_SEED, k=1, dreieck=1, weg=1):
    return {"familie": familie, "n": n, "seed": seed, "k": k, "dreieck": dreieck, "weg": weg}


PRESETS = {
    "Prisma: Subtour fehlt": _preset("prisma", k=1, dreieck=1, weg=4),
    "Prisma: 2-Matching fehlt": _preset("prisma", k=1, dreieck=2, weg=1),
    "Karte 30": _preset("karte", 30, 34074),
    "Haufen 30": _preset("haufen", 30, 36000),
    "Zufallsmatrix 12: Kämme reichen nicht": _preset("matrix", 12, 33398),
    "Lange Wege (12)": _preset("prisma", k=3, dreieck=1, weg=1),
}
PRESET_HELP = {
    "Prisma: Subtour fehlt": "6 Orte, Dreiecke je 1, Verbinder 4: das Grad-LP wählt zwei Dreiecke, eine Subtourzeile genügt.",
    "Prisma: 2-Matching fehlt": "6 Orte, Dreiecke 2, Verbinder 1: alle Subtourzeilen sind erfüllt, erst eine 2-Matching-Ungleichung schließt.",
    "Karte 30": "30 Orte auf einer Karte: jede Stufe lädt etwas nach, und die Komponentenregel bleibt vor dem Ziel stehen.",
    "Haufen 30": "30 Orte in vier Haufen: das Grad-LP liegt weit darunter, die Subtourzeilen schließen alles.",
    "Zufallsmatrix 12: Kämme reichen nicht": "12 Orte mit zufälligen Kosten: ein Kamm hebt den LP-Wert, aber auch danach bleibt eine Lücke "
                                             "(Experiment 4).",
    "Lange Wege (12)": "Prisma mit Wegen aus 3 Kanten: alle Kämme sind erfüllt, und trotzdem liegt das LP unter dem Optimum.",
}
