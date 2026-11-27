# Rundreise-Polytop: Subtour, 2-Matching, Kämme – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-rundreise-polytop-demo.streamlit.app/)** -->

Teil der „Konzepte“-Reihe „Fehlende Ungleichungen: Polyedrische Kombinatorik erklärt“ für die Website „Sebastian Hanisch – Operations Research und Machine
Learning“. In der Demo [Matching-Polytop](https://github.com/sebastian-hanisch/matching-polytop-demo) hat jeder Knoten höchstens einen Partner, und **eine**
Familie weiterer Ungleichungen genügt. Hier fällt genau diese Annahme weg: Bei der **symmetrischen Rundreise** hat jeder Ort genau zwei Nachbarn, und alles
hängt zusammen. Die 2-Matching-Ungleichungen sind das Gegenstück zu den ungeraden Mengen, und der Kern der Trennung ist derselbe (kleinster ungerader Schnitt unter
den Schnitten eines Gomory-Hu-Baums); der Hilfsgraph ist ein anderer: dort mit einem Zusatzknoten, hier mit geteilten Kanten. Gerechnet wird auf selbst gewählten Zufallsfamilien und einer kleinen Beispielfamilie (Prisma), nicht auf echten Daten.

## Warum dieses Problem

Das LP, das nur verlangt, dass jeder Ort genau zwei Nachbarn hat, liefert oft getrennte Kreise oder halbe Verbindungen. Man kann fehlende
Ungleichungen nachladen. Die Frage der Demo: **Wie weit kommt man mit den bekannten Familien (Subtourzeilen, 2-Matching-Ungleichungen, Kämme), und woran
erkennt man, dass noch etwas fehlt?** In einem Reiter geht es um die Gegenfrage, ob viele Facetten zwingend eine große Beschreibung bedeuten.

## Modell

* **n Orte** (Knoten des vollständigen Graphen), symmetrische ganzzahlige Kosten, gesucht ist die kürzeste Rundreise.
* **Grad-LP:** min c·x, x(δ(v)) = 2 je Ort, 0 ≤ x ≤ 1.
* **Subtourzeile:** x(δ(S)) ≥ 2 für jede Menge S mit mindestens zwei Orten auf beiden Seiten. Grad-LP mit allen: **Subtour-LP**.
* **2-Matching-Ungleichung** (Blütenform): Griff H, ungerade Menge F von Kanten in δ(H): x(δ(H) ohne F) + Σ_F (1 − x_e) ≥ 1.
* **Kamm:** Griff H und p ≥ 3 (ungerade) disjunkte Zähne T_j, jeder mit Orten in H und außerhalb: x(δ(H)) + Σ x(δ(T_j)) ≥ 3p + 1. Zähne aus je einer Kante
  geben die 2-Matching-Ungleichung.
* **Rundreise-Polytop:** konvexe Hülle aller Rundreisen, Dimension n(n − 3)/2; seine Eckpunkte heißen Ecken. **Facette:** Ungleichung, die in keiner
  Beschreibung fehlen darf.
* **Lücke** = (Optimum − LP-Wert)/Optimum. **Ganzzahlig und optimal** verlangt beides: die LP-Lösung ist eine 0/1-Lösung und kostet so viel wie die kürzeste
  Rundreise.
* **Löser:** HiGHS, dualer Simplex (`highs-ds`); welche Ecke bei mehreren Optima erscheint, hängt vom Löser ab. Verletzt heißt: Subtour unter 2 − 10⁻⁶,
  2-Matching unter 1 − 10⁻⁶.
* **Kürzeste Rundreise:** gefunden als LP-Lösung oder mit `scipy.optimize.milp` (nachgeladene Subtourzeilen; die schon geladenen Ungleichungen werden
  mitgegeben). **Ein MILP-Status allein gilt nicht als Beleg** (`rpt_referenz.py`). Die App prüft jede gemeldete Rundreise unabhängig und belegt ihre Länge L
  so: (1) eine **selbst nachgerechnete Dualschranke** des LP liegt über L − 1 (die Kosten sind ganzzahlig, die Schranke folgt aus schwacher Dualität für
  beliebige Dualwerte und hängt nicht davon ab, ob der Löser richtig gerechnet hat); sonst (2) bis 13 Orte eine Dynamik über Teilmengen (Held und Karp);
  sonst (3) ein eigenes Branch-and-Bound mit derselben Schranke und Zeitgrenze. Gelingt keiner der Belege, steht dort „nur laut HiGHS-MILP“ (nach einer
  Plausibilitätsprüfung mit 2-opt und Or-opt) oder „nicht bestimmt“ (dann ohne Lücke).

**Instanzfamilien** (Zufall nur über SplitMix64, `rpt_rng.py`):

* **Karte:** Orte mit Koordinaten (below(1001), below(1001)); Kosten = auf eine ganze Zahl gerundeter Abstand (durch das Runden nur nahezu metrisch).
* **Haufen:** 4 Mittelpunkte (150 + below(701) je Koordinate), je Ort ein Mittelpunkt below(4) und je Koordinate ein Versatz −100 + below(201).
* **Zufallsmatrix:** je Paar Kosten 1 + below(1000), nicht metrisch.
* **Prisma:** zwei Dreiecke, drei Wege aus k Kanten dazwischen; Dreieckskanten kosten c, Wegkanten d, übrige Paare den kürzesten Weg (metrisch).

Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

| Verfahren | Was | Art |
|---|---|---|
| Subtourtrennung | erst die Zusammenhangskomponenten der benutzten Verbindungen, sonst alle Schnitte des Gomory-Hu-Baums unter 2 | exakt |
| Aufzählen | alle Griffe, zu jedem die beste ungerade Zahnmenge | exakt, nur klein (Gegenprobe bis 16 Orte) |
| Padberg–Rao | jede benutzte Verbindung wird geteilt (u–w mit Kapazität x, w–v mit 1 − x); kleinster ungerader Schnitt unter 1 unter den Schnitten eines Gomory-Hu-Baums (Bauart nach Gusfield, Flüsse mit `scipy.sparse.csgraph.maximum_flow`) | exakt, polynomiell |
| Komponentenregel | Griffe nur aus den Zusammenhangskomponenten der gebrochenen Kanten, je mit der besten Zahnmenge | Heuristik |
| alle je Runde / eine je Runde | alle gefundenen verletzten Ungleichungen oder nur die am stärksten verletzte nachladen (Gleichstand nach Rundung auf 7 Stellen: kleinerer Griff, dann Knotennummern; Komponentenregel: erste Komponente) | Auswahlregel |
| Kammsuche | alle Griffe und alle disjunkten Zahnfamilien, die verletzt sein können | vollständig, bis 12 Orte |

**Gemeinsames Trennverfahren (`rpt_trennung.py`)** ist eine Kopie aus der Matching-Polytop-Demo. Dass die Baumschnitte über alle Knoten den kleinsten
ungeraden Schnitt enthalten, ist eine **eigene Herleitung** (im Modulkopf), nicht extern geprüft, in den Tests gegen das Aufzählen geprüft.

**Zahnzerlegung (eigene Herleitung, nicht extern geprüft).** Padberg–Rao liefert auch Zahnmengen mit zwei Kanten an einem Ort. Bringt man diesen Ort v auf
die andere Seite des Griffs und streicht beide Kanten, ist mit x(δ(v)) = 2 der neue Wert gleich dem alten minus 2·S (S = Summe der übrigen Kanten an v, die
den alten Griff kreuzen). Drei Zahnkanten an einem Ort sind nie verletzt, eine einzige Zahnkante unter den Subtourzeilen auch nicht. Deshalb zeigt die App
jede Zeile als Kamm mit disjunkten Zähnen. Die Identität ist in den Tests mit exakten Brüchen geprüft; ins LP kommt die gefundene Form.

**Vollständige Kammsuche.** Unter den Subtourzeilen ist x(δ(T_j)) ≥ 2; ein Kamm ist genau dann verletzt, wenn Σ (x(δ(T_j)) − 2) < p + 1 − x(δ(H)). Innerhalb
dieser Schranke wird alles durchsucht (p = 3 oder 5 bei höchstens 12 Orten).

**Reiter „Und mit Hilfsvariablen?“** Permutaeder (Anordnungen von 1 bis n, 2ⁿ − 2 Facetten) aus einem Komparatornetz (Idee nach Goemans): je Komparator
lo + hi = a + b, lo ≤ a, lo ≤ b; die Ausgänge 1, …, n werden eingesetzt. Zwei Netze: Batchers Merge-Exchange (Sortiereigenschaft für 2 bis 16 Drähte mit dem
0-1-Prinzip geprüft) und das Netz des Sortierens durch Einfügen. **Eigene Herleitung, nicht extern geprüft:** Für jedes Netz liegt die Projektion im
Permutaeder, und eine Permutation liegt genau dann darin, wenn das Netz sie sortiert (Beweis im Mathe-Expander; in den Tests über alle Netze mit einem
gestrichenen Komparator bei 4 Werten per LP gegen den Durchlauf geprüft).

**Grenzen der App:** n von 5 bis 50; Prisma mit 1 bis 4 Kanten je Weg; Kämme bis 12 Orte (darüber ist der Knopf gesperrt); ganze Hülle live bei 5 und 6 Orten,
bei 7 vorgerechnet; Hilfsgraph bis 12 Orte gezeichnet; das MILP und das eigene Branch-and-Bound bekommen je 1 Sekunde, bei fehlendem Beleg gibt es einen Knopf
für je 15 Sekunden.

Rechenzeiten hängen am Rechner und sind kein Befund dieser Demo; bei längeren Rechnungen zeigt die App einen Hinweis mit Zeitangabe.

## Befunde (gemessen, keine Behauptungen)

Alle Reihen sind mit `tools/rpt_make_tables.py` gerechnet und in `rpt_tabellen.json` eingefroren (`--check` rechnet neu und vergleicht). Werte, die nur vom
LP-Wert abhängen, sind plattformfest; die Zahl **ganzzahlig und optimal** gelöster Instanzen, Zahlen nachgeladener Zeilen, gebrochene Lösungen und die
Komponentenregel hängen an der Ecke, die HiGHS bei mehreren Optima liefert, und können auf einem anderen System abweichen. `tests/test_claims.py` prüft jede
Tabelle gegen die Datei und rechnet Teile neu.

Gegenproben der Rechnung: **Jedes Referenzoptimum** aller Reihen (Leiter, Kämme, Prisma: 3.090 Instanzen) stimmt mit einem zweiten Löser überein (CP-SAT mit
`AddCircuit` aus ortools, Status „optimal“; als Hilfe bekommt CP-SAT die für jede Rundreise gültigen Subtourzeilen, deren Form vorher geprüft wird); bis
13 Orte zusätzlich mit einer Dynamik über Teilmengen (2.309 Instanzen, dazu das Prisma mit 15 Orten). Eine Abweichung trat nicht auf. In der Leiter ist
außerdem jedes der 1.480 Optima selbst belegt: 1.311 durch die nachgerechnete Dualschranke, 25 durch die Dynamik über Teilmengen, 144 durch das eigene
Branch-and-Bound. Auf allen 1.000 Instanzen der Leiter bis 16 Orte endet die Trennung durch Aufzählen aller Griffe beim selben LP-Wert wie
Padberg–Rao.

### Handbeispiele

* **Prisma, 6 Orte, Dreiecke 2, Verbinder 1:** 60 Rundreisen, Optimum 10. Die Lösung mit ½ auf den Dreiecken und 1 auf den Verbindern erfüllt alle Grad- und
  Subtourzeilen und kostet 9. Die 2-Matching-Ungleichung mit Griff abc und den drei Verbindern ist um 1 verletzt (als Kamm 9 < 10); mit ihr ist der Wert 10.
* **Prisma, 6 Orte, Dreiecke 1, Verbinder 4:** Das Grad-LP wählt zwei Dreiecke (6); eine Subtourzeile auf einem Dreieck hebt auf 12 = Optimum.

### Wie weit kommt jede Stufe? (ganzzahlig und optimal)

| Familie, Orte (Instanzen) | Grad-LP | + Subtour | + 2-Matching, Komponentenregel (alle / eine je Runde) | + 2-Matching, exakt | mittlere Lücke Grad / Subtour / exakt |
|---|---|---|---|---|---|
| Karte 8 (100) | 44 | 99 | 100 / 100 | 100 | 4,58 / 0,004 / 0,000 % |
| Karte 12 (200) | 32 | 193 | 197 / 196 | 197 | 6,52 / 0,027 / 0,012 % |
| Zufallsmatrix 12 (200) | 100 | 166 | 178 / 178 | 179 | 1,83 / 0,180 / 0,094 % |
| Haufen 12 (200) | 4 | 195 | 197 / 197 | 197 | 22,14 / 0,005 / 0,001 % |
| Karte 16 (100) | 7 | 88 | 95 / 94 | 96 | 7,24 / 0,089 / 0,030 % |
| Zufallsmatrix 16 (100) | 46 | 81 | 88 / 85 | 90 | 1,74 / 0,169 / 0,079 % |
| Haufen 16 (100) | 0 | 92 | 96 / 96 | 96 | 22,06 / 0,052 / 0,027 % |
| Karte 30 (100) | 0 | 45 | 66 / 64 | 76 | 7,18 / 0,469 / 0,102 % |
| Zufallsmatrix 30 (100) | 26 | 57 | 68 / 67 | 69 | 1,26 / 0,266 / 0,154 % |
| Haufen 30 (100) | 0 | 76 | 90 / 89 | 93 | 23,34 / 0,071 / 0,013 % |
| Karte 50 (60) | 0 | 9 | 26 / 22 | 32 | 6,66 / 0,826 / 0,219 % |
| Zufallsmatrix 50 (60) | 10 | 21 | 26 / 24 | 31 | 0,86 / 0,255 / 0,127 % |
| Haufen 50 (60) | 0 | 28 | 41 / 40 | 45 | 20,00 / 0,176 / 0,058 % |

* **In jeder Reihe löst jede Stufe mindestens so viele Instanzen wie die vorige;** außer auf den Karten mit 8 Orten bleibt auch nach exaktem 2-Matching
  etwas offen. Auf den Karten sinkt der Anteil ganzzahlig und optimal nach exaktem 2-Matching mit jeder gemessenen Größe
  (100 % bei 8 Orten, 98,5 % bei 12, 96 % bei 16, 76 % bei 30, 53,3 % bei 50). Bis 16 Orte beruhen die Schritte auf wenigen offenen Instanzen; deutlich ist
  der Rückgang bei 30 und 50. **Bei der Zufallsmatrix nicht in jedem Schritt:** 89,5 % bei 12 Orten, 90 % bei 16.
* Bei den Haufen liegt das Grad-LP je Reihe im Mittel 20,0 bis 23,3 % unter dem Optimum, nach den Subtourzeilen höchstens 0,18 %.
* **Wert und Ecke sind zweierlei:** In drei Reihen erreicht der LP-Wert nach exaktem 2-Matching das Optimum auf einer Instanz mehr, als die Spalte zählt (Haufen
  30: 94 statt 93, Zufallsmatrix 30: 70 statt 69, Zufallsmatrix 50: 32 statt 31); der Löser liefert dort eine gebrochene Ecke, obwohl eine Rundreise denselben
  Wert hat.

### Falle: „Nichts gefunden“ heißt nicht „nichts verletzt“

Die Komponentenregel hört auf, obwohl Padberg–Rao noch eine verletzte 2-Matching-Ungleichung findet (alle je Runde / eine je Runde):

| Reihe | Instanzen mit übersehener Ungleichung | anderer Endwert je Auswahlregel | größter Abstand zum exakten LP-Wert, in % des Optimums (eine je Runde) |
|---|---|---|---|
| Karte 12 (200) | 1 / 2 | 2 | 0,42 % |
| Karte 30 (100) | 15 / 25 | 31 | 0,59 % |
| Karte 50 (60) | 29 / 31 | 29 | 0,81 % |
| Zufallsmatrix 50 (60) | 18 / 23 | 9 | 0,32 % |
| Haufen 50 (60) | 12 / 13 | 16 | 0,17 % |

Über alle 13 Reihen übersieht die Regel mit „alle je Runde“ auf 92 von 1.480 Instanzen etwas, mit „eine je Runde“ auf 123. In allen diesen Fällen mit „eine je
Runde“ lag auch der LP-Wert unter dem exakten; mit „alle je Runde“ nicht immer. **Der Gegenbefund:** Der Wertunterschied ist klein, in beiden Auswahlen
höchstens 0,87 % des Optimums (Zufallsmatrix, 30 Orte), und bis 12 Orte ändert die Regel selten etwas (Karte 8: nie). Bei der exakten Trennung hängt der
Endwert nicht an der Auswahlregel.

### Nachgeladene Zeilen: nur mit der Auswahlregel eine Aussage

Karte, 50 Orte (60 Instanzen), Median / größte Zahl je Instanz:

| Stufe | alle je Runde | eine je Runde |
|---|---|---|
| Subtourzeilen bis zur Subtourschranke (gleicher Endwert) | 24 / 61 | 10 / 22 |
| 2-Matching-Zeilen, exakt (Mittel 20,77 bzw. 10,15) | 7,5 / 175 | 5 / 85 |

Padberg–Rao liefert oft Zahnmengen mit einem doppelt belegten Ort: bei Karte 50 in 654 von 1.855 Zeilen (beide Auswahlregeln zusammen), über alle Reihen in
1.039 von 3.627.

### Wie sieht die gebrochene Lösung nach den Subtourzeilen aus?

| Reihe | gebrochen | davon nur 0, ½, 1 | davon: ½-Kanten nur getrennte Kreise |
|---|---|---|---|
| Karte 30 / 50 | 55 von 100 / 51 von 60 | 54 / 43 | 42 / 34 |
| Haufen 30 / 50 | 24 von 100 / 32 von 60 | 22 / 28 | 20 / 22 |
| Zufallsmatrix 30 / 50 | 43 von 100 / 39 von 60 | 32 / 20 | 28 / 19 |

**Nicht durchgehend halbganzzahlig:** bei der Zufallsmatrix mit 50 Orten ist nur gut die Hälfte der gebrochenen Lösungen halbganzzahlig.

### Reichen alle Kämme? (bis 12 Orte, je 200 Instanzen)

Offen = nach allen 2-Matching-Ungleichungen nicht ganzzahlig und optimal. Danach werden Kämme nachgeladen, bis kein Kamm mehr verletzt ist.

| Reihe | offen | mit allen Kämmen gelöst | mittlere Lücke der offenen vorher → nachher | größte danach |
|---|---|---|---|---|
| Zufallsmatrix 8 | 5 | 5 | 2,25 → 0,00 % | 0,00 % |
| Zufallsmatrix 10 | 10 | 8 | 1,50 → 0,30 % | 2,35 % |
| Zufallsmatrix 12 | 21 | 14 | 0,90 → 0,37 % | 3,43 % |
| Karte 12 | 3 | 3 | 0,78 → 0,00 % | 0,00 % |
| Haufen 12 | 3 | 3 | 0,09 → 0,00 % | 0,00 % |

Karte 8 und 10 sowie Haufen 10: nichts offen. Auf allen 9 Instanzen, die offen bleiben (2 bei 10, 7 bei 12 Orten), liegt der LP-Wert nach allen Kämmen unter
dem Optimum; eine zweite Kammsuche ohne Beschneidung (Dynamik über Teilmengen) findet dort keinen verletzten Kamm. **Auch alle Kämme genügen also nicht
immer.** Die Suche selbst ist an gebrochenen Lösungen mit 7 Orten gegen das Aufzählen aller Zahnfamilien geprüft.

### Prisma mit langen Wegen

| Kanten je Weg k (Orte) | Subtour-LP | + 2-Matching | + alle Kämme | Optimum | Optimum / Subtour-LP |
|---|---|---|---|---|---|
| 1 (6) | 6 | 6 | 6 | 6 | 1,00 |
| 2 (9) | 9 | 10 | 10 | 10 | 1,11 |
| 3 (12) | 12 | 13 | 13,33 | 14 | 1,17 |
| 4 (15) | 15 | 16,5 | nicht gerechnet | 18 | 1,20 |

(Alle Kantenlängen 1.) Für k = 1 bis 4 nachgerechnet: Optimum 4k + 2 bei Subtour-LP 3k + 3. Setzt sich das fort, geht der Quotient gegen 4/3; **für alle k ist
das hier nicht bewiesen.** Dass der Quotient aus Optimum und Subtour-LP bei metrischen Kosten nie über 4/3 liegt, gilt als offene Vermutung; bewiesen ist eine obere Schranke
etwas unter 3/2 (Karlin, Klein und Oveis Gharan, nur Abstract gelesen). Mit 6 Orten und Dreieckskosten c, Verbinder 1: c = 2 ergibt 9 gegen 10, c = 3 12 gegen
13, c = 5 18 gegen 19 (2-Matching schließt jeweils); Dreieck 1, Verbinder 4: Grad-LP 6, Subtour-LP 12 = Optimum.

### Warum nicht alle Ungleichungen?

| Orte | Dimension | Rundreisen | Facetten | davon Subtour (2^(n−1) − n − 1) | Herkunft |
|---|---|---|---|---|---|
| 5 | 5 | 12 | 20 | 10 | hier gerechnet (live) |
| 6 | 9 | 60 | 100 | 25 | hier gerechnet (live) |
| 7 | 14 | 360 | 3.437 | 56 | hier gerechnet (vorgerechnet, ganze Zahlen, etwa zwei Minuten) |
| 8 | 20 | 2.520 | 194.187 | 119 | veröffentlicht (SMAPO), nicht nachgerechnet |
| 9 | 27 | 20.160 | 42.104.442 | 246 | veröffentlicht (SMAPO), nicht nachgerechnet |

Facetten gelten modulo der Gradgleichungen. Bei 5 Orten bestehen sie aus den Schranken (10 + 10), bei 6 aus 15 + 15 Schranken, 10 Subtourzeilen mit 3 Orten
und 60 2-Matching-Ungleichungen mit 3 Zähnen. Bei 7 Orten: 21 + 21 Schranken, 35 Subtourzeilen, 840 2-Matching-Ungleichungen mit 3 Zähnen, 1.260 Kämme mit
Zähnen aus 2, 2 und 3 Orten und **1.260 weitere**, die keine Kämme mit 3 Zähnen sind (Familie hier nicht bestimmt). Bei 50 Orten gibt es 2⁴⁹ − 51
Subtourfacetten; auf den 60 Karten wurden im Median 24 nachgeladen (alle je Runde).

### Mit Hilfsvariablen (Reiter)

| Werte n | Facetten des Permutaeders | Batcher: Ungleichungen / Variablen / Gleichungen | Einfüge-Netz: Ungleichungen |
|---|---|---|---|
| 3 | 6 | 6 / 6 / 3 | 6 |
| 4 | 14 | 10 / 10 / 5 | 12 |
| 5 | 30 | 18 / 18 / 9 | 20 |
| 16 | 65.534 | 126 / 126 / 63 | 240 |

* Bei 3 Werten spart die Erweiterung nichts (6 gegen 6). Bei 16 Werten sind es 126 Ungleichungen statt 65.534 Facetten, und der LP-Wert ist der durch Sortieren
  bestimmte (auf zufälligen Zielen geprüft bis 10 Werte).
* **Ein gestrichener Komparator** (Batcher, n = 4): in der Projektion liegen noch 14, 14, 12, 12 bzw. 16 von 24 Permutationen (Komparator 1 bis 5), genau die,
  die das Netz noch sortiert. Welche und wie viele verloren gehen, hängt am Netz (beim Einfüge-Netz mit 4 Werten sind es je nach Komparator zwischen 6 und 20).
* **Nur zitiert** (nur die Abstracts gelesen): Rothvoß: jede erweiterte Formulierung des perfekten Matching-Polytops des vollständigen Graphen K_n braucht
  2^Ω(n) Ungleichungen, obwohl gewichtetes perfektes Matching polynomiell lösbar ist; Fiorini, Massar, Pokutta, Tiwary und de Wolf: kein LP polynomieller Größe
  projiziert auf das Rundreise-Polytop, auch ohne Symmetrie. An kleinen Instanzen ist das nicht messbar; kein Messwert der Demo belegt es.
* **Spannbäume** (Martins Formulierung, hier nach einer Übersicht): Für K_n ab 4 Knoten hat das Spannbaumpolytop m + 2ⁿ − n − 2 Facetten (m Verbindungen), die Formulierung
  n(n − 1)² Ungleichungen. Weniger Ungleichungen als Facetten hat sie erst ab n = 10 (810 statt 1.057, bei 855 statt 45 Variablen und 541 Gleichungen); bei n = 4
  sind es 36 statt 16. Dasselbe Polytop ohne Hilfsvariablen, mit Rangungleichungen und Trennen, zeigt die [Matroid-Polytop-Demo](https://github.com/sebastian-hanisch/matroid-polytop-demo).

## Ehrliche Grenzen

* Alle Häufigkeiten gelten für die drei selbst gewählten Zufallsfamilien mit den genannten Zufalls-Seeds. Das ist keine Aussage über Rundreisen im Allgemeinen
  und kein Anwendungsnachweis. „Offen“ heißt offen nach diesen Familien, nicht schwer zu lösen.
* Die Leiter ist nur bis 50 Orte gemessen, die Kämme nur bis 12 Orte.
* Ganzzahlig-und-optimal-Zahlen, Zeilenzahlen, Runden, gebrochene Lösungen und die Komponentenregel hängen an Auswahlregel, Löser und System.
* Die App rechnet mit Gleitkommazahlen (Toleranz 10⁻⁶), auch die Dualschranke als Beleg; die Maximalflüsse mit auf 10⁻⁷ gerundeten Kapazitäten. Nur die ganze
  Hülle bei 5 bis 7 Orten wird in ganzen Zahlen gerechnet.
* Die kürzeste Rundreise ist in der App nicht immer zweit belegt: Bei großen Instanzen kann dort „nur laut HiGHS-MILP“ stehen; Lücke und Verdikte hängen dann
  an diesem Wert. CP-SAT läuft nicht in der App, sondern nur beim Vorrechnen und in den Tests. Knoten, die der LP-Löser im eigenen Branch-and-Bound als
  unzulässig meldet, werden ihm geglaubt.
* Facettenzahlen ab 8 Orten sind veröffentlichte Werte, nicht nachgerechnet. Die Sätze sind nach Skripten, Aufsätzen, Datenseiten und teils nur nach Abstracts
  oder Sekundärzitaten angegeben; viele Originale sind nicht gelesen.
* Die Sätze zu den Hilfsvariablen (Rothvoß, Fiorini u. a.) werden nur zitiert.
* Die hier gemessenen LP-Lücken sagen nichts darüber, wie gut eine Tourenheuristik auf diesen Instanzen abschneidet.

## Befunde und Korrekturen gegenüber früheren Rechnungen

* **2-Matching bei 30 und 50 Orten:** Eine frühere Rechnung trennte dort nur mit der Komponentenregel und unterschätzte damit die Stufe (Tabelle oben: Karte
  50: 26 mit der Regel gegen 32 exakt, Karte 30: 66 gegen 76).
* **Nachladezahlen** ohne Auswahlregel sind keine Aussage (Karte 50, Subtour: Median 24 oder 10).
* **„Mit der Größe bleibt mehr offen“** gilt in den gemessenen Kartenreihen in jedem Schritt (deutlich erst bei 30 und 50 Orten), bei der Zufallsmatrix
  zwischen 12 und 16 Orten nicht.
* **Gelöst ist nicht dasselbe wie Wert erreicht:** Zwei Rechnungen mit demselben Verfahren können sich bei den Spalten „ganzzahlig und optimal“ um einzelne
  Instanzen unterscheiden, weil der Löser bei mehreren optimalen Ecken eine andere wählt; der LP-Wert selbst ist derselbe.
* **Komponentenregel:** Auf dem Prisma mit Wegen aus 3 Kanten hört sie mit „alle je Runde“ mit verletzten Ungleichungen auf, obwohl ihr LP-Wert mit dem
  exakten übereinstimmt (13); mit „eine je Runde“ bleibt sie bei 12,6. „Verletzt“ heißt also nicht immer „Wert zu klein“.
* **Netzformulierung:** Ein Netz, das nicht sortiert, gibt nie ein zu großes Polytop, nur ein zu kleines. Welche Permutationen verloren gehen und wie viele,
  hängt am Netz.

## Verifikation und Tests

`python -m pytest tests/ -q` (242 Tests, mit `pytest-xdist` und `-n 4` rund eine Minute):

* **Mini-Instanzen je Einheit** (`test_algorithm.py`, Prisma mit Handwerten), Randfälle und Eigenschaften (`test_eigenschaften_rundreise_polytop.py`: keine
  nachgeladene Zeile schneidet eine Rundreise ab, Werte steigen monoton, exakter Endwert unabhängig von der Auswahlregel, Kosten mal Faktor, Umnummerieren,
  Zahnzerlegung mit exakten Brüchen, 4 und 5 Orte: Subtour-LP ist die Hülle).
* **Orakel** (`test_oracle_rundreise_polytop.py`): Subtourstufe gegen das LP mit allen Subtourzeilen; exakte 2-Matching-Stufe bei 6 Orten gegen das LP mit
  allen 2-Matching-Ungleichungen; Padberg–Rao gegen das Aufzählen aller Griffe; Optimum gegen alle Rundreisen; Kammsuche gegen das Aufzählen aller Zahnfamilien
  (7 Orte); ganze Hülle bei 5 und 6 Orten gegen alle Rundreisen; Mindestschnitt gegen networkx; Netz-LP gegen Sortieren und gegen den Durchlauf; Martins
  Formulierung gegen networkx (Kruskal). Zur Referenz: die Dualschranke bleibt mit verfälschten Dualwerten unter der kürzesten Rundreise; das eigene
  Branch-and-Bound gegen die Dynamik über Teilmengen (Instanzen, bei denen es verzweigen muss); ein absichtlich falsches MILP-Ergebnis wird als „nicht
  bestimmt“ gemeldet; Vergleich mit CP-SAT.
* **Verdikte** an den Schwellen, **Presets** (jedes zeigt, was sein Hilfetext sagt), **Claims** (`test_claims.py`), **Kopien** (SplitMix64-Vorlage,
  `modul_frisch.py`, Trennverfahren, doppelte Beschreibung; übersprungen, wenn die Quelle fehlt), App-Zustände, Layout, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App mit den Experimenten 1 bis 5 und dem Reiter zu Hilfsvariablen |
| `rpt_instanzen.py`, `rpt_rng.py` | Instanzfamilien, SplitMix64 |
| `rpt_lp.py` | LP mit Gradgleichungen und nachgeladenen Zeilen (HiGHS), Struktur gebrochener Lösungen |
| `rpt_subtour.py`, `rpt_zweimatching.py`, `rpt_kamm.py` | Trennungen: Subtour, 2-Matching (Padberg–Rao, Komponentenregel, Aufzählen, Zahnzerlegung), Kämme |
| `rpt_trennung.py` | Gomory-Hu-Baum und ungerade Schnitte (Kopie aus der Matching-Polytop-Demo) |
| `rpt_leiter.py` | die Stufen als Komposition |
| `rpt_referenz.py` | kürzeste Rundreise (MILP mit Zeitgrenze) und ihr Beleg: Dualschranke, Dynamik über Teilmengen, eigenes Branch-and-Bound, 2-opt |
| `rpt_dd.py`, `rpt_huelle.py` | doppelte Beschreibung in ganzen Zahlen (Kopie aus Ganzzahlige Ecken), Facetten aller Rundreisen und ihre Familien |
| `rpt_netz.py` | Komparatornetze, erweiterte Formulierung des Permutaeders, Zählungen zum Spannbaum |
| `rpt_evaluation.py`, `rpt_messreihen.py` | Auswertung, Verdikte, Messreihen |
| `rpt_visualization.py`, `rpt_pdf_export.py` | Zeichnungen, PDF-Bericht |
| `rpt_presets.py`, `rpt_constants.py`, `rpt_layout.py` | Presets und Permalink, Grenzen und Texte, fester Kopf |
| `modul_frisch.py` | lädt geänderte Hilfsmodule nach einem Update neu |
| `rpt_tabellen.json` | vorgerechnete Messreihen und die Hülle bei 7 Orten |
| `tools/rpt_make_tables.py` | Erzeuger der Tabellen (`--check`; `--teil` leiter, kaemme, prisma oder huelle) |
| `tests/` | siehe oben |

## Bewusst nicht umgesetzt

* Keine weiteren Facettenfamilien (etwa Cliquenbäume), kein Branch-and-Cut mit 2-Matching- oder Kammtrennung in den Knoten (das eigene
  Branch-and-Bound mit Subtourzeilen dient nur als Beleg der kürzesten Rundreise).
* Keine Kammsuche über 12 Orte und keine Heuristik dafür.
* Keine Rechnung der ganzen Hülle ab 8 Orten.
* Zu den Hilfsvariablen keine eigene Seite: kein Nachladen für das Permutaeder oder Spannbäume, keine unteren Schranken (nichtnegativer Rang).
* Keine gerichtete Rundreise (siehe [Zuordnung gegen Rundreise](https://github.com/sebastian-hanisch/zuordnung-vs-rundreise-demo)), keine Tourenheuristiken
  (siehe [Lin-Kernighan](https://github.com/sebastian-hanisch/lin-kernighan-demo)).
* Der Gomory-Hu-Baum selbst ist Thema der [Gomory-Hu-Demo](https://github.com/sebastian-hanisch/gomory-hu-demo); Nachladen mit denselben Schnittbedingungen (anderes
  Trennverfahren) und Gradschranke zeigt [Zusammenhang verstärken](https://github.com/sebastian-hanisch/zusammenhang-verstaerken-demo) (andere Netze, andere Regel, kein
  Zahlenvergleich).

**Literatur** (Stufen: gelesen, nur Abstract oder nur Sekundärzitat): Subtourzeilen nach Dantzig, Fulkerson und Johnson und die Kammformen nach Chvátal sowie
Grötschel und Padberg, hier nach A. Letchford, Math. Oper. Res. 25 (2000), dort auch SEP = Rundreise-Polytop bis 5 Orte; Padberg und Rao (1982), hier nach
Letchford, Reinelt und Theis, SIAM J. Discrete Math. 22 (2008); Facettenzahlen nach SMAPO (Universität Heidelberg, Datenseite); Allgemeine Kamm-Trennung:
Vorabdruck arXiv:2610.09065, nicht geprüft; Grötschel, Lovász und Schrijver nach einem Sekundärzitat; Permutaeder nach Rado (Übersicht); Goemans, Rothvoß,
Fiorini u. a. (nur Abstracts); Martin (1991) nach einer Übersicht von Conforti, Cornuéjols und Zambelli.

## Lokal ausführen

Entwickelt mit Python 3.14; die CI prüft mit Python 3.12.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
python tools/rpt_make_tables.py --check                 # mit CP-SAT-Gegenprobe, etwa eine halbe Stunde
python tools/rpt_make_tables.py --check --ohne-cpsat    # nur nachrechnen
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Fehlende Ungleichungen: von Ecken bis Runden](https://sebastianhanisch.net/konzepte-polyeder.html).
