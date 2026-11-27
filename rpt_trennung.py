"""Gemeinsames Trennverfahren: Schnitte eines Gomory-Hu-Baums, darunter der kleinste ungerade Schnitt (nach Padberg und Rao).

Kopie von mpt_trennung.py aus matching-polytop-demo (https://github.com/sebastian-hanisch/matching-polytop-demo); ab der ersten Importzeile
unverändert (geprüft in tests/test_kopien.py). In dieser Demo benutzen es die Subtourtrennung (Schwelle 2, ohne Markierung, `baumschnitte`) und die
2-Matching-Trennung (Schwelle 1, `ungerade_schnitte` auf dem geteilten Hilfsgraphen).

Eingabe ist ein ungerichteter Graph mit Knoten 0 .. n-1, Kanten (u, v, Kapazität) mit Kapazität >= 0 und, für die ungeraden Schnitte, eine
Markierung 0/1 je Knoten.

Rechenweg: (1) Kanten mit Kapazität >= grenze liegen in keinem Schnitt unter der Schwelle; ihre Endknoten werden zusammengezogen. (2) Über alle
verbleibenden Knoten wird ein Gomory-Hu-Baum gebaut (Bauart nach Gusfield, Fassung, die einen Schnittbaum liefert); die Maximalflüsse rechnet
`scipy.sparse.csgraph.maximum_flow` mit ganzzahligen Kapazitäten (Maßstab SKALA, für den Fluss bei `deckel` gekappt). (3) Jede Baumkante liefert
einen Schnitt; sein Wert wird aus den Gleitkomma-Kapazitäten neu berechnet.

Warum die Baumschnitte für den kleinsten ungeraden Schnitt genügen, auch wenn der Baum über alle Knoten und nicht nur über die markierten gebaut
ist (eigene Herleitung, nicht extern geprüft, in den Tests gegen das Aufzählen aller Schnitte geprüft): Sei S ein ungerader Schnitt mit kleinstem
Wert. Die Seiten der Baumkanten, die S trennt, haben zusammen dieselbe Parität der markierten Knoten wie S; also trennt eine dieser Baumkanten
ungerade viele markierte Knoten ab. Ihre Endknoten liegen auf verschiedenen Seiten von S, und ihr Schnitt ist ein kleinster Schnitt zwischen
diesen beiden Knoten, also höchstens so groß wie S. Das Zusammenziehen ändert keinen Schnitt unter der Schwelle. Rundung beim Skalieren: Je Kante
weicht die Kapazität um höchstens 0,5 / SKALA ab.

Keine eigenen Abhängigkeiten, kein Streamlit; nur numpy und scipy.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

SKALA = 10_000_000     # Maßstab der ganzzahligen Kapazitäten für den Maximalfluss
TOL = 1e-6             # verletzt heißt: Schnittwert < grenze - TOL
NULL = 1e-9            # Kanten bis zu dieser Kapazität zählen nicht; ab grenze - NULL wird zusammengezogen
DECKEL = 2.0           # für den Fluss werden Kapazitäten hier gekappt; damit sind Schwellen bis 2 exakt


def zusammenziehen(n: int, kanten, grenze: float = 1.0):
    """Zieht die Endknoten aller Kanten mit Kapazität >= grenze zusammen; parallele Kanten werden addiert.

    Liefert (anzahl, kapazität, gruppen): `kapazität` ist {(a, b): c} mit a < b auf den neuen Knoten 0 .. anzahl-1, `gruppen[a]` die alten Knoten
    des neuen Knotens a. Die neuen Knoten sind in der Reihenfolge ihres kleinsten alten Knotens nummeriert."""
    kopf = list(range(n))

    def wurzel(a):
        while kopf[a] != a:
            kopf[a] = kopf[kopf[a]]
            a = kopf[a]
        return a

    for u, v, c in kanten:
        if c >= grenze - NULL:
            kopf[wurzel(u)] = wurzel(v)
    nummer, gruppen, neu = {}, [], []
    for a in range(n):
        r = wurzel(a)
        if r not in nummer:
            nummer[r] = len(gruppen)
            gruppen.append([])
        neu.append(nummer[r])
        gruppen[nummer[r]].append(a)
    kapazitaet = {}
    for u, v, c in kanten:
        a, b = neu[u], neu[v]
        if a != b and c > NULL:
            schluessel = (a, b) if a < b else (b, a)
            kapazitaet[schluessel] = kapazitaet.get(schluessel, 0.0) + float(c)
    return len(gruppen), kapazitaet, gruppen


def flussmatrix(anzahl: int, kapazitaet: dict, deckel: float = DECKEL):
    """Symmetrische ganzzahlige Kapazitätsmatrix für den Maximalfluss (gekappt bei `deckel`, Maßstab SKALA)."""
    zeilen, spalten, werte = [], [], []
    for (a, b), c in kapazitaet.items():
        k = int(round(min(c, deckel) * SKALA))
        zeilen += [a, b]
        spalten += [b, a]
        werte += [k, k]
    return csr_matrix((np.array(werte, dtype=np.int32), (np.array(zeilen, dtype=np.int32), np.array(spalten, dtype=np.int32))),
                      shape=(anzahl, anzahl))


def min_schnitt(matrix, s: int, t: int):
    """Kleinster Schnitt zwischen s und t. Liefert (Wert in Einheiten von 1/SKALA, boolesches Feld der Seite von s: im Restgraphen erreichbar)."""
    fluss = maximum_flow(matrix, s, t)
    rest = (matrix - fluss.flow).tocsr()
    rest.data = (rest.data > 0).astype(np.int32)
    rest.eliminate_zeros()
    seite = np.zeros(matrix.shape[0], dtype=bool)
    seite[breadth_first_order(rest, s, directed=True, return_predecessors=False)] = True
    return int(fluss.flow_value), seite


def gomory_hu(anzahl: int, kapazitaet: dict, deckel: float = DECKEL):
    """Gomory-Hu-Baum (Schnittbaum) nach Gusfield. Liefert (eltern, gewichte, flüsse): Wurzel 0 (eltern[0] = -1), gewichte[i] ist der Wert des
    kleinsten Schnitts zwischen i und eltern[i] (aus den gekappten, gerundeten Kapazitäten), flüsse die Zahl der Maximalflüsse (anzahl - 1)."""
    if anzahl <= 0:
        return [], [], 0
    matrix = flussmatrix(anzahl, kapazitaet, deckel)
    eltern = [0] * anzahl
    eltern[0] = -1
    wert = [0] * anzahl
    fluesse = 0
    for s in range(1, anzahl):
        t = eltern[s]
        f, seite = min_schnitt(matrix, s, t)
        fluesse += 1
        wert[s] = f
        for i in range(anzahl):
            if i != s and seite[i] and eltern[i] == t:
                eltern[i] = s
        if eltern[t] >= 0 and seite[eltern[t]]:
            eltern[s] = eltern[t]
            eltern[t] = s
            wert[s] = wert[t]
            wert[t] = f
    return eltern, [w / SKALA for w in wert], fluesse


def teilbaeume(eltern) -> dict:
    """Für jeden Knoten außer der Wurzel die Knoten seines Teilbaums: das ist die eine Seite des Schnitts der Baumkante (i, eltern[i])."""
    kinder = [[] for _ in eltern]
    wurzeln = []
    for i, p in enumerate(eltern):
        (kinder[p] if p >= 0 else wurzeln).append(i)
    reihenfolge = list(wurzeln)
    for a in reihenfolge:                      # Breitensuche; die Liste wächst während des Durchlaufs
        reihenfolge.extend(kinder[a])
    menge = [{i} for i in range(len(eltern))]
    for a in reversed(reihenfolge):
        if eltern[a] >= 0:
            menge[eltern[a]] |= menge[a]
    return {i: frozenset(menge[i]) for i in range(len(eltern)) if eltern[i] >= 0}


def schnittwert(kapazitaet: dict, seite) -> float:
    """Wert des Schnitts einer Knotenmenge, aus den ungerundeten Kapazitäten."""
    return float(sum(c for (a, b), c in kapazitaet.items() if (a in seite) != (b in seite)))


def baumschnitte(n: int, kanten, grenze: float = 1.0):
    """Alle Schnitte des Gomory-Hu-Baums des zusammengezogenen Graphen. Liefert eine Liste (wert, seite), `seite` als Menge alter Knoten,
    in der Reihenfolge der Baumkanten (ungefiltert, also auch Werte über der Schwelle). `grenze` bestimmt nur das Zusammenziehen und muss <= DECKEL sein."""
    if grenze > DECKEL + NULL:
        raise ValueError(f"grenze = {grenze} liegt über dem Deckel {DECKEL}")
    anzahl, kapazitaet, gruppen = zusammenziehen(n, kanten, grenze)
    if anzahl < 2:
        return []
    eltern, _, _ = gomory_hu(anzahl, kapazitaet)
    ergebnis = []
    for _, seite in sorted(teilbaeume(eltern).items()):
        alte = frozenset(x for a in seite for x in gruppen[a])
        ergebnis.append((schnittwert(kapazitaet, seite), alte))
    return ergebnis


def reihenfolge_schluessel(schnitt):
    """Feste Gleichstandsregel der Schnitte: nach dem Wert (auf 7 Stellen gerundet), dann nach der Größe der Seite, dann nach den Knotennummern."""
    wert, seite = schnitt
    return round(wert, 7), len(seite), sorted(seite)


def filtere_ungerade(schnitte, markiert, grenze: float = 1.0, tol: float = TOL):
    """Aus einer Liste (wert, seite) die Schnitte mit ungerader Zahl markierter Knoten und Wert < grenze - tol, sortiert nach `reihenfolge_schluessel`."""
    treffer = [(w, s) for w, s in schnitte if sum(markiert[a] for a in s) % 2 == 1 and w < grenze - tol]
    return sorted(treffer, key=reihenfolge_schluessel)


def ungerade_schnitte(n: int, kanten, markiert, grenze: float = 1.0, tol: float = TOL):
    """Alle verletzten ungeraden Baumschnitte: Liste (wert, seite), aufsteigend nach Wert. Ist die Liste leer, gibt es keinen ungeraden Schnitt mit
    Wert < grenze - tol (bis auf die Rundung der Kapazitäten). Die Summe aller Markierungen muss gerade sein, sonst ist „ungerade“ nicht seitenunabhängig."""
    if sum(markiert) % 2:
        raise ValueError("die Zahl der markierten Knoten muss gerade sein")
    return filtere_ungerade(baumschnitte(n, kanten, grenze), markiert, grenze, tol)
