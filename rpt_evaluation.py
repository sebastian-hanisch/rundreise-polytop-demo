"""Auswertung für die App: Kennzahlen, Prüfungen der Ergebnisse, Verdikte als reine Funktionen (Art, Text) und Zugriff auf die vorgerechneten
Reihen."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import rpt_constants as C
import rpt_instanzen as I
import rpt_kamm as K
import rpt_leiter as L
import rpt_lp as LP
import rpt_netz as NZ
import rpt_referenz as R
import rpt_zweimatching as ZM

TABELLEN_DATEI = Path(__file__).resolve().parent / "rpt_tabellen.json"
OPT_TOL = 1e-4


@dataclass(frozen=True)
class Settings:
    familie: str = "prisma"
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    k: int = 1
    dreieck: int = 1
    weg: int = 4

    def bereinigt(self) -> "Settings":
        """Werte, die bei dieser Familie keine Wirkung haben, auf feste Werte setzen (ein Cache-Schlüssel je Instanz)."""
        if self.familie == "prisma":
            return Settings("prisma", 0, 0, self.k, self.dreieck, self.weg)
        return Settings(self.familie, self.n, self.seed, 1, 1, 1)


def instanz_von(s: Settings) -> I.Instanz:
    return I.erzeuge(s.familie, s.n, s.seed, s.k, s.dreieck, s.weg)


def instanz_text(s: Settings) -> str:
    if s.familie == "prisma":
        return f"Prisma, Wege aus {s.k} {'Kante' if s.k == 1 else 'Kanten'} ({3 * s.k + 3} Orte), Dreieckskosten {s.dreieck}, Wegkosten {s.weg}"
    return f"{C.FAMILIEN_NAMEN[s.familie]}, {s.n} Orte, Zufalls-Seed {s.seed}"


# ---------------------------------------------------------------- Formatierung

def zahl(wert, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma; ganze Zahlen ohne Nachkommastellen, ab 10.000 mit Tausenderpunkt."""
    if wert is None:
        return "–"
    if abs(wert - round(wert)) < 1e-6:
        text = f"{int(round(wert)):,}".replace(",", ".")
        return text if abs(wert) >= 10000 else text.replace(".", "")
    return f"{wert:.{stellen}f}".replace(".", ",")


def ganzzahl(wert: int) -> str:
    """Große ganze Zahlen mit Tausenderpunkt."""
    return f"{wert:,}".replace(",", ".")


def prozent(wert, stellen: int = 2) -> str:
    return "–" if wert is None else f"{wert:.{stellen}f}".replace(".", ",") + " %"


def bruch(wert: float) -> str:
    """Wert als Bruch mit kleinem Nenner, sonst mit drei Nachkommastellen."""
    f = LP.als_bruch(wert, 60)
    if f is None:
        return f"{wert:.3f}".replace(".", ",")
    if f.denominator == 1:
        return str(f.numerator)
    zeichen = {(1, 2): "½", (1, 3): "⅓", (2, 3): "⅔", (1, 4): "¼", (3, 4): "¾"}
    return zeichen.get((f.numerator, f.denominator), f"{f.numerator}/{f.denominator}")


def anzahl_wort(zahl_: int, einzahl: str, mehrzahl: str) -> str:
    """„1 Zeile“, „3 Zeilen“."""
    return f"{zahl_} {einzahl if zahl_ == 1 else mehrzahl}"


def luecke(wert: Optional[float], opt: Optional[int]) -> Optional[float]:
    """(Optimum − LP-Wert) / Optimum in Prozent; ohne bestimmtes Optimum keine Lücke."""
    if wert is None or opt is None or opt <= 0:
        return None
    return max(0.0, 100.0 * (opt - wert) / opt)


def ganz_und_optimal(wert: float, x, opt: Optional[int]) -> bool:
    return opt is not None and LP.ist_ganzzahlig(x) and abs(wert - opt) < OPT_TOL


# ---------------------------------------------------------------- Analyse einer Instanz

@dataclass
class Stufenwert:
    wert: float
    x: tuple
    zeilen: int           # in dieser Stufe nachgeladen
    runden: int
    ganz: bool
    status: str


@dataclass
class Analyse:
    inst: I.Instanz
    leiter: L.Leiter
    stufen: dict          # 'grad' / 'subtour' / 'zm' -> Stufenwert
    optimum: R.Optimum
    fehler: list


def referenz_text(o: R.Optimum) -> str:
    """Woran die App die kürzeste Rundreise glaubt (reine Funktion aus dem Status)."""
    if o.status == R.DUALSCHRANKE:
        return (f"Kürzeste Rundreise {zahl(o.wert)}: gefunden (LP-Lösung oder HiGHS-MILP), belegt durch eine selbst nachgerechnete "
                "Dualschranke des LP, die jede billigere Rundreise ausschließt.")
    if o.status == R.HELD_KARP:
        return (f"Kürzeste Rundreise {zahl(o.wert)}: gefunden mit HiGHS-MILP, zweit geprüft durch eine Dynamik über Teilmengen (Held und "
                "Karp).")
    if o.status == R.ENUMERATION:
        return (f"Kürzeste Rundreise {zahl(o.wert)}: gefunden mit HiGHS-MILP, belegt durch ein eigenes Branch-and-Bound mit nachgerechneten "
                f"Dualschranken ({anzahl_wort(o.knoten, 'Knoten', 'Knoten')}).")
    if o.status == R.NUR_MILP:
        return (f"Kürzeste Rundreise {zahl(o.wert)}: **nur laut HiGHS-MILP.** Ein zweiter Beleg gelang in der Zeitgrenze nicht; 2-opt und "
                "Or-opt fanden nichts Kürzeres. Lücken und Verdikte hängen hier an diesem Wert.")
    return f"Kürzeste Rundreise nicht bestimmt ({o.grund})."


def pruefe_loesung(inst: I.Instanz, x, schranke: Optional[int]) -> list:
    """Unabhängige Prüfung einer LP-Lösung: Gradgleichungen, Schranken, Wert höchstens Optimum."""
    fehler = []
    if any(abs(g - 2.0) > 1e-6 for g in LP.gradsummen(inst.n, inst.kanten, x)):
        fehler.append("eine Gradgleichung ist verletzt")
    if any(v < -1e-9 or v > 1 + 1e-9 for v in x):
        fehler.append("ein Wert liegt außerhalb von 0 bis 1")
    if schranke is not None and sum(c * v for c, v in zip(inst.kosten, x)) > schranke + OPT_TOL:
        fehler.append("der LP-Wert liegt über dem Optimum")
    return fehler


def pruefe_zeilen_an_tour(modell: LP.Modell, tour) -> list:
    """Jede nachgeladene Zeile muss die optimale Rundreise zulassen."""
    im_x = set(tour)
    for spalten, werte, grenze in zip(modell.spalten, modell.werte, modell.grenzen):
        if sum(w for j, w in zip(spalten, werte) if j in im_x) < grenze - 1e-9:
            return ["eine nachgeladene Ungleichung schneidet die optimale Rundreise ab"]
    return []


def analyse(s: Settings, zeitgrenze_opt: float = C.ZEITGRENZE_OPT, subtour_modus: str = L.ALLE,
            zeitgrenze_beleg: float = C.ZEITGRENZE_BELEG) -> Analyse:
    inst = instanz_von(s)
    lt = L.leiter(inst, subtour_modus)
    opt = R.optimum(inst.n, inst.kanten, inst.kosten, lt.modell_zm, zeitgrenze_opt, zeitgrenze_beleg)
    sub_wert = lt.modell_sub.wert
    stufen = {
        "grad": Stufenwert(lt.start.wert, lt.start.x, 0, 0, LP.ist_ganzzahlig(lt.start.x), L.FERTIG),
        "subtour": Stufenwert(sub_wert, tuple(lt.modell_sub.x), lt.subtour.zeilen, lt.subtour.runden, LP.ist_ganzzahlig(lt.modell_sub.x),
                              lt.subtour.status),
        "zm": Stufenwert(lt.modell_zm.wert, tuple(lt.modell_zm.x), lt.zweimatching.zeilen, lt.zweimatching.runden,
                         LP.ist_ganzzahlig(lt.modell_zm.x), lt.zweimatching.status),
    }
    fehler = []
    for name, st in stufen.items():
        fehler += [f"{C.STUFEN_NAMEN[name]}: {t}" for t in pruefe_loesung(inst, st.x, opt.wert)]
    if opt.wert is not None:
        if not R.ist_rundreise(inst.n, inst.kanten, opt.tour) or R.tourlaenge(inst.kosten, opt.tour) != opt.wert:
            fehler.append("die gemeldete kürzeste Rundreise ist keine")
        fehler += pruefe_zeilen_an_tour(lt.modell_zm, opt.tour)
    return Analyse(inst, lt, stufen, opt, fehler)


def schritte_bis(a: Analyse, stufe: str) -> list:
    """Die Schritte vom Grad-LP bis zum Ende der gewählten Stufe."""
    if stufe == "grad":
        return [a.leiter.start]
    if stufe == "subtour":
        return [a.leiter.start] + a.leiter.subtour.schritte
    return a.leiter.schritte()


def verdict_stufe(a: Analyse, stufe: str) -> tuple:
    """Verdikt zur gewählten Stufe aus den gerechneten Werten."""
    jetzt = a.stufen[stufe]
    opt = a.optimum.wert
    vorher_name = {"subtour": "grad", "zm": "subtour"}.get(stufe)
    vorher = a.stufen[vorher_name] if vorher_name else None
    return verdikt_stufenwerte(stufe, None if vorher is None else (vorher.wert, vorher.ganz), jetzt.wert, jetzt.ganz, jetzt.zeilen,
                               jetzt.status, opt)


ZEILENWORT = {"subtour": ("Subtourzeile", "Subtourzeilen"), "zm": ("2-Matching-Ungleichung", "2-Matching-Ungleichungen")}


def verdikt_stufenwerte(stufe: str, vorher, wert: float, ganz: bool, zeilen: int, status: str, opt: Optional[int]) -> tuple:
    """Reine Funktion: vorher = (Wert, ganzzahlig) der vorigen Stufe oder None."""
    if status != L.FERTIG:
        return "warning", (f"Diese Stufe wurde an der Runden- oder Zeitgrenze abgebrochen: {zahl(wert)} ist nur eine untere Schranke, "
                           "das Ende ist nicht bestimmt.")
    if opt is None:
        return "info", (f"LP-Wert nach dieser Stufe: {zahl(wert)}. Das ganzzahlige Optimum ist nicht bestimmt, deshalb gibt es keine Lücke.")
    fertig = ganz and abs(wert - opt) < OPT_TOL
    if stufe == "grad":
        if fertig:
            return "success", f"Schon das Grad-LP ist ganzzahlig und optimal: Wert {zahl(wert)}. Hier muss nichts nachgeladen werden."
        if abs(wert - opt) < OPT_TOL:
            return "warning", (f"Der Wert des Grad-LP stimmt ({zahl(wert)}), die gelieferte Lösung ist aber gebrochen.")
        return "warning", (f"Das Grad-LP hat den Wert {zahl(wert)}, die kürzeste Rundreise kostet {zahl(opt)}: Lücke "
                           f"**{prozent(luecke(wert, opt))}** des Optimums.")
    wert_v, ganz_v = vorher
    if ganz_v and abs(wert_v - opt) < OPT_TOL:
        return "info", "Schon die vorige Stufe war ganzzahlig und optimal; diese Stufe lädt nichts."
    einzahl, mehrzahl = ZEILENWORT[stufe]
    zeilentext = anzahl_wort(zeilen, einzahl, mehrzahl)
    vortext = "Grad-LP" if stufe == "subtour" else "Subtour-LP"
    if fertig:
        menge = ("einer " + einzahl) if zeilen == 1 else zeilentext
        return "success", f"{vortext} {zahl(wert_v)}, mit {menge} {zahl(wert)} = Optimum, und die Lösung ist eine Rundreise."
    if abs(wert - opt) < OPT_TOL:
        return "info", (f"{vortext} {zahl(wert_v)}, nach {zeilentext} stimmt der Wert ({zahl(wert)}), die gelieferte Lösung ist aber noch "
                        "gebrochen: es gibt auch eine ganzzahlige beste Lösung, der Löser liefert eine andere.")
    if zeilen == 0:
        return "warning", (f"Diese Stufe findet keine verletzte Ungleichung, aber es fehlt noch etwas: Lücke **{prozent(luecke(wert, opt))}**.")
    return "warning", (f"Auch nach dieser Stufe fehlt etwas ({zeilentext} nachgeladen): LP-Wert {zahl(wert)}, Optimum {zahl(opt)}, Lücke "
                       f"**{prozent(luecke(wert, opt))}**.")


def verdict_gebrochen(art: str, daten) -> tuple:
    """Wie sieht eine LP-Lösung aus? (Ergebnis von rpt_lp.struktur)"""
    if art == "ganzzahlig":
        return "success", "Nichts ist gebrochen: alle Werte sind 0 oder 1."
    if art == "halb":
        nur_kreise, laengen = daten
        gruppen = ", ".join(f"{laengen.count(l)} × Länge {l}" for l in sorted(set(laengen)))
        if nur_kreise:
            if len(laengen) == 1:
                return "info", f"Nur 0, ½ und 1; die halben Kanten bilden einen Kreis durch {laengen[0]} Orte."
            return "info", f"Nur 0, ½ und 1; die halben Kanten bilden {len(laengen)} getrennte Kreise ({gruppen})."
        return "info", (f"Nur 0, ½ und 1, aber die halben Kanten bilden nicht nur getrennte Kreise (zusammenhängende Teile, Zahl der Orte: "
                        f"{', '.join(str(l) for l in laengen)}).")
    if art == "anders":
        nenner = ", ".join(str(d) for d in daten)
        return "warning", f"Nicht halbganzzahlig: hier kommen Werte mit Nenner {nenner} vor."
    return "warning", "Nicht halbganzzahlig: mindestens ein Wert ist kein Bruch mit kleinem Nenner."


def zeile_beschreiben(inst: I.Instanz, x_davor, eintrag) -> str:
    """Beschreibt eine nachgeladene Ungleichung mit ihrem Wert an der Lösung davor."""
    kanten = inst.kanten
    if isinstance(eintrag, frozenset):
        w = sum(x_davor[j] for j in LP.schnittkanten(kanten, eintrag))
        return (f"**Subtourzeile** x(δ(S)) ≥ 2 mit S = {I.menge_name(inst, eintrag)}: vorher x(δ(S)) = {bruch(w)}.")
    if isinstance(eintrag, ZM.Zeile):
        griff, zaehne, disjunkt = ZM.kammform(inst.n, kanten, eintrag.griff, eintrag.zaehne)
        links, rechts = ZM.kammwert(kanten, x_davor, griff, zaehne)
        w = ZM.wert(kanten, x_davor, griff, zaehne)
        zahn = ", ".join(I.kante_name(inst, j, kanten) for j in zaehne[:8]) + (" …" if len(zaehne) > 8 else "")
        text = (f"**2-Matching-Ungleichung** mit Griff H = {I.menge_name(inst, griff)} und {len(zaehne)} Zahnkanten ({zahn}): vorher "
                f"x(δ(H) ohne F) + Σ(1 − x) = {bruch(w)} < 1")
        if disjunkt:
            text += f"; als Kamm x(δ(H)) + Σ x(δ(T)) = {bruch(links)} < {rechts}."
        else:
            text += "."
        if tuple(zaehne) != tuple(eintrag.zaehne):
            text += (f" Gefunden wurde eine Form mit {len(eintrag.zaehne)} Zahnkanten, von denen sich zwei an einem Ort trafen; gezeigt ist die "
                     "verkürzte Form (Zahnzerlegung).")
        return text
    verletzung, griff, zaehne = eintrag
    teile = ", ".join(I.menge_name(inst, t, 8) for t in zaehne)
    return (f"**Kamm** mit Griff H = {I.menge_name(inst, griff)} und Zähnen {teile}: vorher x(δ(H)) + Σ x(δ(T)) = "
            f"{bruch(3 * len(zaehne) + 1 - verletzung)} < {3 * len(zaehne) + 1}.")


def hervorhebung(inst: I.Instanz, eintrag):
    """(Menge, Griff, Zahnkanten, Zahnmengen) für die Zeichnung einer Ungleichung."""
    if isinstance(eintrag, frozenset):
        return eintrag, None, (), ()
    if isinstance(eintrag, ZM.Zeile):
        griff, zaehne, _ = ZM.kammform(inst.n, inst.kanten, eintrag.griff, eintrag.zaehne)
        return None, griff, zaehne, ()
    _, griff, zaehne = eintrag
    return None, griff, (), tuple(zaehne)


# ---------------------------------------------------------------- Experiment: Komponentenregel gegen Padberg–Rao

@dataclass
class Regellauf:
    wert: float
    x: tuple
    zeilen: int
    runden: int
    status: str
    rest: Optional[ZM.Zeile]     # eine verletzte Ungleichung am Ende (exakt gesucht) oder None


def regellauf(a: Analyse, regel: str, modus: str) -> Regellauf:
    modell = a.leiter.modell_sub.kopie()
    v = L.zweimatching_stufe(modell, regel, modus)
    rest = ZM.trenne_padberg_rao(a.inst.n, a.inst.kanten, modell.x)
    return Regellauf(modell.wert, tuple(modell.x), v.zeilen, v.runden, v.status, rest[0] if rest else None)


def verdict_falle(regel: str, rest_wert: Optional[float], wert_regel: float, wert_exakt: float, opt: Optional[int], status: str,
                  zeilen: int) -> tuple:
    """rest_wert: Wert der am stärksten verletzten übrig gebliebenen Ungleichung (None: nichts mehr verletzt)."""
    name = C.REGEL_NAMEN[regel]
    if status != L.FERTIG:
        return "warning", f"{name}: an der Runden- oder Zeitgrenze abgebrochen, das Ende ist nicht bestimmt."
    if rest_wert is not None:
        if abs(wert_exakt - wert_regel) < OPT_TOL:
            folge = (f"Der LP-Wert stimmt hier trotzdem mit dem exakten überein ({zahl(wert_exakt, 4)}); die Regel weiß nur nicht, ob sie "
                     "fertig ist.")
        else:
            abstand = "" if opt is None else (f", {prozent(100.0 * max(0.0, wert_exakt - wert_regel) / opt, 3)} des Optimums unter dem exakten "
                                              "Wert")
            folge = f"LP-Wert {zahl(wert_regel, 4)} gegen {zahl(wert_exakt, 4)} mit exakter Trennung{abstand}."
        return "warning", f"Die {name} stoppt, aber eine 2-Matching-Ungleichung ist noch verletzt (Wert {bruch(rest_wert)} < 1). {folge}"
    if regel == L.KOMPONENTEN:
        if zeilen == 0:
            return "info", ("Hier genügt die einfache Regel: nach den Subtourzeilen ist keine 2-Matching-Ungleichung verletzt, und sie lädt "
                            "nichts.")
        return "success", f"Hier genügt die einfache Regel: nach {zeilen} Ungleichungen ist nichts mehr verletzt (LP-Wert {zahl(wert_regel, 4)})."
    if zeilen == 0:
        return "info", "Padberg–Rao findet nach den Subtourzeilen keine verletzte 2-Matching-Ungleichung und lädt nichts."
    return "success", f"Padberg–Rao ist fertig: nach {zeilen} Ungleichungen ist keine mehr verletzt (LP-Wert {zahl(wert_regel, 4)})."


# ---------------------------------------------------------------- Experiment: Kämme

def kamm_instanz_von(wahl: str, s: Settings) -> I.Instanz:
    if wahl == "prisma2":
        return I.prisma(2)
    if wahl == "prisma3":
        return I.prisma(3)
    if wahl == "offen":
        familie, n, seed = C.OFFENES_BEISPIEL
        return I.zufallsinstanz(familie, n, seed)
    return instanz_von(s)


@dataclass
class Kammlauf:
    inst: I.Instanz
    wert_sub: float
    wert_zm: float
    ganz_zm: bool
    x_zm: tuple
    schritte: list
    kaemme: int
    wert: float
    x: tuple
    ganz: bool
    status: str
    optimum: R.Optimum


def kammlauf(inst: I.Instanz) -> Kammlauf:
    if inst.n > K.MAX_N:
        raise ValueError(f"die Kammsuche ist auf {K.MAX_N} Orte begrenzt")
    lt = L.leiter(inst, modus=L.EINE)
    modell = lt.modell_zm.kopie()
    kv = L.kamm_stufe(modell)
    opt = R.optimum(inst.n, inst.kanten, inst.kosten, modell, C.ZEITGRENZE_OPT_LANG, C.ZEITGRENZE_OPT_LANG)
    start = L.Schritt("zweimatching", lt.modell_zm.wert, tuple(lt.modell_zm.x))
    return Kammlauf(inst, lt.modell_sub.wert, lt.modell_zm.wert, LP.ist_ganzzahlig(lt.modell_zm.x), tuple(lt.modell_zm.x),
                    [start] + kv.schritte, kv.zeilen, modell.wert, tuple(modell.x), LP.ist_ganzzahlig(modell.x), kv.status, opt)


def verdict_kamm(wert_zm: float, ganz_zm: bool, wert: float, ganz: bool, kaemme: int, status: str, opt: Optional[int]) -> tuple:
    if status != L.FERTIG:
        return "warning", "Die Kammsuche wurde an der Runden- oder Zeitgrenze abgebrochen: das Ende ist nicht bestimmt."
    if opt is None:
        return "info", f"Alle Kämme sind erfüllt, der LP-Wert ist {zahl(wert)}; das Optimum ist nicht bestimmt."
    if ganz_zm and abs(wert_zm - opt) < OPT_TOL:
        return "info", f"Schon die 2-Matching-Ungleichungen genügten: {zahl(wert_zm)} = Optimum; es wird kein Kamm nachgeladen."
    if ganz and abs(wert - opt) < OPT_TOL:
        return "success", f"Mit {kaemme} {'Kamm' if kaemme == 1 else 'Kämmen'} gelöst: Wert {zahl(wert)} = Optimum, die Lösung ist eine Rundreise."
    if abs(wert - opt) < OPT_TOL:
        return "info", (f"Alle Kämme sind erfüllt, und der Wert stimmt ({zahl(wert)}); die gelieferte Lösung ist aber gebrochen.")
    return "warning", (f"Alle Kämme sind erfüllt ({kaemme} nachgeladen), der Wert ist **{zahl(wert)}**, das Optimum **{zahl(opt)}**. "
                     "Dafür braucht es weitere Ungleichungen.")


# ---------------------------------------------------------------- vorgerechnete Reihen

@lru_cache(maxsize=1)
def tabellen() -> dict:
    return json.loads(TABELLEN_DATEI.read_text(encoding="utf-8"))


def leiter_zeile(familie: str, n: int):
    return tabellen()["leiter"].get(f"{familie}-{n}")


def reihen_name(zeile: dict) -> str:
    return f"{C.FAMILIEN_NAMEN[zeile['familie']]}, {zeile['n']} Orte"


def verdict_stufentabelle(zeile: dict) -> tuple:
    """Wie weit kommt jede Stufe? Aus einer Tabellenzeile."""
    g, anzahl = zeile["geloest"], zeile["instanzen"]
    text = (f"**{reihen_name(zeile)}:** ganzzahlig und optimal {g['grad']} / {g['subtour']} / {g['exakt']} von {anzahl} "
            "(Grad-LP / alle Subtourzeilen / alle 2-Matching-Ungleichungen).")
    grad_luecke, sub_luecke = zeile["luecke_mittel"]["grad"], zeile["luecke_mittel"]["subtour"]
    if grad_luecke >= 10:
        text += (f" Das Grad-LP liegt im Mittel {prozent(grad_luecke, 1)} unter dem Optimum, nach den Subtourzeilen sind es noch "
                 f"{prozent(sub_luecke)}.")
    else:
        text += f" Mittlere Lücke {prozent(grad_luecke)} / {prozent(sub_luecke)} / {prozent(zeile['luecke_mittel']['exakt'], 3)}."
    if g["exakt"] == anzahl:
        return "success", text + " Die 2-Matching-Ungleichungen genügen hier auf allen Instanzen."
    return "info", text + f" Auf {anzahl - g['exakt']} Instanzen bleibt danach etwas offen."


def anteile_ueber_n(familie: str) -> list:
    """[(n, Anteil ganzzahlig und optimal in % je Stufe, Instanzen)] einer Familie, nach n sortiert."""
    zeilen = [z for z in tabellen()["leiter"].values() if z["familie"] == familie]
    return [(z["n"], {s: 100.0 * z["geloest"][s] / z["instanzen"] for s in z["geloest"]}, z["instanzen"])
            for z in sorted(zeilen, key=lambda z: z["n"])]


def verdict_groesse(familie: str, anteile: list) -> tuple:
    """Bleibt mit der Größe mehr offen? Prüft, ob der Anteil nach der exakten 2-Matching-Stufe in jedem Schritt fällt."""
    werte = [(n, a["exakt"]) for n, a, _ in anteile]
    verlauf = ", ".join(f"{n}: {prozent(w, 1)}" for n, w in werte)
    steigt = [(a, b) for a, b in zip(werte, werte[1:]) if b[1] >= a[1]]
    name = C.FAMILIEN_NAMEN[familie]
    if not steigt:
        return "info", (f"{name}: Der Anteil ganzzahlig und optimal gelöster Instanzen nach allen 2-Matching-Ungleichungen fällt in jedem "
                        f"gemessenen Schritt ({verlauf}).")
    stellen = "; ".join(f"von {a[0]} auf {b[0]} Orte nicht ({prozent(a[1], 1)} → {prozent(b[1], 1)})" for a, b in steigt)
    return "warning", f"{name}: Der Anteil fällt nicht in jedem Schritt: {stellen}. Gemessener Verlauf: {verlauf}."


def verdict_komponenten(zeilen: list) -> tuple:
    """Zählung der Falle über alle Reihen und der größte Wertunterschied mit „eine je Runde“."""
    schlimmste = max(zeilen, key=lambda z: z["komponentenregel"]["eine"]["abstand_max"])
    abstand = schlimmste["komponentenregel"]["eine"]["abstand_max"]
    gesamt = sum(z["komponentenregel"]["alle"]["uebersehen"] for z in zeilen)
    instanzen = sum(z["instanzen"] for z in zeilen)
    return "info", (f"Über die {len(zeilen)} Messreihen endet die Komponentenregel (alle je Runde) auf {gesamt} von {instanzen} Instanzen, obwohl "
                    f"noch eine 2-Matching-Ungleichung verletzt ist. Der Wertunterschied bleibt klein: mit „eine je Runde“ höchstens "
                    f"{prozent(abstand)} des Optimums ({reihen_name(schlimmste)}).")


def verdict_facetten(zeile: dict) -> tuple:
    """Facettenzahl gegen nachgeladene Zeilen an der Reihe `zeile` (Karte mit den meisten Orten)."""
    n = zeile["n"]
    median = zeile["subtourzeilen"]["alle_median"]
    return "info", (f"Bei {n} Orten gibt es 2^{n - 1} − {n + 1} Subtourfacetten ({ganzzahl(2 ** (n - 1) - n - 1)}). Auf den "
                    f"{zeile['instanzen']} Karten dieser Größe wurden davon im Median {zahl(median, 1)} nachgeladen (alle je Runde, "
                    f"größte {zeile['subtourzeilen']['alle_max']}); wie viele es werden, hängt an der Auswahlregel (eine je Runde: Median "
                    f"{zahl(zeile['subtourzeilen']['eine_median'], 1)}).")


# ---------------------------------------------------------------- Reiter: Permutaeder aus einem Komparatornetz

@dataclass
class Netzbild:
    n: int
    art: str
    komparatoren: list           # alle
    entfernt: Optional[int]      # Index des gestrichenen Komparators
    rest: list                   # ohne den gestrichenen
    ungleichungen: int
    variablen: int
    gleichungen: int
    facetten: int
    ziel: tuple
    lp_wert: float
    lp_z: tuple
    sort_wert: float
    sortiert: bool
    drin: Optional[list]         # Permutationen in der Projektion (nur n <= 5)
    draussen: Optional[list]


def netz_ziel(n: int, seed: int) -> tuple:
    """Zufällige ganzzahlige Zielkoeffizienten 0 .. 100."""
    from rpt_rng import SplitMix64
    rng = SplitMix64(seed)
    return tuple(rng.below(101) for _ in range(n))


def netzbild(n: int, art: str, entfernt: Optional[int], seed: int) -> Netzbild:
    voll = NZ.netz(art, n)
    if entfernt is not None and not 0 <= entfernt < len(voll):
        entfernt = None
    rest = voll if entfernt is None else voll[:entfernt] + voll[entfernt + 1:]
    f = NZ.netz_formulierung(rest, n)
    ziel = netz_ziel(n, seed)
    lp_wert, lp_z = NZ.netz_lp(rest, n, ziel)
    sort_wert, _ = NZ.sortier_optimum(ziel)
    drin = draussen = None
    if n <= 5:
        drin, draussen = NZ.sortierte_permutationen(rest, n)
    return Netzbild(n, art, voll, entfernt, rest, f.ungleichungen, f.variablen, f.gleichungen, NZ.permutaeder_facetten(n), ziel, lp_wert,
                    tuple(lp_z), sort_wert, NZ.sortiert_alles(rest, n), drin, draussen)


def verdict_netz(n: int, ungleichungen: int, variablen: int, gleichungen: int, facetten: int, entfernt: bool, sortiert: bool,
                 drin: Optional[int], gesamt: Optional[int], lp_wert: float, sort_wert: float) -> tuple:
    gleich = abs(lp_wert - sort_wert) < 1e-6
    if not entfernt or sortiert:
        groesse = f"{ungleichungen} Ungleichungen in {variablen} Variablen ({gleichungen} Gleichungen) statt {ganzzahl(facetten)} Facetten"
        kopf = "Der gestrichene Komparator war hier überflüssig, das Netz sortiert weiter. " if entfernt else ""
        if drin is not None:
            kopf += f"Alle {drin} Permutationen sind in der Projektion: sie ist das Permutaeder. "
        if ungleichungen >= facetten:
            return "warning", (kopf + f"{ungleichungen} Ungleichungen gegen {facetten} Facetten: Hier spart die Erweiterung nichts.")
        wert = ("der LP-Wert ist der durch Sortieren bestimmte" if gleich else "der LP-Wert weicht vom Sortieren ab (Prüfung fehlgeschlagen)")
        return ("success" if gleich else "error"), kopf + f"{groesse}; {wert}."
    wirkung = (f"Der LP-Wert {zahl(lp_wert)} ist dann größer als das Minimum {zahl(sort_wert)} über das ganze Permutaeder."
               if lp_wert > sort_wert + 1e-6 else
               f"Für diese Zielkoeffizienten fällt es nicht auf: der LP-Wert ist trotzdem {zahl(lp_wert)}, wie durch Sortieren.")
    if drin is not None:
        return "warning", (f"Die Projektion enthält nur noch **{drin} von {gesamt}** Permutationen, genau die, die das Netz noch sortiert. "
                         f"Sie wird zu klein, nie zu groß. {wirkung}")
    return "warning", (f"Das Netz sortiert nicht mehr alle Eingaben (0-1-Prinzip), also fehlen der Projektion Permutationen: sie ist echt kleiner als "
                     f"das Permutaeder, nie größer. {wirkung}")
