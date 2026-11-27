"""Die Leiter: Grad-LP, dann Subtourzeilen, dann 2-Matching-Ungleichungen, dann (bis 12 Orte) Kämme. Nur Komposition der Trennungen.

Auswahlregel je Runde: „alle“ lädt alle gefundenen verletzten Ungleichungen nach, „eine“ nur die erste in der festen Reihenfolge der jeweiligen
Trennung. Nach jeder Runde 2-Matching oder Kamm wird zuerst wieder bis zur Subtourschranke nachgeladen. Jede Rechnung hat eine Rundengrenze und
eine Zeitgrenze; wird eine erreicht, ist der Status „abgebrochen“ und der Wert nur eine untere Schranke."""

import time
from dataclasses import dataclass, field

import rpt_kamm as K
import rpt_lp as LP
import rpt_subtour as ST
import rpt_zweimatching as ZM

ALLE, EINE = "alle", "eine"
PADBERG_RAO, KOMPONENTEN = "pr", "komp"
FERTIG, ABGEBROCHEN = "fertig", "abgebrochen"
MAX_RUNDEN = 600
ZEITGRENZE = 20.0


@dataclass(frozen=True)
class Schritt:
    stufe: str            # 'grad', 'subtour', 'zweimatching', 'kamm'
    wert: float
    x: tuple
    neu: tuple = ()       # neu geladene Ungleichungen: Menge (frozenset), Zeile (ZM.Zeile) oder (Verletzung, Griff, Zähne)


@dataclass
class Verlauf:
    """Die Schritte einer Stufe samt Zählern."""
    schritte: list = field(default_factory=list)
    status: str = FERTIG
    zeilen: int = 0                       # nachgeladene Zeilen dieser Stufe
    runden: int = 0                       # Runden dieser Stufe
    subtourzeilen: int = 0                # Subtourzeilen, die während dieser Stufe nachgeladen wurden


class Uhr:
    def __init__(self, sekunden: float):
        self.ende = time.perf_counter() + sekunden

    def abgelaufen(self) -> bool:
        return time.perf_counter() > self.ende


def grad_stufe(modell: LP.Modell) -> Schritt:
    wert, x = modell.loese()
    return Schritt("grad", wert, tuple(x))


def subtour_stufe(modell: LP.Modell, modus: str = ALLE, uhr: Uhr = None, stufe: str = "subtour") -> Verlauf:
    """Lädt Subtourzeilen nach, bis keine mehr verletzt ist."""
    uhr = uhr or Uhr(ZEITGRENZE)
    v = Verlauf()
    while True:
        verletzt = ST.verletzte_subtours(modell.n, modell.kanten, modell.x)
        if not verletzt:
            return v
        if v.runden >= MAX_RUNDEN or uhr.abgelaufen():
            v.status = ABGEBROCHEN
            return v
        if modus == EINE:
            verletzt = verletzt[:1]
        neu = [s for _, s in verletzt if modell.add(("S", s), *ST.zeile(modell.kanten, s))]
        if not neu:
            v.status = ABGEBROCHEN      # verletzte Zeile ist schon im LP: Löser-Toleranz, nicht weiterlaufen
            return v
        v.zeilen += len(neu)
        v.subtourzeilen += len(neu)
        v.runden += 1
        wert, x = modell.loese()
        v.schritte.append(Schritt(stufe, wert, tuple(x), tuple(neu)))


def trenne(regel: str, modell: LP.Modell) -> list:
    if regel == KOMPONENTEN:
        return ZM.trenne_komponentenregel(modell.n, modell.kanten, modell.x)
    return ZM.trenne_padberg_rao(modell.n, modell.kanten, modell.x)


def zweimatching_stufe(modell: LP.Modell, regel: str = PADBERG_RAO, modus: str = ALLE, subtour_modus: str = ALLE, uhr: Uhr = None) -> Verlauf:
    """Lädt 2-Matching-Ungleichungen nach, bis die Trennung nichts mehr findet. Voraussetzung: das LP steht an der Subtourschranke oder wird dort
    in jeder Runde zuerst hingebracht."""
    uhr = uhr or Uhr(ZEITGRENZE)
    v = Verlauf()
    while True:
        sub = subtour_stufe(modell, subtour_modus, uhr, stufe="zweimatching")
        v.schritte += sub.schritte
        v.subtourzeilen += sub.subtourzeilen
        if sub.status != FERTIG:
            v.status = sub.status
            return v
        gefunden = trenne(regel, modell)
        if not gefunden:
            return v
        if v.runden >= MAX_RUNDEN or uhr.abgelaufen():
            v.status = ABGEBROCHEN
            return v
        if modus == EINE:
            gefunden = gefunden[:1]
        neu = [z for z in gefunden if modell.add(("B", z.griff, z.zaehne), *ZM.zeile(modell.kanten, z.griff, z.zaehne))]
        if not neu:
            v.status = ABGEBROCHEN
            return v
        v.zeilen += len(neu)
        v.runden += 1
        wert, x = modell.loese()
        v.schritte.append(Schritt("zweimatching", wert, tuple(x), tuple(neu)))


def kamm_stufe(modell: LP.Modell, subtour_modus: str = ALLE, uhr: Uhr = None) -> Verlauf:
    """Bis 12 Orte: exakt 2-Matching (eine je Runde), dann je Runde den am stärksten verletzten Kamm, bis kein Kamm mehr verletzt ist."""
    uhr = uhr or Uhr(ZEITGRENZE)
    v = Verlauf()
    while True:
        zm = zweimatching_stufe(modell, PADBERG_RAO, EINE, subtour_modus, uhr)
        v.schritte += [Schritt("kamm", s.wert, s.x, s.neu) for s in zm.schritte]
        v.subtourzeilen += zm.subtourzeilen
        if zm.status != FERTIG:
            v.status = zm.status
            return v
        kamm = K.trenne_kamm(modell.n, modell.kanten, modell.x)
        if kamm is None:
            return v
        if v.runden >= MAX_RUNDEN or uhr.abgelaufen():
            v.status = ABGEBROCHEN
            return v
        _, griff, zaehne = kamm
        if not modell.add(("K", griff, tuple(zaehne)), *K.zeile(modell.kanten, griff, zaehne)):
            v.status = ABGEBROCHEN
            return v
        v.zeilen += 1
        v.runden += 1
        wert, x = modell.loese()
        v.schritte.append(Schritt("kamm", wert, tuple(x), ((kamm[0], griff, tuple(zaehne)),)))


def neues_modell(inst) -> LP.Modell:
    return LP.Modell(inst.n, inst.kanten, inst.kosten)


@dataclass
class Leiter:
    """Die ersten drei Stufen einer Instanz mit den Modellen nach jeder Stufe."""
    start: Schritt
    subtour: Verlauf
    zweimatching: Verlauf
    modell_sub: LP.Modell
    modell_zm: LP.Modell

    @property
    def wert_grad(self) -> float:
        return self.start.wert

    @property
    def wert_subtour(self) -> float:
        return self.modell_sub.wert

    @property
    def wert_zweimatching(self) -> float:
        return self.modell_zm.wert

    def schritte(self) -> list:
        return [self.start] + self.subtour.schritte + self.zweimatching.schritte

    def status(self) -> str:
        return FERTIG if self.subtour.status == FERTIG and self.zweimatching.status == FERTIG else ABGEBROCHEN


def leiter(inst, subtour_modus: str = ALLE, regel: str = PADBERG_RAO, modus: str = ALLE) -> Leiter:
    modell = neues_modell(inst)
    start = grad_stufe(modell)
    sub = subtour_stufe(modell, subtour_modus)
    nach_sub = modell.kopie()
    zm = zweimatching_stufe(modell, regel, modus, subtour_modus)
    return Leiter(start, sub, zm, nach_sub, modell)
