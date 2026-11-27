"""PDF-Bericht einer Leiter (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import rpt_constants as C
from rpt_evaluation import luecke, prozent, referenz_text, zahl

_ERSATZ = {"–": "-", "—": "-", "„": '"', "“": '"', "⅓": "1/3", "⅔": "2/3", "·": "*", "≤": "<=", "≥": ">=", "→": "->", "−": "-",
           "δ": "delta", "Σ": "Summe", "∈": "in", "…": "...", "‚": "'", "‘": "'", " ": " "}


def latin1(text: str) -> str:
    for alt, neu in _ERSATZ.items():
        text = text.replace(alt, neu)
    return text.encode("latin-1", "replace").decode("latin-1")


def _absatz(pdf, text, hoehe=6, **kw):
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, hoehe, latin1(text), **kw)


def erzeuge_pdf(a, instanz_text: str, verdikte: list) -> bytes:
    """`verdikte`: [(Stufe, Text)] der drei Stufen."""
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    _absatz(pdf, "Rundreise-Polytop: Subtour, 2-Matching, Kämme", 8)
    pdf.set_font("Helvetica", "", 11)
    _absatz(pdf, f"Instanz: {instanz_text}.")
    opt = a.optimum.wert
    _absatz(pdf, referenz_text(a.optimum).replace("**", ""))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    _absatz(pdf, "Die Leiter (alle je Runde)", 7)
    pdf.set_font("Helvetica", "", 10)
    for stufe in C.STUFEN:
        s = a.stufen[stufe]
        gap = luecke(s.wert, opt)
        zeile = f"{C.STUFEN_NAMEN[stufe]}: LP-Wert {zahl(s.wert, 4)}"
        if gap is not None:
            zeile += f", Lücke {prozent(gap)}"
        zeile += f", nachgeladen {s.zeilen} Zeilen in {s.runden} Runden, Lösung {'ganzzahlig' if s.ganz else 'gebrochen'}."
        _absatz(pdf, zeile, 5)
    pdf.ln(2)
    for stufe, text in verdikte:
        _absatz(pdf, f"{C.STUFEN_NAMEN[stufe]}: {text.replace('**', '')}", 5)
    if a.optimum.tour:
        pdf.ln(2)
        namen = ", ".join(f"{a.inst.namen[a.inst.kanten[j][0]]}-{a.inst.namen[a.inst.kanten[j][1]]}" for j in a.optimum.tour)
        _absatz(pdf, f"Kanten einer kürzesten Rundreise: {namen}", 5)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    _absatz(pdf, "Welche Ecke der Löser (HiGHS, dualer Simplex) bei mehreren Optima liefert, kann vom System abhängen; Zahl und Reihenfolge "
                 "der nachgeladenen Zeilen können sich dann unterscheiden. sebastianhanisch.net", 5)
    return bytes(pdf.output())
