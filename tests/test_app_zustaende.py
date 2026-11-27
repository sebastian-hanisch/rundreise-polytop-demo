"""App in allen Familien, an den Reglerrändern, mit allen Auswahlfeldern, Kammknopf, Reiter, PDF; dazu Textprüfungen am App-Quelltext."""

import os
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rpt_constants as C
import rpt_evaluation as E
import rpt_leiter as L
import rpt_netz as NZ
import rpt_pdf_export as PDF

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _ok(at):
    """Weder Ausnahme noch rote Box: „error“ ist gescheiterten Prüfungen vorbehalten, Befunde sind Warnungen."""
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _app(query=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    return at


def _text(at) -> str:
    teile = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for art in (at.success, at.info, at.warning, at.error):
        teile += [e.value for e in art]
    return " ".join(teile)


@pytest.mark.parametrize("query", [
    {"familie": "karte", "n": "5", "seed": "0"},
    {"familie": "karte", "n": "50", "seed": "999999"},
    {"familie": "haufen", "n": "5", "seed": "3"},
    {"familie": "haufen", "n": "50", "seed": "1"},
    {"familie": "matrix", "n": "5", "seed": "1"},
    {"familie": "matrix", "n": "13", "seed": "4"},
    {"familie": "matrix", "n": "50", "seed": "9"},
    {"familie": "prisma", "k": "1", "dreieck": "1", "weg": "1"},
    {"familie": "prisma", "k": "4", "dreieck": "5", "weg": "5"},
])
def test_raender_der_regler(query):
    _app(query)


def test_alle_stufen_und_nachladeregeln():
    at = _app({"familie": "karte", "n": "30", "seed": "34074"})
    for stufe in C.STUFEN:
        for modus in C.MODI:
            at.radio(key="stufe_radio").set_value(stufe).run()
            at.radio(key="subtourmodus_radio").set_value(modus).run()
            _ok(at)
    schieber = next(s for s in at.slider if s.label.startswith("Schritt ("))
    for wert in (0, 1, schieber.max):
        next(s for s in at.slider if s.label.startswith("Schritt (")).set_value(int(wert)).run()
        _ok(at)


def test_auswahl_einer_ungleichung_in_einer_runde():
    at = _app({"familie": "karte", "n": "30", "seed": "34074"})
    next(s for s in at.slider if s.label.startswith("Schritt (")).set_value(1).run()
    boxen = [b for b in at.selectbox if b.label == "Welche Ungleichung dieser Runde hervorheben?"]
    assert boxen
    boxen[0].set_value(1).run()
    _ok(at)


def test_falle_alle_regeln_und_auswahlen_mit_karte_30():
    at = _app({"familie": "karte", "n": "30", "seed": "34074"})
    for regel in C.REGELN:
        for modus in C.MODI:
            at.radio(key="regel_radio").set_value(regel).run()
            at.radio(key="modus_radio").set_value(modus).run()
            _ok(at)
            erwartet = "stoppt, aber eine 2-Matching-Ungleichung ist noch verletzt" in _text(at)
            assert erwartet == (regel == L.KOMPONENTEN)


def test_alle_messreihen_und_familien_in_experiment_zwei():
    at = _app()
    for key in E.tabellen()["leiter"]:
        next(s for s in at.selectbox if s.label == "Messreihe").set_value(key).run()
        _ok(at)
    for fam in ("karte", "haufen", "matrix"):
        next(s for s in at.selectbox if s.label == "Über die Größe (Familie)").set_value(fam).run()
        _ok(at)


def test_messreihe_passt_sich_der_instanz_an():
    at = _app({"familie": "haufen", "n": "30", "seed": "36000"})
    assert next(s for s in at.selectbox if s.label == "Messreihe").value == "haufen-30"
    assert "Vorgerechnet** (Haufen, 30 Orte" in _text(at)
    at = _app({"familie": "haufen", "n": "31", "seed": "1"})
    assert next(s for s in at.selectbox if s.label == "Messreihe").value == "karte-50"
    assert "keine vorgerechnete Reihe" in _text(at)


def test_kammknopf_fuer_alle_beispiele():
    at = _app({"familie": "matrix", "n": "10", "seed": "33022"})
    for wahl in C.KAMM_BEISPIELE:
        at.selectbox(key="kamm_beispiel").set_value(wahl).run()
        at.button(key="kamm_knopf").click().run()
        _ok(at)
        assert any(m.label == "Nachgeladene Kämme" for m in at.metric)


def test_kammknopf_gesperrt_ueber_zwoelf_orten():
    at = _app({"familie": "karte", "n": "13", "seed": "1"})
    assert at.button(key="kamm_knopf").disabled
    assert "Gesperrt: die Instanz links hat 13 Orte" in _text(at)
    at.selectbox(key="kamm_beispiel").set_value("prisma3").run()
    assert not at.button(key="kamm_knopf").disabled
    at.button(key="kamm_knopf").click().run()
    _ok(at)
    assert "Alle Kämme sind erfüllt" in _text(at) and "13,33" in _text(at)


def test_kammergebnis_verschwindet_nach_wechsel_der_instanz():
    at = _app({"familie": "matrix", "n": "10", "seed": "33022"})
    at.button(key="kamm_knopf").click().run()
    assert any(m.label == "Nachgeladene Kämme" for m in at.metric)
    at.sidebar.number_input(key="seed_input").set_value(1).run()
    assert not any(m.label == "Nachgeladene Kämme" for m in at.metric)


def test_huellen_auswahl_und_netzreiter():
    at = _app()
    for n in (5, 6, 7):
        at.radio(key="huelle_n").set_value(n).run()
        _ok(at)
    for art in NZ.NETZE:
        at.radio(key="netz_art").set_value(art).run()
        for n in (3, 4, 5, 6, 16):
            at.select_slider(key="netz_n").set_value(n).run()
            _ok(at)
            weg = next(s for s in at.selectbox if s.label == "Gestrichener Komparator")
            weg.set_value(len(NZ.netz(art, n)) - 1).run()
            _ok(at)
    at.number_input(key="netz_seed").set_value(17).run()
    _ok(at)


def test_netzreiter_verdikte_in_der_app():
    at = _app()
    assert "10 Ungleichungen in 10 Variablen (5 Gleichungen) statt 14 Facetten" in _text(at)
    next(s for s in at.selectbox if s.label == "Gestrichener Komparator").set_value(4).run()
    assert "**16 von 24** Permutationen" in _text(at)
    at.select_slider(key="netz_n").set_value(16).run()
    assert "126 Ungleichungen in 126 Variablen (63 Gleichungen) statt 65.534 Facetten" in _text(at)


def test_pdf_knopf_und_inhalt():
    at = _app({"familie": "prisma", "k": "1", "dreieck": "2", "weg": "1"})
    at.button(key="pdf_knopf").click().run()
    _ok(at)
    assert at.get("download_button")
    a = E.analyse(E.Settings("prisma", 0, 0, 1, 2, 1))
    daten = PDF.erzeuge_pdf(a, "Prisma – Test mit „Anführungszeichen“ und ½ ⅓", [(s, E.verdict_stufe(a, s)[1]) for s in C.STUFEN])
    assert daten[:4] == b"%PDF" and len(daten) > 1000


def test_optimum_nicht_bestimmt_zeigt_keine_luecke_und_den_knopf(monkeypatch):
    """Meldet das MILP die Zeitgrenze, zeigt die App „nicht bestimmt“, keine Lücke und den Knopf für den längeren Lauf. Die Instanz hat nach
    allen Stufen eine gebrochene Lösung, sodass das MILP wirklich gebraucht wird."""
    import streamlit as st

    import rpt_referenz as R

    class Zeitgrenze:
        status, x = 1, None

    aufrufe = []
    monkeypatch.setattr(R, "milp", lambda *a, **k: aufrufe.append(k) or Zeitgrenze())
    st.cache_data.clear()
    try:
        a = E.analyse(E.Settings("matrix", 12, 33398).bereinigt(), 1.0)
        assert aufrufe and a.optimum.wert is None and a.optimum.status == R.UNBESTIMMT and a.optimum.grund == "Zeitgrenze"
        assert E.verdict_stufe(a, "zm")[0] == "info" and "keine Lücke" in E.verdict_stufe(a, "zm")[1]
        at = _app({"familie": "matrix", "n": "12", "seed": "33398"})
        assert [m.value for m in at.metric if m.label == "Kürzeste Rundreise"][0] == "nicht bestimmt"
        assert "Kürzeste Rundreise nicht bestimmt (Zeitgrenze)." in _text(at)
        assert any("zeigt die App keine Lücke" in w.value for w in at.warning)
        assert all(not m.delta for m in at.metric if m.label.startswith(("Nur Grad", "+ ")))
        monkeypatch.undo()
        at.button(key="opt_knopf").click().run()
        _ok(at)
        assert [m.value for m in at.metric if m.label == "Kürzeste Rundreise"][0] == "2113"
        assert not [b for b in at.button if b.key == "opt_knopf"]          # belegt: kein Knopf mehr
    finally:
        st.cache_data.clear()


def test_belegte_instanz_zeigt_den_beleg_und_keinen_knopf():
    at = _app({"familie": "karte", "n": "30", "seed": "34074"})
    assert "belegt durch eine selbst nachgerechnete Dualschranke" in _text(at)
    assert not [b for b in at.button if b.key == "opt_knopf"]


def _app_quelltext() -> str:
    return (ROOT / "app.py").read_text(encoding="utf-8")


def test_keine_planungswoerter_und_keine_umgangssprache_in_texten():
    verboten = [r"\bPhase 0\b", r"\bVormessung\b", r"\bStück\b", r"\bPlan\b", r"\bLinie\b", r"läuft lokal", r"\bträgt\b", r"\btragen\b",
                r"Saatwert", r"\bV\d\b", r"\bB\d\b", r"\bZ\d\b"]
    dateien = [ROOT / "app.py", ROOT / "README.md", *sorted(ROOT.glob("rpt_*.py")), *sorted((ROOT / "tools").glob("*.py")),
               *sorted((ROOT / "tests").glob("*.py"))]
    eigene = ROOT / "tests" / "test_app_zustaende.py"
    for datei in dateien:
        if datei == eigene or datei.name in ("test_footer.py", "test_runtime_imports.py", "test_modul_frisch.py", "test_app.py"):
            continue
        text = datei.read_text(encoding="utf-8")
        if datei.name == "README.md":
            text = text.split("\n---\n")[0]          # die Footer-Zeile stammt vom Footer-Werkzeug
        for muster in verboten:
            treffer = re.search(muster, text)
            assert not treffer, f"{datei.name}: {muster} ({text[max(0, treffer.start() - 40): treffer.end() + 40]!r})"


def test_prozentzeichen_in_formeln_sind_geschuetzt():
    for block in re.findall(r"\$\$(.+?)\$\$", _app_quelltext(), flags=re.S):
        assert not re.search(r"(?<!\\)%", block), block


def test_je_displaygleichung_ein_eigener_block():
    """Kein $$-Block enthält ein weiteres $ (KaTeX-Fehler, nur im Browser sichtbar)."""
    for block in re.findall(r"\$\$(.+?)\$\$", _app_quelltext(), flags=re.S):
        assert "$" not in block


def test_links_zeigen_nur_auf_erlaubte_repos():
    erlaubt = {"matching-polytop-demo", "gomory-hu-demo", "zusammenhang-verstaerken-demo", "ganzzahlige-ecken-demo", "matroid-polytop-demo",
               "zuordnung-vs-rundreise-demo", "chvatal-gomory-demo", "lin-kernighan-demo", "rundreise-polytop-demo"}
    for datei in (ROOT / "app.py", ROOT / "README.md"):
        for repo in re.findall(r"github\.com/sebastian-hanisch/([a-z0-9-]+)", datei.read_text(encoding="utf-8")):
            assert repo in erlaubt, (datei.name, repo)
