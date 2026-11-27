"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Je nach Familie sind Regler ausgeblendet; jeder Regler wird deshalb von einem Preset aus geändert, bei dem er sichtbar ist."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import rpt_constants as C
import rpt_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
SICHTBAR_IN = {"familie_select": "Prisma: Subtour fehlt", "n_slider": "Karte 30", "seed_input": "Karte 30",
               "k_slider": "Prisma: Subtour fehlt", "dreieck_slider": "Prisma: Subtour fehlt", "weg_slider": "Prisma: Subtour fehlt"}


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def _neuer_wert(key, regler):
    if key == "familie_select":
        return "matrix"          # Standard-Zufallsinstanz: kein Preset
    if key == "seed_input":
        return regler.value + 1
    spec = P.SETTING_SPECS[key]
    return regler.value + 1 if regler.value + 1 <= spec.hi else regler.value - 1


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    for name in C.PRESETS:
        at = _app()
        at.sidebar.button(key=f"preset_{name}").click().run()
        schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
        assert schluessel <= set(P.SETTING_SPECS), schluessel - set(P.SETTING_SPECS)
        assert schluessel == {"familie_select", *P.SICHTBAR[C.PRESETS[name]["familie"]]}, name


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    name = SICHTBAR_IN[state_key]
    at.sidebar.button(key=f"preset_{name}").click().run()
    regler = _regler(at, state_key)
    regler.set_value(_neuer_wert(state_key, regler)).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_ausgeblendete_regler_behalten_ihren_wert():
    at = _app()
    at.sidebar.button(key="preset_Karte 30").click().run()
    _regler(at, "n_slider").set_value(17).run()
    _regler(at, "familie_select").set_value("prisma").run()
    _regler(at, "familie_select").set_value("karte").run()
    assert not at.exception
    assert _regler(at, "n_slider").value == 17


def test_permalink_wird_begrenzt():
    at = AppTest.from_file(APP, default_timeout=300)
    for key, wert in {"familie": "karte", "n": "500", "seed": "-4"}.items():
        at.query_params[key] = wert
    at.run()
    assert not at.exception
    assert _regler(at, "n_slider").value == C.N_MAX and _regler(at, "seed_input").value == 0


def test_unsinn_im_permalink_wird_ignoriert():
    at = AppTest.from_file(APP, default_timeout=300)
    for key, wert in {"familie": "quark", "k": "abc"}.items():
        at.query_params[key] = wert
    at.run()
    assert not at.exception and _regler(at, "familie_select").value == "prisma"
