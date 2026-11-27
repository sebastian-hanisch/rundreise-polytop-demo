"""Presets: vollständig, innerhalb der Reglergrenzen, und jedes zeigt, was sein Hilfetext sagt (geprüft an LP-Werten, die nicht vom Löser abhängen,
wo das möglich ist)."""

import pytest

import rpt_constants as C
import rpt_evaluation as E
import rpt_leiter as L
import rpt_presets as P


def einstellungen(name) -> E.Settings:
    p = C.PRESETS[name]
    return E.Settings(**p).bereinigt()


def analyse(name):
    return E.analyse(einstellungen(name), 30.0)


def test_jedes_preset_hat_hilfe_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_das_erste_preset_ist_die_voreinstellung():
    erste = next(iter(C.PRESETS.values()))
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == erste[key] or state_key in ("n_slider", "seed_input")
    assert E.Settings().bereinigt() == einstellungen(next(iter(C.PRESETS)))


def test_prisma_subtour_fehlt():
    a = analyse("Prisma: Subtour fehlt")
    assert (a.stufen["grad"].wert, a.stufen["subtour"].wert, a.optimum.wert) == (pytest.approx(6), pytest.approx(12), 12)
    assert a.stufen["subtour"].zeilen == 1 and a.stufen["zm"].zeilen == 0
    assert E.verdict_stufe(a, "subtour") == ("success", "Grad-LP 6, mit einer Subtourzeile 12 = Optimum, und die Lösung ist eine Rundreise.")
    assert E.verdict_stufe(a, "zm") == ("info", "Schon die vorige Stufe war ganzzahlig und optimal; diese Stufe lädt nichts.")


def test_prisma_2matching_fehlt():
    a = analyse("Prisma: 2-Matching fehlt")
    assert a.stufen["subtour"].zeilen == 0 and a.stufen["subtour"].wert == pytest.approx(9)
    assert a.stufen["zm"].wert == pytest.approx(10) and a.stufen["zm"].ganz and a.optimum.wert == 10
    assert E.verdict_stufe(a, "zm")[0] == "success"
    assert E.verdict_stufe(a, "subtour")[0] == "warning" and "10,00 %" in E.verdict_stufe(a, "subtour")[1]


def test_karte_30_jede_stufe_laedt_und_die_komponentenregel_bleibt_stehen():
    a = analyse("Karte 30")
    assert a.stufen["grad"].wert < a.stufen["subtour"].wert - 1 and a.stufen["subtour"].wert < a.stufen["zm"].wert - 1
    assert a.stufen["subtour"].zeilen > 0 and a.stufen["zm"].zeilen > 0
    assert not a.stufen["subtour"].ganz
    assert a.stufen["zm"].ganz and a.stufen["zm"].wert == pytest.approx(a.optimum.wert)
    r = E.regellauf(a, L.KOMPONENTEN, L.ALLE)
    assert r.rest is not None and r.wert < a.stufen["zm"].wert - 1


def test_haufen_30_die_subtourzeilen_schliessen_alles():
    a = analyse("Haufen 30")
    assert E.luecke(a.stufen["grad"].wert, a.optimum.wert) > 20
    assert a.stufen["subtour"].ganz and a.stufen["subtour"].wert == pytest.approx(a.optimum.wert)
    assert a.stufen["zm"].zeilen == 0


def test_zufallsmatrix_12_ein_kamm_hilft_aber_es_bleibt_eine_luecke():
    k = E.kammlauf(E.instanz_von(einstellungen("Zufallsmatrix 12: Kämme reichen nicht")))
    assert k.kaemme >= 1 and k.wert > k.wert_zm + 1 and k.wert < k.optimum.wert - 1
    assert E.verdict_kamm(k.wert_zm, k.ganz_zm, k.wert, k.ganz, k.kaemme, k.status, k.optimum.wert)[0] == "warning"
    familie, n, seed = C.OFFENES_BEISPIEL
    s = einstellungen("Zufallsmatrix 12: Kämme reichen nicht")
    assert (s.familie, s.n, s.seed) == (familie, n, seed)
    assert seed in E.tabellen()["kaemme"]["matrix-12"]["ungeloeste_seeds"]


def test_lange_wege_alle_kaemme_erfuellt_und_trotzdem_unter_dem_optimum():
    k = E.kammlauf(E.instanz_von(einstellungen("Lange Wege (12)")))
    assert k.wert == pytest.approx(40 / 3) and k.optimum.wert == 14
    art, text = E.verdict_kamm(k.wert_zm, k.ganz_zm, k.wert, k.ganz, k.kaemme, k.status, k.optimum.wert)
    assert art == "warning" and "**13,33**" in text and "**14**" in text
