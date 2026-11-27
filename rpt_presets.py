"""Permalink (Begrenzen), Presets, ausblendbare Regler (KEPT) und der Zufalls-Seed-Knopf."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import rpt_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


_STANDARD = next(iter(C.PRESETS.values()))
SETTING_SPECS = {
    "familie_select": SettingSpec("familie", _choice(C.FAMILIEN), _STANDARD["familie"]),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "k_slider": SettingSpec("k", int, _STANDARD["k"], C.K_MIN, C.K_MAX),
    "dreieck_slider": SettingSpec("dreieck", int, _STANDARD["dreieck"], C.KOSTEN_MIN, C.KOSTEN_MAX),
    "weg_slider": SettingSpec("weg", int, _STANDARD["weg"], C.KOSTEN_MIN, C.KOSTEN_MAX),
}
PRESET_KEYS = {"familie": "familie_select", "n": "n_slider", "seed": "seed_input", "k": "k_slider", "dreieck": "dreieck_slider",
               "weg": "weg_slider"}
# Regler, die je nach Familie ausgeblendet sind: Streamlit verwirft ihren Zustand, sobald sie nicht gezeichnet werden; der letzte Wert bleibt hier.
KEPT = {key: f"_kept_{key}" for key in ("n_slider", "seed_input", "k_slider", "dreieck_slider", "weg_slider")}
SICHTBAR = {
    "karte": ("n_slider", "seed_input"),
    "haufen": ("n_slider", "seed_input"),
    "matrix": ("n_slider", "seed_input"),
    "prisma": ("k_slider", "dreieck_slider", "weg_slider"),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in KEPT and state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: fehlt sein Zustand, kommt der zuletzt gewählte oder der Standardwert."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def merke(state_key):
    """Nach dem Zeichnen: den Wert für spätere Läufe ohne diesen Regler festhalten."""
    st.session_state[KEPT[state_key]] = st.session_state[state_key]


def stash_kept_widget_state():
    """Permalink und Preset legen den Wert eines ausblendbaren Reglers nur in KEPT ab (der Regler holt ihn mit `seed_widget`)."""
    for state_key, kept in KEPT.items():
        if state_key in st.session_state:
            st.session_state[kept] = st.session_state.pop(state_key)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen und auf die Reglergrenzen begrenzen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError, OverflowError):
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = value
    stash_kept_widget_state()
    st.session_state["permalink_loaded"] = True


def sync_query_params(werte):
    """`werte`: {state_key: Wert} der sichtbaren Regler; unsichtbare verschwinden aus der Adresszeile."""
    try:
        for state_key, spec in SETTING_SPECS.items():
            if state_key in werte:
                st.query_params[spec.url_param] = str(werte[state_key])
            elif spec.url_param in st.query_params:
                del st.query_params[spec.url_param]
    except Exception:
        pass


def apply_einstellungen(einstellungen):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = einstellungen[key]
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = einstellungen[key]
    stash_kept_widget_state()


def apply_preset(name):
    apply_einstellungen(C.PRESETS[name])


def _aktueller_wert(state_key):
    if state_key in st.session_state:
        return st.session_state[state_key]
    if state_key in KEPT:
        return st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)
    return SETTING_SPECS[state_key].default


def aktives_preset():
    """Name des Presets, das den aktuellen Einstellungen entspricht, sonst None. Verglichen werden die Familie und alle bei ihr sichtbaren Regler."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    familie = _aktueller_wert("familie_select")
    for name, preset in C.PRESETS.items():
        if preset["familie"] != familie:
            continue
        if all(_aktueller_wert(k) == preset[preset_key_of[k]] for k in SICHTBAR.get(familie, ())):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    neu = random.randint(0, C.SEED_MAX)
    st.session_state["seed_input"] = neu
    st.session_state[KEPT["seed_input"]] = neu
