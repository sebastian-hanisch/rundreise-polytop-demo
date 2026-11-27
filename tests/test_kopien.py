"""Kopien stimmen mit ihrer Quelle überein (übersprungen, wenn die Quelle nicht neben dem Repo liegt)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJEKTE = ROOT.parent


def _lies(pfad: Path) -> str:
    return pfad.read_text(encoding="utf-8").replace("\r\n", "\n")


def _quelle(*teile) -> Path:
    pfad = PROJEKTE.joinpath(*teile)
    if not pfad.exists():
        pytest.skip(f"Quelle {pfad} liegt nicht neben dem Repo")
    return pfad


def test_rng_ist_die_vorlage():
    assert _lies(ROOT / "rpt_rng.py") == _lies(_quelle("sebastianhanisch-website", "tools", "templates", "rng_splitmix64_exact.py"))


@pytest.mark.parametrize("datei", ["modul_frisch.py", "tests/test_modul_frisch.py"])
def test_modul_frisch_ist_die_kopie(datei):
    assert _lies(ROOT / datei) == _lies(_quelle("taktik-muster-demo", *datei.split("/")))


@pytest.mark.parametrize("kopie, quelle, marke", [
    ("rpt_trennung.py", ("matching-polytop-demo", "mpt_trennung.py"), "import numpy as np\n"),
    ("rpt_dd.py", ("ganzzahlige-ecken-demo", "gze_dd.py"), "from dataclasses import dataclass\n"),
])
def test_modulkopie_ab_dem_ersten_import_wie_die_quelle(kopie, quelle, marke):
    q = _lies(_quelle(*quelle))
    k = _lies(ROOT / kopie)
    assert k[k.index(marke):] == q[q.index(marke):]
    assert "Kopie von " + quelle[-1] in k          # die Herkunft steht im Modulkopf


@pytest.mark.parametrize("datei, erlaubt", [("rpt_trennung.py", {"numpy", "scipy"}), ("rpt_dd.py", {"dataclasses", "fractions", "math"})])
def test_kopierte_module_haben_keine_eigenen_abhaengigkeiten(datei, erlaubt):
    baum = ast.parse(_lies(ROOT / datei))
    namen = {a.name.split(".")[0] for k in ast.walk(baum) if isinstance(k, ast.Import) for a in k.names}
    namen |= {k.module.split(".")[0] for k in ast.walk(baum) if isinstance(k, ast.ImportFrom) and k.module}
    assert namen <= erlaubt
