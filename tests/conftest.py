"""Gemeinsame Fixtures: die von Hand gerechneten Prisma-Instanzen mit 6 Orten.

Ecken a, b, c = 0, 1, 2 (erstes Dreieck) und d, e, f = 3, 4, 5 (zweites); Verbinder a–d, b–e, c–f.
  H1 (Dreiecke 2, Verbinder 1): die Lösung mit ½ auf den Dreiecken und 1 auf den Verbindern erfüllt alle Subtourzeilen und kostet 9; die
      Rundreisen kosten mindestens 10. Eine 2-Matching-Ungleichung mit Griff abc und den drei Verbindern ist um 1 verletzt.
  H2 (Dreiecke 1, Verbinder 4): das Grad-LP wählt die zwei Dreiecke (Kosten 6), die Subtourzeile auf abc hebt den Wert auf 12 = Optimum.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rpt_instanzen as I  # noqa: E402

DREIECKE = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
VERBINDER = [(0, 3), (1, 4), (2, 5)]


def vektor(n, werte: dict) -> list:
    """LP-Vektor über alle Paare aus {(u, v): Wert}."""
    return [werte.get(e, 0.0) for e in I.paare(n)]


@pytest.fixture
def h1():
    return I.prisma(1, dreieck=2, weg=1)


@pytest.fixture
def h2():
    return I.prisma(1, dreieck=1, weg=4)


@pytest.fixture
def halbe_loesung():
    """½ auf beiden Dreiecken, 1 auf den Verbindern."""
    return vektor(6, {**{e: 0.5 for e in DREIECKE}, **{e: 1.0 for e in VERBINDER}})


@pytest.fixture
def zwei_dreiecke():
    return vektor(6, {e: 1.0 for e in DREIECKE})
