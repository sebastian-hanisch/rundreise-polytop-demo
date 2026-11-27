"""Rundreise-Polytop: Subtour, 2-Matching, Kämme - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Das LP der symmetrischen Rundreise, Stufe für Stufe verschärft: Subtourzeilen, 2-Matching-Ungleichungen (Padberg–Rao gegen die Komponentenregel),
Kämme bis 12 Orte, die Facettenzahlen und ein Reiter zu Hilfsvariablen (Permutaeder aus einem Sortiernetz).

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import modul_frisch
import rpt_constants as C
import rpt_evaluation as E
import rpt_huelle as H
import rpt_instanzen as I
import rpt_kamm as K
import rpt_leiter as L
import rpt_lp as LP
import rpt_netz as NZ
import rpt_pdf_export as PDF
import rpt_presets as P
import rpt_referenz as R
import rpt_trennung as T
import rpt_visualization as V
import rpt_zweimatching as ZM
from rpt_layout import dunkles_design, kopf_css

modul_frisch.aktualisieren(__file__)

st.set_page_config(page_title="Rundreise-Polytop – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die Stufen und die kürzeste Rundreise (meist unter 2 Sekunden, selten bis etwa 4) …", max_entries=60)
def _analyse(settings, subtour_modus, zeitgrenze):
    beleg = max(C.ZEITGRENZE_BELEG, zeitgrenze) if zeitgrenze > C.ZEITGRENZE_OPT else C.ZEITGRENZE_BELEG
    return E.analyse(settings, zeitgrenze, subtour_modus, beleg)


@st.cache_data(show_spinner="Rechne die Trennregel (meist unter 1 Sekunde) …", max_entries=120)
def _regellauf(settings, subtour_modus, regel, modus):
    return E.regellauf(_analyse(settings, subtour_modus, C.ZEITGRENZE_OPT), regel, modus)


@st.cache_data(show_spinner="Lade Kämme nach (bis 12 Orte, meist unter 2 Sekunden) …", max_entries=40)
def _kammlauf(wahl, settings):
    return E.kammlauf(E.kamm_instanz_von(wahl, settings))


@st.cache_data(show_spinner="Berechne die ganze Hülle aller Rundreisen …")
def _huelle(n):
    return H.bericht(n)


@st.cache_data(show_spinner=False, max_entries=200)
def _netzbild(n, art, entfernt, seed):
    return E.netzbild(n, art, entfernt, seed)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rpt_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Rundreise-Polytop: Subtour, 2-Matching, Kämme")
st.markdown(
    """
Gesucht ist die **kürzeste Rundreise** durch n Orte. Das LP, das nur verlangt, dass jeder Ort genau **zwei** Nachbarn hat (**Gradgleichungen**),
liefert oft Bruchwerte oder mehrere getrennte Kreise. Hier werden die fehlenden Ungleichungen stufenweise **nachgeladen**: erst **Subtourzeilen**, dann
**2-Matching-Ungleichungen**, bis 12 Orte auch **Kämme**. In den Messreihen löst jede Stufe mindestens so viele Instanzen wie die vorige; alle werden
damit meist nicht gelöst.

Weiter unten: die Experimente 1 bis 5 (in 5 auch der Reiter **Und mit Hilfsvariablen?**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/rundreise-teilkreise-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Gesucht ist die **kürzeste Rundreise**: ein einziger Weg durch alle Orte, zurück zum Start. Ein Rechenverfahren, das nur verlangt, dass jeder Ort genau zwei Nachbarn hat, kann **getrennte kleine Runden** oder **halbe Verbindungen** liefern. Die Demo ergänzt fehlende Regeln erst dann, wenn die aktuelle Antwort sie verletzt, und rechnet neu.

Zuerst kommen Regeln gegen getrennte Runden, dann Regeln gegen bestimmte halbe Verbindungen, bei kleinen Aufgaben noch eine allgemeinere Art, die Kämme. Jede nachgeladene Regel gilt für jede echte Rundreise. Trotzdem bleiben manche Aufgaben offen: Die Antwort besteht dann nicht aus lauter ganzen Verbindungen oder ist billiger als die kürzeste Rundreise. Die Zählungen gelten nur für die erzeugten Beispielaufgaben.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/rundreise-teilkreise-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert das Nachladen", expanded=True):
    st.markdown(
        """
1. **LP lösen:** jede Verbindung zwischen zwei Orten bekommt einen Wert $x_e$ zwischen 0 und 1, an jedem Ort ist die Summe 2, die Kosten sollen
   klein sein. Der Löser (HiGHS, dualer Simplex) liefert eine **Ecke** des erlaubten Bereichs; sie kann Werte wie ½ enthalten.
2. **Trennen:** eine verletzte Ungleichung suchen. **Subtour:** aus jeder echten Ortsmenge müssen mindestens zwei Verbindungen hinausführen.
   **2-Matching:** ein **Griff** (Ortsmenge) und ungerade viele **Zahnkanten**, die ihn verlassen. **Kamm:** ein Griff und ungerade viele **Zähne**
   aus beliebig vielen Orten.
3. **Zufügen und neu lösen,** bis nichts mehr verletzt ist. Jede nachgeladene Ungleichung gilt für **jede** Rundreise; sie schneidet nie eine ab.
4. **Was dann offen bleibt:** Die **ganzzahlige Hülle** aller Rundreisen (das **Rundreise-Polytop**) hat viel mehr **Facetten** (Ungleichungen, die
   in keiner Beschreibung fehlen dürfen) als diese Familien. Die **Lücke** ist (Optimum − LP-Wert) / Optimum.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()


def _regler(state_key, zeichne):
    """Ein ausblendbarer Regler: Zustand vorbereiten, zeichnen, Wert merken."""
    P.seed_widget(state_key)
    wert = zeichne()
    P.merke(state_key)
    return wert


with st.sidebar:
    P.sidebar_presets()
    st.header("⚙️ Einstellungen")
    familie = st.selectbox("Familie", C.FAMILIEN, format_func=C.FAMILIEN_NAMEN.get, key="familie_select")
    st.caption(C.FAMILIEN_HILFE[familie])
    werte = {k: P.SETTING_SPECS[k].default for k in P.KEPT}
    if familie in C.ZUFALLSFAMILIEN:
        werte["n_slider"] = _regler("n_slider", lambda: st.slider(
            "Orte n", *P.bounds("n_slider"), key="n_slider",
            help="Bis 12 Orte werden auch Kämme nachgeladen. Bei 50 Orten rechnet die App meist unter 2 Sekunden."))
        werte["seed_input"] = _regler("seed_input", lambda: st.number_input("Zufalls-Seed", *P.bounds("seed_input"), key="seed_input", step=1))
        st.button("🎲 Neue Instanz erzeugen", width="stretch", on_click=P.randomize_seed)
    else:
        werte["k_slider"] = _regler("k_slider", lambda: st.slider(
            "Kanten je Weg", *P.bounds("k_slider"), key="k_slider", help="1 bis 4 Kanten je Weg: 6, 9, 12 oder 15 Orte."))
        werte["dreieck_slider"] = _regler("dreieck_slider", lambda: st.slider(
            "Kosten je Dreieckskante", *P.bounds("dreieck_slider"), key="dreieck_slider"))
        werte["weg_slider"] = _regler("weg_slider", lambda: st.slider(
            "Kosten je Wegkante", *P.bounds("weg_slider"), key="weg_slider", help="Bei einer Kante je Weg ist das der Verbinder."))

sichtbar = {k: werte[k] for k in P.SICHTBAR[familie]}
P.sync_query_params({"familie_select": familie, **sichtbar})

settings = E.Settings(familie, int(werte["n_slider"]), int(werte["seed_input"]), int(werte["k_slider"]), int(werte["dreieck_slider"]),
                      int(werte["weg_slider"])).bereinigt()
schluessel = "-".join(str(v) for v in (settings.familie, settings.n, settings.seed, settings.k, settings.dreieck, settings.weg))
instanz_text = E.instanz_text(settings)

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 1. Die Leiter an einer Instanz")
spalte_stufe, spalte_modus = st.columns([3, 2])
stufe = spalte_stufe.radio("Bis zu welcher Stufe nachladen?", C.STUFEN, index=2, format_func=C.STUFEN_NAMEN.get, horizontal=True,
                           key="stufe_radio")
subtour_modus = spalte_modus.radio("Subtourzeilen nachladen", C.MODI, format_func=C.MODUS_NAMEN.get, horizontal=True, key="subtourmodus_radio",
                                   help="Am Ende der Subtourstufe ist der LP-Wert bei beiden Regeln gleich; die Zahl der Zeilen kann sich unterscheiden. "
                                        "2-Matching: immer alle je Runde (Experiment 3 vergleicht die Regeln).")
laenger = st.session_state.get("opt_laenger") == schluessel
zeitgrenze = C.ZEITGRENZE_OPT_LANG if laenger else C.ZEITGRENZE_OPT
a = _analyse(settings, subtour_modus, zeitgrenze)
inst = a.inst
opt = a.optimum.wert
st.caption(f"{instanz_text}: {inst.n} Orte, {len(inst.kanten)} Verbindungen.")
for text in a.fehler:
    st.error(f"Prüfung fehlgeschlagen: {text}.")

spalten = st.columns(4)
for spalte, name in zip(spalten, C.STUFEN):
    s = a.stufen[name]
    gap = E.luecke(s.wert, opt)
    spalte.metric(C.STUFEN_NAMEN[name], E.zahl(s.wert), delta=None if gap is None else f"Lücke {E.prozent(gap)}", delta_color="off",
                  help=None if name == "grad" else f"{s.zeilen} Zeilen in {s.runden} Runden in dieser Stufe.")
spalten[3].metric("Kürzeste Rundreise", E.zahl(opt) if opt is not None else "nicht bestimmt",
                  help="Gefunden mit HiGHS-MILP (oder als LP-Lösung). Ein MILP-Status allein gilt nicht als Beleg; woran die App den Wert "
                       "glaubt, steht darunter.")
st.caption(E.referenz_text(a.optimum))
if opt is None:
    st.warning("Ohne bestimmtes Optimum zeigt die App keine Lücke.")
if not a.optimum.belegt:
    if not laenger:
        st.button(f"Länger suchen und belegen (je bis {int(C.ZEITGRENZE_OPT_LANG)} Sekunden)", key="opt_knopf",
                  on_click=lambda: st.session_state.update(opt_laenger=schluessel))

art, verdikt1 = E.verdict_stufe(a, stufe)
getattr(st, art)(verdikt1)

schritte = E.schritte_bis(a, stufe)
letzter = len(schritte) - 1
if letzter >= 1:
    schritt = st.slider("Schritt (Runde des Nachladens)", 0, letzter, letzter, key=f"schritt_{schluessel}_{stufe}_{subtour_modus}",
                        help="Schritt 0 ist das LP mit nur den Gradgleichungen.")
else:
    schritt = 0
    st.caption("Keine Runde: nach den Gradgleichungen wird hier nichts nachgeladen.")
aktuell = schritte[schritt]
menge = griff = None
zahnkanten, zahnmengen = (), ()
gewaehlt = None
if aktuell.neu:
    if len(aktuell.neu) > 1:
        nummer = st.selectbox("Welche Ungleichung dieser Runde hervorheben?", range(len(aktuell.neu)),
                              format_func=lambda i: f"{i + 1} von {len(aktuell.neu)}", key=f"zeile_{schluessel}_{stufe}_{subtour_modus}_{schritt}")
    else:
        nummer = 0
    gewaehlt = aktuell.neu[nummer]
    menge, griff, zahnkanten, zahnmengen = E.hervorhebung(inst, gewaehlt)
st.plotly_chart(V.build_karte(inst, aktuell.x, menge, griff, zahnkanten, zahnmengen), width="stretch",
                key=f"karte1_{schluessel}_{stufe}_{subtour_modus}_{schritt}")
zustand = "ganzzahlig" if LP.ist_ganzzahlig(aktuell.x) else ("gebrochen, nur Werte 0, ½ und 1" if LP.ist_halbzahlig(aktuell.x) else "gebrochen")
teile = [f"**Schritt {schritt}** ({C.STUFEN_KURZ[aktuell.stufe]}): LP-Wert {E.zahl(aktuell.wert, 4)}, Lösung {zustand}."]
if aktuell.neu:
    teile.append(f"In dieser Runde {len(aktuell.neu)} neue {'Ungleichung' if len(aktuell.neu) == 1 else 'Ungleichungen'}.")
    teile.append(E.zeile_beschreiben(inst, schritte[schritt - 1].x, gewaehlt))
st.markdown(" ".join(teile))
if inst.n > C.BESCHRIFTEN_BIS:
    st.caption("Durchgezogen: x = 1, gestrichelt: gebrochen. Den Wert zeigt die Maus an der Mitte der Verbindung; Zahlen an den Kanten stehen nur bis "
               "16 Orte.")
st.plotly_chart(V.build_stufen(schritte, opt, schritt), width="stretch", key=f"stufen_{schluessel}_{stufe}_{subtour_modus}_{schritt}")

pdf_stamm = f"{schluessel}_{subtour_modus}"
if st.button("📄 PDF-Bericht erstellen", key="pdf_knopf"):
    verdikte = [(s, E.verdict_stufe(a, s)[1]) for s in C.STUFEN]
    st.session_state["pdf"] = (pdf_stamm, PDF.erzeuge_pdf(a, instanz_text, verdikte))
if st.session_state.get("pdf", (None, None))[0] == pdf_stamm:
    st.download_button("PDF herunterladen", st.session_state["pdf"][1], file_name="rundreise-polytop.pdf", mime="application/pdf",
                       key="pdf_download")

zeile1 = E.leiter_zeile(settings.familie, settings.n) if settings.familie != "prisma" else None
if zeile1:
    g = zeile1["geloest"]
    st.info(f"**Vorgerechnet** ({E.reihen_name(zeile1)}, {zeile1['instanzen']} Instanzen): ganzzahlig und optimal "
            f"{g['grad']} / {g['subtour']} / {g['exakt']} (Grad-LP / alle Subtourzeilen / alle 2-Matching-Ungleichungen).")
elif settings.familie != "prisma":
    st.caption("Für genau diese Größe gibt es keine vorgerechnete Reihe (die Reihen stehen in Experiment 2).")

with st.expander("Wie sieht die gebrochene Lösung aus?"):
    st.markdown("Für die Lösung im oben gewählten Schritt. **Halbganzzahlig** heißt: nur 0, ½ und 1.")
    art_s, daten = LP.struktur(inst.n, inst.kanten, aktuell.x)
    art, text = E.verdict_gebrochen(art_s, daten)
    getattr(st, art)(text)
    gebrochene = [(j, w) for j, w in enumerate(aktuell.x) if LP.TOL_GANZ <= w <= 1 - LP.TOL_GANZ]
    if gebrochene:
        st.dataframe([{"Verbindung": I.kante_name(inst, j, inst.kanten), "Wert": E.bruch(w), "Kosten": inst.kosten[j]} for j, w in gebrochene[:60]],
                     width="stretch", hide_index=True)
        if len(gebrochene) > 60:
            st.caption(f"Gezeigt sind 60 von {len(gebrochene)} gebrochenen Verbindungen.")
    st.markdown("**Vorgerechnet: die Lösung nach allen Subtourzeilen** (Ecke von HiGHS; hängt bei mehreren Optima vom Löser ab)")
    st.dataframe(
        [{"Reihe": f"{E.reihen_name(z)} ({z['instanzen']})", "gebrochen": z["subtour_gebrochen"]["gebrochen"],
          "davon halbganzzahlig": z["subtour_gebrochen"]["halb"],
          "davon: ½-Kanten nur getrennte Kreise": z["subtour_gebrochen"]["halb_kreise"],
          "andere Nenner": z["subtour_gebrochen"]["anders"] + z["subtour_gebrochen"]["unbekannt"]}
         for z in E.tabellen()["leiter"].values() if z["n"] >= 30],
        width="stretch", hide_index=True)
    st.caption("Die gebrochene Lösung ist nicht durchgehend halbganzzahlig. Unter den halbganzzahligen Lösungen bilden die ½-Kanten oft, aber "
               "nicht immer, getrennte Kreise.")

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 2. Wie weit kommt jede Stufe?")
st.markdown(
    "Vorgerechnete Reihen mit festen Zufalls-Seeds. **Ganzzahlig und optimal** verlangt beides: Die LP-Lösung ist eine 0/1-Lösung und kostet so "
    "viel wie die kürzeste Rundreise. Diese kommt aus dem LP oder aus HiGHS-MILP; **jedes** dieser Optima ist beim Vorrechnen mit einem "
    "zweiten Löser (CP-SAT) geprüft, bis 13 Orte zusätzlich mit einer Dynamik über Teilmengen."
)
leiter_tab = E.tabellen()["leiter"]
passend = f"{settings.familie}-{settings.n}" if f"{settings.familie}-{settings.n}" in leiter_tab else "karte-50"
reihe = st.selectbox("Messreihe", list(leiter_tab), index=list(leiter_tab).index(passend),
                     format_func=lambda k: f"{E.reihen_name(leiter_tab[k])} ({leiter_tab[k]['instanzen']} Instanzen)", key=f"reihe_{passend}")
z2 = leiter_tab[reihe]
st.plotly_chart(V.build_stufenbalken(z2), width="stretch", key=f"balken_{reihe}")
art, text = E.verdict_stufentabelle(z2)
getattr(st, art)(text)
with st.expander("Alle Reihen, mit mittlerer und größter Lücke"):
    st.dataframe(
        [{"Reihe": E.reihen_name(z), "Instanzen": z["instanzen"], "Grad-LP": z["geloest"]["grad"], "+ Subtour": z["geloest"]["subtour"],
          "+ 2-M. Komp. (alle / eine)": f"{z['geloest']['komp_alle']} / {z['geloest']['komp_eine']}", "+ 2-M. exakt": z["geloest"]["exakt"],
          "mittlere Lücke Grad / Subtour / exakt": f"{E.prozent(z['luecke_mittel']['grad'])} / {E.prozent(z['luecke_mittel']['subtour'], 3)} / "
                                                   f"{E.prozent(z['luecke_mittel']['exakt'], 3)}",
          "größte Lücke exakt": E.prozent(z["luecke_max"]["exakt"])} for z in leiter_tab.values()],
        width="stretch", hide_index=True)
    st.markdown("**Nachgeladene Zeilen** (Median / größte je Instanz) – ohne die Auswahlregel keine Aussage:")
    st.dataframe(
        [{"Reihe": E.reihen_name(z),
          "Subtour, alle je Runde": f"{E.zahl(z['subtourzeilen']['alle_median'], 1)} / {z['subtourzeilen']['alle_max']}",
          "Subtour, eine je Runde": f"{E.zahl(z['subtourzeilen']['eine_median'], 1)} / {z['subtourzeilen']['eine_max']}",
          "2-Matching exakt, alle je Runde": f"{E.zahl(z['zweimatchingzeilen']['alle_median'], 1)} / {z['zweimatchingzeilen']['alle_max']}",
          "2-Matching exakt, eine je Runde": f"{E.zahl(z['zweimatchingzeilen']['eine_median'], 1)} / {z['zweimatchingzeilen']['eine_max']}"}
         for z in leiter_tab.values()],
        width="stretch", hide_index=True)
familien_reihen = sorted({z["familie"] for z in leiter_tab.values()}, key=C.FAMILIEN.index)
fam2 = st.selectbox("Über die Größe (Familie)", familien_reihen, index=familien_reihen.index(z2["familie"]),
                    format_func=C.FAMILIEN_NAMEN.get, key=f"groesse_{z2['familie']}")
anteile = E.anteile_ueber_n(fam2)
st.plotly_chart(V.build_groesse(anteile), width="stretch", key=f"groessenbild_{fam2}")
art, text = E.verdict_groesse(fam2, anteile)
getattr(st, art)(text)
st.caption("Die Reihen haben verschiedene Instanzzahlen (60 bis 200); „offen“ heißt offen nach diesen Familien, nicht schwer zu lösen.")

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 3. Falle: „Nichts gefunden“ heißt nicht „nichts verletzt“")
st.markdown("Nach den Subtourzeilen sucht man verletzte 2-Matching-Ungleichungen. Eine naheliegende einfache Regel hört auf, wenn **sie** nichts "
            "findet. Ob dann wirklich nichts verletzt ist, zeigt nur eine exakte Trennung.")
with st.expander("Die Trennverfahren: wer macht was?", expanded=True):
    st.dataframe([{"Name": n_, "Was": w_, "Art": a_} for n_, w_, a_ in C.VERFAHREN_TABELLE], width="stretch", hide_index=True)
    st.markdown("**Zahnzerlegung:** Padberg–Rao liefert auch Formen, bei denen zwei Zahnkanten an einem Ort liegen. Bringt man diesen Ort auf "
                "die andere Seite des Griffs und streicht beide Zahnkanten, wird die Ungleichung nie schwächer (eigene Herleitung, siehe 📐). "
                "Die App zeigt jede Zeile in dieser verkürzten Form, also als Kamm mit getrennten Zähnen.")
c_regel, c_modus = st.columns(2)
regel = c_regel.radio("Trennverfahren", C.REGELN, format_func=C.REGEL_NAMEN.get, horizontal=True, key="regel_radio")
modus3 = c_modus.radio("Auswahl", C.MODI, format_func=C.MODUS_NAMEN.get, horizontal=True, key="modus_radio")
st.caption(C.REGEL_BESCHREIBUNG[regel])
lauf = _regellauf(settings, subtour_modus, regel, modus3)
exakt = a.stufen["zm"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("LP-Wert am Ende der Regel", E.zahl(lauf.wert, 4))
m2.metric("LP-Wert mit exakter Trennung", E.zahl(exakt.wert, 4))
m3.metric("2-Matching-Zeilen (Regel)", lauf.zeilen, help=f"In {lauf.runden} Runden.")
m4.metric("Danach noch verletzt?", "ja" if lauf.rest is not None else "nein")
art, text = E.verdict_falle(regel, None if lauf.rest is None else lauf.rest.wert, lauf.wert, exakt.wert, opt, lauf.status, lauf.zeilen)
getattr(st, art)(text)
if lauf.rest is not None:
    _, griff3, zahn3, _ = E.hervorhebung(inst, lauf.rest)
    st.plotly_chart(V.build_karte(inst, lauf.x, griff=griff3, zahnkanten=zahn3), width="stretch", key=f"karte3_{schluessel}_{regel}_{modus3}")
    st.markdown("Die Ungleichung, die die Regel übersehen hat (gefunden von Padberg–Rao): " + E.zeile_beschreiben(inst, lauf.x, lauf.rest))

st.markdown("**Vorgerechnet: Wie oft passiert das?** (Instanzen, auf denen die Komponentenregel endet, obwohl noch etwas verletzt ist)")
art, text = E.verdict_komponenten(list(leiter_tab.values()))
getattr(st, art)(text)
st.dataframe(
    [{"Reihe": f"{E.reihen_name(z)} ({z['instanzen']})",
      "übersehen, alle je Runde": z["komponentenregel"]["alle"]["uebersehen"],
      "übersehen, eine je Runde": z["komponentenregel"]["eine"]["uebersehen"],
      "anderer Endwert je Auswahl": z["komponentenregel"]["endwert_verschieden"],
      "größter Abstand zu exakt (eine je Runde)": E.prozent(z["komponentenregel"]["eine"]["abstand_max"]),
      "größter Abstand (alle je Runde)": E.prozent(z["komponentenregel"]["alle"]["abstand_max"])}
     for z in leiter_tab.values()],
    width="stretch", hide_index=True)
st.caption("Bei der exakten Trennung hängt der Endwert nicht an der Auswahl (auf jeder Instanz der Reihen geprüft).")

with st.expander("So trennt Padberg–Rao hier: der geteilte Hilfsgraph"):
    if inst.n > C.HILFSGRAPH_MAX_N:
        st.info(f"Gezeichnet bis {C.HILFSGRAPH_MAX_N} Orte. Wählen Sie links zum Beispiel ein Prisma.")
    else:
        x_sub = a.stufen["subtour"].x
        knoten, graph, markiert, teil = ZM.hilfsgraph(inst.n, inst.kanten, x_sub)
        schnitte = T.ungerade_schnitte(knoten, graph, markiert, grenze=1.0)
        seite = schnitte[0][1] if schnitte else None
        st.plotly_chart(V.build_hilfsgraph(inst, inst.kanten, teil, x_sub, seite), width="stretch", key=f"hilfsgraph_{schluessel}")
        st.markdown(
            "Gezeigt an der Lösung nach den Subtourzeilen: jede benutzte Verbindung u–v wird in u–w (Kapazität x) und w–v (Kapazität 1 − x) geteilt. "
            "Markiert sind alle w und die Orte, an denen ungerade viele (1 − x)-Hälften hängen; Hälften mit Kapazität 1 werden vor dem Baum "
            "zusammengezogen. Gesucht ist ein Schnitt unter 1 mit ungerade vielen markierten Knoten."
        )
        if seite is None:
            st.success("Kein ungerader Schnitt liegt unter 1: keine 2-Matching-Ungleichung ist verletzt.")
        else:
            wert_schnitt = schnitte[0][0]
            st.markdown(f"Der kleinste ungerade Baumschnitt hat den Wert **{E.bruch(wert_schnitt)}** < 1. Seine Orte bilden den Griff; die "
                        "Verbindungen, deren (1 − x)-Hälfte er trennt, sind die Zahnkanten (rot hervorgehoben sind alle getrennten Hälften).")

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 4. Wo 2-Matching endet: Kämme")
st.markdown(
    "Ein **Kamm** hat einen Griff H und ungerade viele (mindestens 3) getrennte **Zähne**, jeder mit Orten in H und außerhalb: "
    "$x(\\delta(H)) + \\sum_j x(\\delta(T_j)) \\ge 3p + 1$. 2-Matching-Ungleichungen sind Kämme mit Zähnen aus je zwei Orten. Die App sucht Kämme "
    f"**vollständig** (alle Griffe und alle Zahnfamilien, die verletzt sein können), deshalb nur bis {K.MAX_N} Orte."
)
wahl = st.selectbox("Beispiel", C.KAMM_BEISPIELE, format_func=C.KAMM_BEISPIEL_NAMEN.get, key="kamm_beispiel")
kamm_inst = E.kamm_instanz_von(wahl, settings)
kamm_schluessel = f"{wahl}_{schluessel}" if wahl == "aktuell" else wahl
zu_gross = kamm_inst.n > K.MAX_N
st.button("Kämme nachladen", key="kamm_knopf", disabled=zu_gross, on_click=lambda: st.session_state.update(kamm_wunsch=kamm_schluessel))
if zu_gross:
    st.caption(f"Gesperrt: die Instanz links hat {kamm_inst.n} Orte, die vollständige Kammsuche prüft alle Griffe und alle getrennten Zahnfamilien "
               f"und ist deshalb auf {K.MAX_N} Orte begrenzt. Wählen Sie oben ein anderes Beispiel.")
elif st.session_state.get("kamm_wunsch") == kamm_schluessel:
    kl = _kammlauf(wahl, settings)
    k_opt = kl.optimum.wert
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LP nach allen 2-Matching-Ungleichungen", E.zahl(kl.wert_zm))
    c2.metric("LP nach allen Kämmen", E.zahl(kl.wert))
    c3.metric("Kürzeste Rundreise", E.zahl(k_opt) if k_opt is not None else "nicht bestimmt")
    c4.metric("Nachgeladene Kämme", kl.kaemme)
    st.caption(E.referenz_text(kl.optimum))
    art, text = E.verdict_kamm(kl.wert_zm, kl.ganz_zm, kl.wert, kl.ganz, kl.kaemme, kl.status, k_opt)
    getattr(st, art)(text)
    letzter_k = len(kl.schritte) - 1
    if letzter_k >= 1:
        s4 = st.slider("Schritt der Kammstufe", 0, letzter_k, letzter_k, key=f"kammschritt_{kamm_schluessel}",
                       help="Schritt 0 ist die Lösung nach allen 2-Matching-Ungleichungen. Je Runde kommt höchstens ein Kamm dazu; davor werden "
                            "wieder alle verletzten Subtourzeilen und 2-Matching-Ungleichungen (diese eine je Runde) geladen.")
    else:
        s4 = 0
    schritt4 = kl.schritte[s4]
    eintrag = schritt4.neu[0] if schritt4.neu else None
    menge4 = griff4 = None
    zk4, zm4 = (), ()
    if eintrag is not None:
        menge4, griff4, zk4, zm4 = E.hervorhebung(kl.inst, eintrag)
    st.plotly_chart(V.build_karte(kl.inst, schritt4.x, menge4, griff4, zk4, zm4), width="stretch", key=f"karte4_{kamm_schluessel}_{s4}")
    if eintrag is not None:
        st.markdown(f"LP-Wert {E.zahl(schritt4.wert, 4)}. " + E.zeile_beschreiben(kl.inst, kl.schritte[s4 - 1].x, eintrag))
    else:
        st.markdown(f"Start: Lösung nach allen 2-Matching-Ungleichungen, LP-Wert {E.zahl(schritt4.wert, 4)}.")
else:
    st.caption("Der Knopf rechnet die Kämme für das gewählte Beispiel.")

st.markdown("**Vorgerechnet: Reichen alle Kämme?** (je 200 Instanzen; offen = nach allen 2-Matching-Ungleichungen nicht ganzzahlig und optimal)")
kamm_tab = E.tabellen()["kaemme"]
st.dataframe(
    [{"Reihe": E.reihen_name(z), "offen nach 2-Matching": z["offen"], "mit allen Kämmen gelöst": z["mit_kaemmen_geloest"],
      "mittlere Lücke der offenen vorher → nachher": "–" if not z["offen"] else
      f"{E.prozent(z['luecke_vorher_mittel'])} → {E.prozent(max(z['luecke_nachher_mittel'], 0.0))}",
      "größte Lücke danach": "–" if not z["offen"] else E.prozent(max(z["luecke_nachher_max"], 0.0))} for z in kamm_tab.values()],
    width="stretch", hide_index=True)
prisma_tab = E.tabellen()["prisma"]
with st.expander("Prisma-Familie: wie groß wird der Abstand?"):
    st.dataframe(
        [{"Kanten je Weg k": z["k"], "Orte": z["orte"], "Subtour-LP": E.zahl(z["subtour"]), "+ 2-Matching": E.zahl(z["zweimatching"]),
          "+ alle Kämme": E.zahl(z["kaemme"]) if z["kaemme"] is not None else "nicht gerechnet", "Optimum": z["optimum"],
          "Optimum / Subtour-LP": E.zahl(z["optimum"] / z["subtour"], 3)} for z in prisma_tab["wege"]],
        width="stretch", hide_index=True)
    st.markdown("Für k = 1 bis 4 ist nachgerechnet: Optimum 4k + 2 bei LP-Wert 3k + 3 nach allen Subtourzeilen. Setzt sich das fort, geht der "
                "Quotient gegen 4/3; **für alle k ist das hier nicht bewiesen**. Dass der Quotient aus Optimum und Subtour-LP bei metrischen Kosten "
                "nie über 4/3 liegt, ist eine offene Vermutung; bewiesen ist eine obere Schranke etwas unter 3/2.")
    st.markdown("**Sechs Orte, andere Kosten** (Dreieckskanten c, Verbinder d):")
    st.dataframe(
        [{"c": z["dreieck"], "d": z["verbinder"], "Grad-LP": E.zahl(z["grad"]), "+ Subtour": E.zahl(z["subtour"]),
          "+ 2-Matching": E.zahl(z["zweimatching"]), "Optimum": z["optimum"]} for z in prisma_tab["kosten"]],
        width="stretch", hide_index=True)

st.markdown("---")

# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 5. Warum nicht einfach alle Ungleichungen?")
reiter_facetten, reiter_hilfe = st.tabs(["Wie viele Facetten gibt es?", "Und mit Hilfsvariablen?"])

with reiter_facetten:
    huelle_tab = E.tabellen()["huelle"]
    veroeffentlicht = {8: 194187, 9: 42104442}
    zeilen = []
    for n_ in (5, 6, 7, 8, 9):
        if n_ <= H.LIVE_MAX_N:
            facetten, herkunft = _huelle(n_)["facetten"], "hier live berechnet"
        elif n_ == 7:
            facetten, herkunft = huelle_tab["7"]["facetten"], "hier vorgerechnet (ganze Zahlen)"
        else:
            facetten, herkunft = veroeffentlicht[n_], "veröffentlicht (SMAPO), nicht nachgerechnet"
        zeilen.append({"Orte n": n_, "Dimension n(n−3)/2": n_ * (n_ - 3) // 2, "Rundreisen (n−1)!/2": math.factorial(n_ - 1) // 2,
                       "Facetten": E.ganzzahl(facetten), "davon Subtour (2^(n−1) − n − 1)": H.subtourfacetten(n_), "Herkunft": herkunft})
    st.dataframe(zeilen, width="stretch", hide_index=True)
    st.caption("Facetten des Polytops der **symmetrischen** Rundreise auf dem vollständigen Graphen, gezählt modulo der Gradgleichungen (dieselbe "
               "Facette hat viele Schreibweisen). Die Zahlen für 6 bis 9 Orte stehen auch bei SMAPO (Universität Heidelberg). Die Spalte Subtour "
               "zählt x(δ(S)) ≥ 2 mit mindestens zwei Orten auf jeder Seite, also die Schranke x ≤ 1 mit.")
    n_h = st.radio("Familien der Facetten für n =", (5, 6, 7), index=1, horizontal=True, key="huelle_n")
    bericht = _huelle(n_h) if n_h <= H.LIVE_MAX_N else huelle_tab["7"]
    st.dataframe([{"Familie": name, "Facetten": E.ganzzahl(zahl_)} for name, zahl_ in bericht["familien"].items()], width="stretch",
                 hide_index=True)
    if n_h == 7:
        st.caption("Bei 7 Orten bleiben 1 260 Facetten, die keine Kämme mit 3 Zähnen sind; welcher Familie sie angehören, wird hier nicht "
                   "bestimmt. Die Zeile „vorgerechnet“ kommt aus tools/rpt_make_tables.py (Rechenzeit etwa zwei Minuten).")
    else:
        st.caption("Zuordnung über die Mengen der straffen Rundreisen. Bei 5 Orten bestehen die Facetten nur aus den Schranken, bei 6 Orten kommen "
                   "die 3er-Subtourzeilen und 2-Matching-Ungleichungen dazu.")
    art, text = E.verdict_facetten(leiter_tab["karte-50"])
    getattr(st, art)(text)
    st.markdown(f"**Hier:** {inst.n} Orte, {E.ganzzahl(H.subtourfacetten(inst.n))} Subtourfacetten; nachgeladen: "
                f"{E.anzahl_wort(a.stufen['subtour'].zeilen, 'Subtourzeile', 'Subtourzeilen')} in der Subtourstufe "
                f"({C.MODUS_NAMEN[subtour_modus]}), {E.anzahl_wort(a.stufen['zm'].zeilen, '2-Matching-Ungleichung', '2-Matching-Ungleichungen')} "
                "in der nächsten.")
    st.markdown(
        "**Trennen statt Aufschreiben.** Nach Grötschel, Lovász und Schrijver (hier nach einem Sekundärzitat) gilt für ein Polyeder, bei dem es "
        "eine rationale Beschreibung mit höchstens φ Bits je Ungleichung gibt: Aus einem Orakel für das Trennen lässt sich das Optimieren in "
        "Orakel-Polynomialzeit gewinnen und umgekehrt (polynomiell in der Dimension, in φ und in der Länge der Eingabe; Werkzeug ist die "
        "Ellipsoidmethode, kein praktisches Verfahren). Für Subtour- und 2-Matching-Ungleichungen sind polynomielle exakte Trennungen bekannt. Die "
        "Komplexität der Trennung allgemeiner Kämme galt lange als offen; ein **Vorabdruck vom Oktober 2026, nicht geprüft**, behauptet "
        "NP-Schwere."
    )

with reiter_hilfe:
    st.markdown(
        "Viele Facetten heißen nicht, dass jede Beschreibung groß sein muss. Eine **erweiterte Formulierung** beschreibt ein Polytop als "
        "**Projektion** eines Polyeders mit **Hilfsvariablen**. Beim **Permutaeder** (konvexe Hülle aller Anordnungen von 1 bis n, 2ⁿ − 2 Facetten) "
        "geht das mit einem **Sortiernetz** (Idee nach Goemans): je Vergleich zweier Werte a, b neue Werte lo, hi mit a + b = lo + hi, "
        "lo ≤ a und lo ≤ b, am Ende stehen 1, …, n."
    )
    r1, r2, r3, r4 = st.columns([2, 2, 2, 2])
    n_netz = r1.select_slider("Werte n", options=list(range(3, C.NETZ_N_MAX + 1)), value=4, key="netz_n")
    art_netz = r2.radio("Netz", NZ.NETZE, format_func=C.NETZ_NAMEN.get, key="netz_art")
    anzahl_komp = len(NZ.netz(art_netz, n_netz))
    weg = r3.selectbox("Gestrichener Komparator", range(-1, anzahl_komp), format_func=lambda i: "keiner" if i < 0 else f"Nr. {i + 1}",
                       key=f"netz_weg_{art_netz}_{n_netz}")
    netz_seed = r4.number_input("Zufalls-Seed der Zielkoeffizienten", 0, C.SEED_MAX, 1, step=1, key="netz_seed")
    st.caption(C.NETZ_BESCHREIBUNG[art_netz])
    nb = _netzbild(int(n_netz), art_netz, None if weg < 0 else int(weg), int(netz_seed))
    st.plotly_chart(V.build_netz(nb.n, nb.komparatoren, nb.entfernt), width="stretch", key=f"netzbild_{nb.n}_{nb.art}_{nb.entfernt}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ungleichungen (mit Hilfsvariablen)", nb.ungleichungen, help=f"{nb.variablen} Variablen, {nb.gleichungen} Gleichungen.")
    k2.metric("Facetten (ohne Hilfsvariablen)", E.ganzzahl(nb.facetten))
    k3.metric("LP-Minimum über der Projektion", E.zahl(nb.lp_wert))
    k4.metric("Minimum über das Permutaeder (Sortieren)", E.zahl(nb.sort_wert))
    art, text = E.verdict_netz(nb.n, nb.ungleichungen, nb.variablen, nb.gleichungen, nb.facetten, nb.entfernt is not None, nb.sortiert,
                               None if nb.drin is None else len(nb.drin), None if nb.drin is None else len(nb.drin) + len(nb.draussen),
                               nb.lp_wert, nb.sort_wert)
    getattr(st, art)(text)
    st.caption(f"Zielkoeffizienten: {', '.join(str(v) for v in nb.ziel)} (Minimum von Σ cᵢ·zᵢ).")
    if nb.drin is not None:
        if nb.n == 4:
            st.plotly_chart(V.build_permutaeder(nb.drin), width="stretch", key=f"perm_{nb.art}_{nb.entfernt}")
            st.caption("Permutaeder für n = 4 (in 3D, Summe 10): jede Ecke ist eine Anordnung von 1 bis 4.")
        if nb.draussen:
            st.markdown("**Eingaben, die das Netz nicht mehr sortiert** (und die deshalb nicht in der Projektion liegen): "
                        + ", ".join("".join(str(v) for v in w) for w in nb.draussen[:40]) + (" …" if len(nb.draussen) > 40 else ""))
    st.markdown(
        "**Warum es nie zu groß wird** (eigene Herleitung, nicht extern geprüft; Beweis im 📐-Abschnitt): Für jedes Netz liegt die Projektion im "
        "Permutaeder, und eine Anordnung liegt genau dann darin, wenn das Netz sie sortiert. Ein Netz, das nicht sortiert, gibt also ein zu "
        "kleines Polytop. Welche Anordnungen beim Streichen verloren gehen, hängt vom Netz ab."
    )
    st.warning(
        "**Falle: „Viele Facetten heißt schwere Aufgabe.“** Das Permutaeder hat 2ⁿ − 2 Facetten, und Sortieren löst jede Zielfunktion. Ob "
        "umgekehrt auch Hilfsvariablen nicht helfen, lässt sich an keiner kleinen Instanz messen; zwei Sätze werden hier nur zitiert (nur die "
        "Abstracts gelesen): Rothvoß: jede erweiterte Formulierung des **perfekten Matching-Polytops** des vollständigen Graphen K_n braucht "
        "$2^{\\Omega(n)}$ Ungleichungen, obwohl gewichtetes perfektes Matching polynomiell lösbar ist (Edmonds). Fiorini, Massar, Pokutta, Tiwary "
        "und de Wolf: kein LP polynomieller Größe projiziert auf das **Rundreise-Polytop**, auch ohne Symmetrieannahme."
    )
    sp = [NZ.martin_groessen(n_) for n_ in (4, 10)]
    st.markdown(
        f"**Spannbäume.** Das Spannbaumpolytop des vollständigen Graphen K_n hat ab 4 Knoten m + 2ⁿ − n − 2 Facetten (m Verbindungen). Martin (1991, hier nach "
        f"einer Übersicht, das Original nicht gelesen) beschreibt es mit Hilfsvariablen je Wurzel. Klein ist das erst mit der Größe: bei n = 4 "
        f"{sp[0]['ungleichungen']} Ungleichungen statt {NZ.spannbaum_facetten(4)} Facetten, erst ab n = 10 weniger "
        f"({sp[1]['ungleichungen']} statt {E.ganzzahl(NZ.spannbaum_facetten(10))}, dafür {sp[1]['variablen']} statt 45 Variablen und "
        f"{sp[1]['gleichungen']} Gleichungen). Dasselbe Polytop ohne Hilfsvariablen, mit Rangungleichungen und Trennen, zeigt die "
        "[Matroid-Polytop-Demo](https://github.com/sebastian-hanisch/matroid-polytop-demo) an dünnen Graphen."
    )

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie fehlt | Wer setzt an |
|---|---|---|
| **Subtour und 2-Matching genügen** | Nicht allgemein: In den gemessenen Kartenreihen bleibt bei 30 und 50 Orten deutlich mehr offen (Experiment 2), und auch alle Kämme lösen nicht jede Instanz (Experiment 4). Das Rundreise-Polytop hat viele weitere Facettenfamilien. | – |
| **Die Kosten sind symmetrisch** | Bei gerichteten Kosten ist das Polytop ein anderes. Die Zuordnungs-Relaxierung dafür zeigt [Zuordnung gegen Rundreise](https://github.com/sebastian-hanisch/zuordnung-vs-rundreise-demo); ihre Lücken sind nicht mit denen hier vergleichbar. | – |
| **Eine Familie genügt** | Beim Matching ist das so ([Matching-Polytop](https://github.com/sebastian-hanisch/matching-polytop-demo)); bei der Rundreise nicht. Der Kern der Trennung für 2-Matching ist derselbe wie dort; der Hilfsgraph hat hier geteilte Kanten statt eines Zusatzknotens. | – |
| **Schnitte genügen zum Lösen** | Ein Branch-and-Cut-Verfahren, das in den Knoten auch 2-Matching- und Kammungleichungen trennt, ist hier nicht Thema; die kürzeste Rundreise findet HiGHS-MILP, und das eigene Branch-and-Bound (nur Subtourzeilen) dient allein als ihr Beleg. | – |
| **Jede Ungleichung kommt aus einer bekannten Familie** | Man kann sie auch aus den vorhandenen Zeilen durch Addieren und Abrunden erzeugen: [Chvátal-Gomory](https://github.com/sebastian-hanisch/chvatal-gomory-demo). | – |
| **Selbst gewählte Zufallsfamilien** | Alle Häufigkeiten gelten für Karten, Haufen und Zufallsmatrizen aus diesem Generator. Zahlen nachgeladener Zeilen und gebrochene Ecken hängen an der Auswahlregel, am Löser und am System. | – |
| **Güte von Heuristiken** | Die hier gemessenen LP-Lücken messen keine Tourenheuristik wie [Lin-Kernighan](https://github.com/sebastian-hanisch/lin-kernighan-demo). | – |

Verwandt: [Gomory-Hu-Baum](https://github.com/sebastian-hanisch/gomory-hu-demo) (das Werkzeug der Trennung),
[Zusammenhang verstärken](https://github.com/sebastian-hanisch/zusammenhang-verstaerken-demo) (dieselben Schnittbedingungen mit anderem Trennverfahren,
Gradschranke statt Gradgleichung), [Ganzzahlige Ecken](https://github.com/sebastian-hanisch/ganzzahlige-ecken-demo) (Ecken und Facetten an kleinen Graphen),
[Matroid-Polytop](https://github.com/sebastian-hanisch/matroid-polytop-demo) (dort genügt eine vollständig bekannte Beschreibung).
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/rundreise-teilkreise-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Orte und ganzzahlige Kosten $c_e$ je Verbindung, in beide Richtungen gleich. Gesucht ist die kürzeste Rundreise; das lineare Programm darf jeder Verbindung einen Anteil $x_e$ zwischen 0 und 1 geben, und fehlende Ungleichungen werden nachgeladen.

| Symbol | Bedeutung |
|---|---|
| $x_e$ | Anteil der Verbindung $e$ (1 = benutzt, 0 = nicht benutzt) |
| $c_e$ | Kosten der Verbindung $e$ |
| $n$ | Zahl der Orte |
| $v$ | ein einzelner Ort |
| $S$ | Menge von Orten, mit mindestens zwei Orten innen und außen |
| $\delta(S)$, $\delta(v)$ | Verbindungen, die die Menge $S$ verlassen, bzw. die Verbindungen am Ort $v$ |
| $x(F)$ | Summe der Anteile $x_e$ über die Verbindungsmenge $F$ |
| $H$, $F$ | Griff (Ortsmenge) und ungerade viele Zahnkanten, die ihn verlassen |
| $T_j$, $p$ | Zahn $j$ eines Kamms (Ortsmenge) und die Zahl der Zähne |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/rundreise-teilkreise-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Grad-LP.**

$$\min \sum_{e} c_e x_e$$

$$x(\delta(v)) = 2 \quad \text{für jeden Ort } v$$

$$0 \le x_e \le 1$$

**Subtourzeilen.** Für jede Menge $S$ mit mindestens zwei Orten auf beiden Seiten:

$$x(\delta(S)) \ge 2$$

Trennung: erst die Zusammenhangskomponenten der benutzten Verbindungen, sonst die Schnitte eines Gomory-Hu-Baums unter 2 (die kleinste Baumkante ist
der globale Mindestschnitt).

**2-Matching-Ungleichung** (Blütenform) mit Griff $H$ und ungerader Menge $F \subseteq \delta(H)$:

$$x(\delta(H) \setminus F) + \sum_{e \in F} (1 - x_e) \ge 1$$

**Trennen (Padberg und Rao).** Jede Trägerkante $uv$ wird in $u - w$ (Kapazität $x_{uv}$) und $w - v$ (Kapazität $1 - x_{uv}$) geteilt. Markiert
sind alle $w$ und die Orte mit ungerade vielen $(1-x)$-Hälften; ein ungerader Schnitt unter 1 gibt $H$ und $F$. Ein kleinster ungerader Schnitt
liegt unter den Schnitten eines Gomory-Hu-Baums über alle Knoten (dass der Baum über alle Knoten genügt: eigene Herleitung, im Kopf von
`rpt_trennung.py`).

**Zahnzerlegung** (eigene Herleitung, nicht extern geprüft): Liegen zwei Zahnkanten an einem Ort $v$ und keine dritte, sei $H' = H \triangle \{v\}$
und $F'$ die Menge $F$ ohne beide. Mit $x(\delta(v)) = 2$ gilt, wobei $R$ die übrigen Verbindungen an $v$ sind, die $H$ verlassen:

$$x(\delta(H') \setminus F') + \sum_{F'} (1 - x_e) = x(\delta(H) \setminus F) + \sum_{F} (1 - x_e) - 2\, x(R)$$

Die verkürzte Ungleichung ist also mindestens so stark. Drei Zahnkanten an einem Ort sind nie verletzt, denn ihre Beiträge $(1-x_e)$ sind zusammen
mindestens $3 - 2 = 1$; auch $|F| = 1$ ist unter den Subtourzeilen nie verletzt, denn der Wert ist $x(\delta(H)) + 1 - 2x_e \ge 1$.

**Kamm** (Schnittform): Griff $H$, $p \ge 3$ ungerade, Zähne $T_1, \dots, T_p$ paarweise disjunkt, jeder mit Orten in $H$ und außerhalb:

$$x(\delta(H)) + \sum_{j=1}^{p} x(\delta(T_j)) \ge 3p + 1$$

Für einen Zahn aus den Enden einer Verbindung $e$ gilt mit den Gradgleichungen $x(\delta(T)) = 4 - 2x_e$; Zähne aus je zwei Orten ergeben die
2-Matching-Ungleichung.

**Vollständige Kammsuche bis 12 Orte.** Unter den Subtourzeilen ist $x(\delta(T_j)) \ge 2$; ein Kamm ist genau dann verletzt, wenn

$$\sum_{j=1}^{p} \left( x(\delta(T_j)) - 2 \right) < p + 1 - x(\delta(H))$$

Durchsucht werden alle Griffe und alle disjunkten Zahnfamilien innerhalb dieser Schranke ($p = 3$ oder $5$).

**Lücke.** $(\text{Optimum} - \text{LP-Wert}) / \text{Optimum}$, in der Anzeige in $\%$ des Optimums.

**Permutaeder (Rado).** $\sum_i z_i = n(n+1)/2$ und für jede echte nichtleere Teilmenge $S$

$$\sum_{i \in S} z_i \ge \frac{|S| (|S| + 1)}{2}$$

**Aus einem Komparatornetz.** Je Komparator mit Eingängen $a, b$:

$$lo + hi = a + b$$

$$lo \le a, \quad lo \le b$$

Die Eingänge des Netzes sind $z$, die Ausgänge am Ende sind $1, \dots, n$.

**Satz und Beweis** (eigene Herleitung, nicht extern geprüft): Für jedes Netz liegt die Projektion im Permutaeder, und eine Permutation $w$ liegt
genau dann darin, wenn das Netz $w$ sortiert. Aus $lo \le a, b$ und $lo + hi = a + b$ folgt $hi \ge \max(a, b)$: $(a, b)$ wird von $(lo, hi)$
majorisiert, also $z$ von $(1, \dots, n)$; nach Rado liegt $z$ im Permutaeder. Sortiert das Netz $w$, ist der Durchlauf eine Belegung. Liegt
umgekehrt eine Permutation $w$ in der Projektion: Wegen $lo \le \min(a, b)$ und $lo + hi = a + b$ kann $\sum (\text{Drahtwert})^2$ an keinem
Komparator fallen. Am Anfang ($w$) und am Ende ($1, \dots, n$) ist die Summe gleich, also wächst sie auch an keinem; damit ist jedes Mal
$\{lo, hi\} = \{a, b\}$ mit $lo = \min$, und der Durchlauf endet sortiert.

**Rechenweg der Demo.** HiGHS, dualer Simplex; verletzt heißt Verletzung über $10^{-6}$, ganzzahlig heißt jeder Wert bis auf $10^{-5}$ gleich 0
oder 1. Die ganze Hülle bei 5 bis 7 Orten rechnet eine doppelte Beschreibung in ganzen Zahlen.

**Woran die kürzeste Rundreise geglaubt wird.** HiGHS-MILP (mit nachgeladenen Subtourzeilen und Zeitgrenze) findet sie; ein MILP-Status allein ist
kein Beleg. Belegt ist der Wert $L$, wenn eine selbst nachgerechnete Dualschranke über $L - 1$ liegt (die Kosten sind ganzzahlig). Mit beliebigen
Werten $\pi$ der Gradgleichungen und $y \ge 0$ der Zeilen $Gx \ge g$ gilt für jedes zulässige $0 \le x \le 1$:

$$c^\top x \ge 2 \sum_v \pi_v + g^\top y + \sum_e \min(0, r_e)$$

$$r = c - A^\top \pi - G^\top y$$

Sonst prüft bis 13 Orte eine Dynamik über Teilmengen, darüber ein eigenes Branch-and-Bound mit derselben Schranke und Zeitgrenze. Gelingt beides
nicht, zeigt die App „nur laut HiGHS-MILP“ (nach einer Plausibilitätsprüfung mit 2-opt und Or-opt) oder „nicht bestimmt“.

**Literatur** (Sätze nach Skripten, Aufsätzen und Datenseiten, soweit gelesen; viele Originale sind nicht gelesen): Dantzig, Fulkerson und Johnson
(Subtourzeilen), nach A. Letchford, Math. Oper. Res. 25 (2000), dort auch die Kammform nach Chvátal sowie Grötschel und Padberg; Padberg und Rao
(1982), hier nach Letchford, Reinelt und Theis, SIAM J. Discrete Math. 22 (2008); Facettenzahlen nach SMAPO (Universität Heidelberg); Rado und das
Permutaeder, Goemans (Sortiernetze, nur Abstract), Rothvoß (nur Abstract), Fiorini u. a. (nur Abstract); Martin (1991), nach einer Übersicht von
Conforti, Cornuéjols und Zambelli.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Fehlende Ungleichungen: von Ecken bis Runden](https://sebastianhanisch.net/konzepte-polyeder.html)."
)
