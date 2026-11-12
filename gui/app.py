"""METIS in the browser: plan an active-learning campaign round by round.

    uv sync --extra tabpfn35 --extra cpu --extra gui        # cu128 instead of cpu with an NVIDIA RTX 50 GPU
    .venv/Scripts/python -m streamlit run gui/app.py        # Windows; .venv/bin/python elsewhere

Three ways in: start a new campaign, continue one from its workbook, or the tutorial.
Everything the app computes comes from ``metis.campaign``; this folder only draws it.
TabPFN-3.5 needs its licence accepted once or ``TABPFN_TOKEN`` set in the shell that starts
the app (docs/REPLAY.md). docs/APP.md describes the screens.
"""
import pathlib
import sys

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='METIS', layout='wide')
_loading = st.empty()
if 'page' not in st.session_state:          # the first run: the imports below take a few seconds
    _loading.info('Starting METIS: loading the scientific libraries. This takes a few seconds, once.')

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gui import home, round_page, setup_page, state  # noqa: E402
from metis.campaign import Campaign, CampaignError, VariableError, WorkbookError  # noqa: E402


SCROLL_TO_TOP = """<script>
const main = window.parent.document.querySelector('[data-testid="stMain"]')
    || window.parent.document.querySelector('section.main');
if (main) { main.scrollTo(0, 0); }
</script>"""


def take_uploads():
    """Handle files handed over by the uploaders in the previous run, before any widget is drawn."""
    ss = st.session_state
    workbook = ss.pop('inbox_workbook', None)
    if workbook is not None:
        try:
            campaign = Campaign.from_workbook(workbook)
        except (CampaignError, WorkbookError, VariableError) as error:
            ss.upload_error = str(error)
            state.go('upload')
        else:
            state.open_campaign(campaign)
    table = ss.pop('inbox_day0', None)
    if table is not None:
        state.load_day0(*table)


def show_screen():
    page = st.session_state.page
    if page == 'round' and st.session_state.round is None:
        page = 'home'
    # Two slots, always in this order: the round screen has its own, so that right after
    # "Generate" it can be drawn in the same run once the set-up screen's slot is emptied.
    # (Redrawing into one slot would leave the lower part of the longer screen standing.)
    first, second = st.empty(), st.empty()
    if page == 'round':
        with second.container():
            round_page.render()
        return
    with first.container():
        generated = {'home': home.render, 'upload': setup_page.render_upload, 'setup': setup_page.render}[page]()
    if generated:               # the set-up screen has just produced a round: show it now
        first.empty()
        with second.container():
            round_page.render()


def main():
    _loading.empty()
    state.init()
    take_uploads()
    show_screen()
    if st.session_state.scroll_top:             # after a change of screen; drawn last, so nothing shifts
        st.session_state.scroll_top = False
        components.html(SCROLL_TO_TOP, height=0)


main()
