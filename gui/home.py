"""The first screen: start a campaign, continue one, or take the tutorial."""
import streamlit as st

from . import state, texts


def render():
    """The home screen: new campaign, continue from a workbook, tutorial (and back to a round in progress)."""
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.title('METIS')
        st.write(texts.INTRO)
        st.button('Start a new campaign', type='primary', key='b_new', on_click=state.start_new,
                  use_container_width=True)
        left, right = st.columns(2)
        left.button('Continue a campaign', key='b_continue', on_click=state.start_upload, use_container_width=True,
                    help='Upload the workbook of the last round, with the results filled in.')
        right.button('Tutorial', key='b_tutorial', on_click=state.start_tutorial, use_container_width=True,
                     help='A guided run through the app with the cell-free dataset of Borkowski et al. (2020).')
        if st.session_state.round is not None:
            st.button('Back to the round in progress', key='b_resume', on_click=state.go, args=('round',),
                      use_container_width=True)
        st.caption('Model: TabPFN-3.5. The same functions are available in Python: `from metis.campaign import '
                   'Campaign` (docs/APP.md).')
