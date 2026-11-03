"""METIS in the browser: upload data, explore it, and get the next batch of experiments.

    uv run --extra gui --extra tabpfn35 --extra cpu streamlit run gui/app.py

TabPFN-3.5 needs its licence accepted once or ``TABPFN_TOKEN`` set in the shell that
starts the app (see docs/REPLAY.md); the other models need nothing extra.
"""
import io
import pathlib
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gui import core  # noqa: E402
from metis_bench.report import INK, INK_2, PALETTE, SURFACE  # noqa: E402

GREY = '#8a8984'
EXAMPLES = {'Example: Borkowski (cell-free, 11 factors)': 'borkowski', 'Example: PURE (69 factors)': 'pure',
            'Example: CETCH yield': 'cetch_yield'}

st.set_page_config(page_title='METIS', layout='wide')


@st.cache_data(show_spinner=False)
def load_upload(data, name):
    return core.read_table(io.BytesIO(data), name)


@st.cache_data(show_spinner=False)
def load_example(name):
    from metis_bench.oracles.measured import load_dataset
    rows, factors = load_dataset(name)
    return rows[factors + ['yield']]


@st.cache_resource(show_spinner=False)
def design_for(df, target, numeric, categorical):
    return core.build_design(df, target, list(numeric), list(categorical))


def figure(w=6.0, h=4.2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    return plt, fig, ax


def bar_chart(series, xlabel):
    s = series.sort_values().tail(20)
    plt, fig, ax = figure(6, 0.28 * len(s) + 1.0)
    ax.barh(s.index.astype(str), s.to_numpy(), color=PALETTE[0], height=0.65)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- data
st.sidebar.header('1. Data')
choice = st.sidebar.radio('Source', ['Upload a file'] + list(EXAMPLES))
if choice == 'Upload a file':
    upload = st.sidebar.file_uploader('CSV or Excel: one row per recipe, one column per factor, one result column',
                                      type=['csv', 'xlsx', 'xls'])
    if upload is None:
        st.title('METIS: choose the next experiments')
        st.info('Upload a table of recipes, or pick an example dataset in the sidebar. Rows without a result '
                'are treated as candidates you could run next.')
        st.stop()
    df = load_upload(upload.getvalue(), upload.name)
else:
    df = load_example(EXAMPLES[choice])

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
default_target = 'yield' if 'yield' in df.columns else (numeric_cols[-1] if numeric_cols else df.columns[-1])
target = st.sidebar.selectbox('Result to maximise', list(df.columns), index=list(df.columns).index(default_target))
guess_num, guess_cat = core.guess_columns(df, target)
ignored = st.sidebar.multiselect('Columns to ignore (IDs, notes, replicate SDs)',
                                 [c for c in df.columns if c != target],
                                 default=[c for c in df.columns if c not in guess_num + guess_cat + [target]])
numeric = st.sidebar.multiselect('Numeric factors', [c for c in df.columns if c not in ignored + [target]],
                                 default=[c for c in guess_num if c not in ignored])
categorical = st.sidebar.multiselect('Categorical factors (e.g. strain, promoter)',
                                     [c for c in df.columns if c not in ignored + [target] + numeric],
                                     default=[c for c in guess_cat if c not in ignored + numeric])
if not numeric and not categorical:
    st.warning('Pick at least one factor column.')
    st.stop()
try:
    design = design_for(df, target, tuple(numeric), tuple(categorical))
except ValueError as err:
    st.error(str(err))
    st.stop()

st.title('METIS: choose the next experiments')
n_meas = int(design.measured.sum())
st.caption('{} recipes, {} with a result, {} factors ({} numeric, {} categorical). Design space: {:.3g} '
           'combinations of the observed levels.'.format(len(design.frame), n_meas, len(numeric) + len(categorical),
                                                         len(numeric), len(categorical), float(design.space.size)))
tab_data, tab_explore, tab_next, tab_help = st.tabs(['Data', 'Explore (PCA)', 'Next batch', 'How it works'])

with tab_data:
    st.dataframe(design.frame[numeric + categorical + [target]].head(500), use_container_width=True)
    if n_meas:
        y = design.frame.loc[design.measured, target]
        c1, c2, c3 = st.columns(3)
        c1.metric('Best ' + target, '{:.4g}'.format(y.max()))
        c2.metric('Median', '{:.4g}'.format(y.median()))
        c3.metric('Recipes with a result', n_meas)
        plt, fig, ax = figure(6, 2.8)
        ax.hist(y, bins=40, color=PALETTE[0])
        ax.set_xlabel(target, fontsize=8, color=INK_2)
        ax.set_ylabel('recipes', fontsize=8, color=INK_2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_explore:
    model_pca, xy, loadings = core.pca(design)
    ev = model_pca.explained_variance_ratio_
    left, right = st.columns([3, 2])
    with left:
        st.subheader('Where the recipes sit')
        plt, fig, ax = figure(6.2, 4.6)
        meas = design.measured
        ax.scatter(xy[~meas, 0], xy[~meas, 1], s=6, color=GREY, linewidths=0, label='no result yet')
        sc = ax.scatter(xy[meas, 0], xy[meas, 1], s=9, c=design.frame.loc[meas, target], cmap='viridis',
                        linewidths=0)
        if meas.any():
            fig.colorbar(sc, ax=ax).set_label(target, fontsize=8, color=INK_2)
        ax.set_xlabel('PC1 ({:.0%} of the variation between recipes)'.format(ev[0]), fontsize=8, color=INK_2)
        ax.set_ylabel('PC2 ({:.0%})'.format(ev[1]), fontsize=8, color=INK_2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader('Factors that spread the recipes out')
        st.caption('PCA loadings: which factors differ most between recipes. This describes the experimental '
                   'design, not what drives the result; use the importance below for that.')
        bar_chart(loadings.pow(2).sum(axis=1).pow(0.5), 'loading on PC1 and PC2 combined')
    st.subheader('Factors the model relies on')
    st.caption('Fit a model to the recipes with a result, then shuffle one factor at a time: the drop in R$^2$ is how '
               'much the model depends on it. In-sample, so it describes the model, not a proof of cause.')
    imp_model = st.selectbox('Model for importance', list(core.MODELS), index=1, key='imp_model')
    if st.button('Compute importance', disabled=n_meas < 5):
        with st.spinner('Fitting and shuffling...'):
            bar_chart(core.importance(design, core.MODELS[imp_model]), 'R$^2$ drop when the factor is shuffled')

with tab_next:
    st.subheader('Settings')
    c1, c2, c3 = st.columns(3)
    model_name = c1.selectbox('Model', list(core.MODELS), key='model')
    model_cfg = core.MODELS[model_name]
    round_number = c2.number_input('This is round', 2, 50, 2, help='Round 1 is the initial (random) design.')
    n_rounds = c3.number_input('Rounds planned', 2, 50, 10, help='The exploration schedule moves from exploring to '
                                                                 'exploiting over the planned rounds.')
    c1, c2, c3 = st.columns(3)
    batch_size = c1.number_input('Recipes in the batch', 1, 384, 10)
    sources = ['Unmeasured combinations of the observed levels', 'Rows of the table without a result']
    source = c2.radio('Candidates', sources, help='Combinations: new recipes on the grid of levels you used. Rows: '
                                                  'only recipes listed in the table (e.g. a fixed library).')
    pool_size = c3.number_input('Random candidates to score', 100, 200000, 10000, step=1000, key='pool',
                                disabled=source == sources[1])
    st.markdown('**Modules**')
    c1, c2, c3 = st.columns(3)
    rule = c1.radio('Batch rule', core.BATCH_RULES, key='rule', help='Distance penalty: picks push each other apart (low = the '
                    "data's typical recipe spacing, high = 3 x). Fantasy: pick one, pretend it was measured at its "
                    'prediction, re-fit, pick the next.')
    hill = c2.checkbox('Hill-climbing', key='hill', help='Start from the best-scoring candidates and step to better neighbours '
                       '(one level or one category at a time) until none is better.')
    n_starts = c2.number_input('Climbs', 1, 100, 10, disabled=not hill)
    max_steps = c2.number_input('Max moves per climb', 1, 100, 20, disabled=not hill)
    show_pca = c3.checkbox('Show the scored candidates on the PCA map', value=True)
    seed = c3.number_input('Random seed', 0, 10 ** 6, 0)
    if model_cfg['type'] == 'tabpfn':
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        st.caption('TabPFN-3.5: non-commercial licence; accept it once or set TABPFN_TOKEN before starting the app. '
                   + ('GPU found.' if gpu else 'No GPU found: this will be slow on large candidate sets.'))
    if st.button('Suggest the next batch', type='primary', disabled=n_meas < 2, key='go'):
        with st.spinner('Fitting the model and scoring candidates...'):
            try:
                st.session_state['proposal'] = core.propose(
                    design, model_cfg, batch_size=int(batch_size), round_number=int(round_number),
                    n_rounds=int(n_rounds), pool_size=int(pool_size),
                    source='grid' if source == sources[0] else 'rows', rule=rule, hill=hill,
                    n_starts=int(n_starts), max_steps=int(max_steps), seed=int(seed))
            except Exception as err:  # shown to the user rather than crashing the page
                st.error('{}: {}'.format(type(err).__name__, err))
    proposal = st.session_state.get('proposal')
    if proposal is not None:
        st.subheader('Suggested batch')
        st.caption('Score = {:.2f} x prediction + {:.2f} x uncertainty (round {} of {}).'.format(
            proposal.weights[0], proposal.weights[1], int(round_number), int(n_rounds)))
        st.dataframe(proposal.table, use_container_width=True)
        st.download_button('Download as CSV', proposal.table.to_csv(index=False).encode(), 'next_batch.csv',
                           'text/csv')
        if proposal.climb is not None:
            c = proposal.climb
            st.caption('Hill-climbing: {} climbs, moves to the top: {} (cap {}); {} new recipes found, {} picked.'.format(
                len(c.steps), ', '.join(str(s) for s in c.steps), int(max_steps), len(c.new),
                int(proposal.table['from hill-climbing'].sum())))
        if show_pca:
            xy_pool = model_pca.transform(design.space.embedding(proposal.pool))
            pct = (np.argsort(np.argsort(proposal.scores, kind='stable'), kind='stable') + 0.5) / len(proposal.scores)
            order = np.argsort(pct)
            plt, fig, ax = figure(7.5, 5.2)
            sc = ax.scatter(xy_pool[order, 0], xy_pool[order, 1], c=pct[order], cmap='Blues', s=4, linewidths=0,
                            vmin=0, vmax=1, rasterized=True)
            ax.scatter(xy[design.measured, 0], xy[design.measured, 1], s=8, color=GREY, linewidths=0,
                       label='measured')
            picks = proposal.chosen
            ax.scatter(xy_pool[picks, 0], xy_pool[picks, 1], s=48, facecolors='none', edgecolors=INK, linewidths=1.1,
                       label='suggested batch')
            climbed = picks[picks >= proposal.n_random]
            if len(climbed):
                ax.scatter(xy_pool[climbed, 0], xy_pool[climbed, 1], s=52, marker='^', color=PALETTE[2],
                           edgecolors=INK, linewidths=0.8, label='found by hill-climbing')
            fig.colorbar(sc, ax=ax).set_label("score percentile among the candidates", fontsize=8, color=INK_2)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
            ax.set_xlabel('PC1', fontsize=8, color=INK_2)
            ax.set_ylabel('PC2', fontsize=8, color=INK_2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with tab_help:
    st.markdown("""
**What this does.** Each round, the model learns from every recipe with a result, scores many candidate
recipes by *predicted result + a bonus for uncertainty*, and proposes the batch to run next. Early rounds weigh
uncertainty more (explore); later rounds weigh the prediction more (exploit).

**Modules** (each can be switched on or off; compare them one at a time):
* **Distance penalty**: after each pick, nearby candidates lose score, so the batch is not near-copies of one
  recipe. *Low* uses the data's typical spacing between recipes as the radius, *high* three times that.
* **Fantasy batch**: pick the best candidate, pretend it was measured at its predicted value, re-fit, pick the
  next. The model's uncertainty drops around earlier picks, so later picks go elsewhere.
* **Hill-climbing**: the random candidates rarely include the very best recipes. Starting from the best ones,
  the climb changes one factor by one level (or one category) at a time while the score improves.

**PCA map.** The two directions along which the recipes differ most. Loadings say which factors those are.
PCA describes the design, not the result: use *Factors the model relies on* for that.

**Models.** TabPFN-3.5 is the default in the METIS benchmark; it needs its (non-commercial) licence accepted
and is fast on a GPU. XGBoost is METIS v1's model; the Gaussian process and linear regression are quick baselines.
""")
