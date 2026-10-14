"""One seeded round of METIS, replayed exactly as the optimization notebook runs it.

``run_scenario(mod)`` drives a module that exposes the legacy ``utils`` API and returns
``{name: csv_text}``. The golden files in ``golden/`` were produced by running this
against ``utils_v1_original.py`` (byte-identical to ``utils.py`` at commit 6eb5766) in the
pinned environment. Comparing CSV text with full ``repr`` float precision makes the check
exact.

Keep this file stable: changing it invalidates the golden files.
"""
import io

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Example settings from METIS_Optimization_Notebook.ipynb (cells 8 and 10).
M = 20
DROP_NL = 100
REACTION_NL = 20000
MAX_NL = 9300
FIXED_PARTS = {'Lysate': 0.33, 'EnergyBuffer': 0.175, 'IPTG-Water': 0.025}
EXPLORATION = {1: 1.41, 2: 1.41, 3: 1.41, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.5, 9: 0.5, 10: 0.5}

LIMITS = {
    'PEG': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 'Conc_Stock': 50.0, 'Alternatives': None},
    'DTT': {'Conc_Min': 0.0, 'Conc_Max': 4.0, 'Conc_Values': None, 'Conc_Stock': 100.0, 'Alternatives': None},
    'Spermidine': {'Conc_Min': 0.2, 'Conc_Max': 2.0, 'Conc_Values': None, 'Conc_Stock': 40.0, 'Alternatives': None},
    'T7Pol': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [0.0078, 0.0156, 0.0273, 0.039, 0.0507, 0.0624, 0.0702, 0.078], 'Conc_Stock': 0.78, 'Alternatives': None},
    't-RNA': {'Conc_Min': 0.04, 'Conc_Max': 0.04, 'Conc_Values': None, 'Conc_Stock': 8.0, 'Alternatives': None},
    'AminoAcids': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [0.75], 'Conc_Stock': 6.0, 'Alternatives': None},
    'TF-plasmid': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0], 'Conc_Stock': 2000.0, 'Alternatives': None},
    'Circuit-plasmid': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [10.0], 'Conc_Stock': 1000.0, 'Alternatives': ['pTAC', 'pTHS', 'pDC', 'pSTAR']},
}

# A factor with two stock solutions exercises find_stock and the *_Stock_Type columns.
MULTI_STOCK_LIMITS = {
    'Mg': {'Conc_Min': None, 'Conc_Max': None, 'Conc_Values': [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], 'Conc_Stock': [20.0, 200.0], 'Alternatives': None},
    'K': {'Conc_Min': 10.0, 'Conc_Max': 60.0, 'Conc_Values': None, 'Conc_Stock': 1000.0, 'Alternatives': None},
}

# Notebook "Default" ensemble (RandomCV = False).
def default_regressors():
    return [XGBRegressor(objective='reg:squarederror', n_estimators=n, learning_rate=l, max_depth=6,
                         min_child_weight=4, subsample=0.9, gamma=0.4, colsample_bytree=0.9)
            for n in (10, 20, 30, 40, 50) for l in (0.01, 0.03, 0.1, 0.2)]


PARAM_GRID = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "max_depth": [2, 3, 4, 6, 8],
    "n_estimators": [10, 20, 40, 60, 80, 100, 300, 500],
    "reg_lambda": [1, 1.5, 2],
    "gamma": [0, 0.1, 0.4, 0.6],
    "min_child_weight": [1, 2, 4]}


def tuned_regressors(data, label, n_iter=12, ensemble_len=20):
    """Notebook cell 22 with RandomCV = True; n_jobs=1 so the run is deterministic."""
    grid = RandomizedSearchCV(estimator=XGBRegressor(objective='reg:squarederror'),
                              param_distributions=PARAM_GRID, cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=1, n_iter=n_iter)
    grid.fit(data.values, label.values)
    results = pd.DataFrame(grid.cv_results_).sort_values('mean_test_score', ascending=False)
    return [XGBRegressor(objective='reg:squarederror', **p) for p in results.params.iloc[0:ensemble_len, ]]


def toy_yield(conc):
    """Deterministic stand-in for the plate reader so the round has labels."""
    x = conc.astype(float)
    y = (np.log1p(x['TF-plasmid']) * x['T7Pol'] * 40
         + 3 * x['PEG'] - 0.4 * (x['PEG'] - 3.0) ** 2
         + 2.0 * x['Spermidine'] - 0.8 * x['DTT']
         + 1.5 * x.get('Circuit-plasmid_pTHS', 0) + 0.7 * x.get('Circuit-plasmid_pDC', 0))
    return pd.DataFrame({'yield': np.round(y.values, 6)})


def _csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=True, lineterminator='\n')
    return buf.getvalue()


def desired_columns(limits):
    cols = []
    for key, value in limits.items():
        cols.append(key)
        if value['Alternatives']:
            cols += ['{}_{}'.format(key, i) for i in value['Alternatives']]
    return cols


def run_scenario(mod):
    out = {}

    # Notebook cell 11: grid inspection.
    lists = {}
    for n, (key, value) in enumerate(LIMITS.items()):
        concs, vols = mod.allowed_output(value, reaction_vol_nl=REACTION_NL, drop_size_nl=DROP_NL, verbose=0)
        out['allowed_{}'.format(key)] = repr((concs, vols))
        lists[n] = vols
    out['percentage_possible'] = repr(mod.percentage_possible(lists, threshold=MAX_NL))

    # Day 1: random design and volumes (cells 15 and 17).
    np.random.seed(20240601)
    conc_1 = mod.random_combination_generator(LIMITS, number_of_combination=M, reaction_vol_nl=REACTION_NL,
                                              max_nl=MAX_NL, drop_size_nl=DROP_NL, return_df=True)
    out['Concentrations_1'] = _csv(conc_1)
    vol_1 = mod.concentration_to_volume(conc_1, LIMITS, reaction_mixture_vol_nl=REACTION_NL, fixed_parts=FIXED_PARTS)
    out['Volumes_1'] = _csv(vol_1)

    # Round 2 (cells 21 to 27).
    cols = desired_columns(LIMITS)
    data = pd.concat([pd.DataFrame(columns=cols), conc_1[cols]]).reset_index(drop=True)
    label = pd.concat([pd.DataFrame(columns=['yield']), toy_yield(conc_1)]).reset_index(drop=True)

    np.random.seed(7)
    regs = tuned_regressors(data, label)
    np.random.seed(11)
    conc_2 = mod.bayesian_optimization(regs, data, label, LIMITS, final_order=cols, df_main=data,
                                       reaction_vol_nl=REACTION_NL, max_nl=MAX_NL, drop_size_nl=DROP_NL,
                                       exploitation=1, exploration=EXPLORATION[2], test_size=M, pool_size=1500,
                                       verbose=0, day=1, days_range=[M] * 10)
    out['Concentrations_2_tuned'] = _csv(conc_2)
    out['Volumes_2_tuned'] = _csv(mod.concentration_to_volume(conc_2, LIMITS, reaction_mixture_vol_nl=REACTION_NL,
                                                              fixed_parts=FIXED_PARTS))

    np.random.seed(12)
    conc_2d = mod.bayesian_optimization(default_regressors(), data, label, LIMITS, final_order=cols, df_main=data,
                                        reaction_vol_nl=REACTION_NL, max_nl=MAX_NL, drop_size_nl=DROP_NL,
                                        exploitation=1, exploration=0.5, test_size=M, pool_size=1500)
    out['Concentrations_2_default'] = _csv(conc_2d)

    # Single-sample mode used by batch_ucb, and batch_ucb itself.
    np.random.seed(13)
    best, expected = mod.bayesian_optimization(default_regressors(), data, label, LIMITS, final_order=cols,
                                               df_main=data, max_nl=MAX_NL, pool_size=500, batch_ucb=True)
    out['bo_single'] = _csv(best) + repr(float(expected))
    np.random.seed(14)
    out['batch_ucb'] = _csv(mod.batch_ucb(default_regressors()[:4], data, label, LIMITS, final_order=cols,
                                          df_main=data, max_nl=MAX_NL, test_size=3, pool_size=300))

    # Multi-stock factor path.
    np.random.seed(15)
    conc_ms = mod.random_combination_generator(MULTI_STOCK_LIMITS, number_of_combination=12, reaction_vol_nl=REACTION_NL,
                                               max_nl=MAX_NL, drop_size_nl=DROP_NL, return_df=True)
    out['Concentrations_multistock'] = _csv(conc_ms)
    out['Volumes_multistock'] = _csv(mod.concentration_to_volume(conc_ms, MULTI_STOCK_LIMITS,
                                                                 reaction_mixture_vol_nl=REACTION_NL, fixed_parts={}))

    # Echo / plate-layout outputs (cells 59 to 62).
    for plate_384, vertical, triplicate in [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]:
        tag = '{}_{}_{}'.format(384 if plate_384 else 96, 'v' if vertical else 'h', 'tri' if triplicate else 'single')
        inter = mod.put_volumes_to_wells(vol_1.drop(columns=['Circuit-plasmid_Type']), plate_384_well=plate_384,
                                         vertical=vertical, triplicate=triplicate, starting_well='A1')
        out['wells_' + tag] = _csv(inter)
        _, aggregated = mod.source_to_destination(inter, desired_order=None, reset_index=True, check_zero=tag.endswith('tri'))
        out['echo_' + tag] = _csv(aggregated)
    named, plates = mod.put_volumes_to_384_wells(vol_1[['PEG', 'DTT']], starting_well='B3', vertical=True)
    out['plate384_B3_v'] = _csv(named) + _csv(plates['PEG'])
    named, plates = mod.put_volumes_to_96_wells(vol_1[['PEG', 'DTT']], starting_well='C2', vertical=False)
    out['plate96_C2_h'] = _csv(named) + _csv(plates['DTT'])
    return out
