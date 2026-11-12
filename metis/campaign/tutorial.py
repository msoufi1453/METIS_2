"""The tutorial campaign: the cell-free buffer optimisation of Borkowski et al.

Borkowski O. et al., "Large scale active-learning-guided exploration for in vitro protein
production optimization", Nature Communications 11, 1872 (2020): 11 buffer components,
each at 10%, 30%, 50% or 100% of its highest concentration, and the protein yield
relative to a reference buffer, over ten plates. ``data/borkowski.csv`` holds the 1,017
measured recipes (the copy in ``data/replay/``).

The tutorial starts from the study's first plate as day 0 data. ``simulate`` stands in for
the lab so the next rounds can be tried without one: it answers from the recipes the
study measured.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import Campaign
from .liquid import LiquidHandling
from .space import RecipeSpace
from .variables import Discrete

DATA = Path(__file__).with_name('data') / 'borkowski.csv'
TARGET = 'Yield'
NAME = 'Borkowski tutorial'
SAMPLE_VOLUME_UL = 10.0
STOCK_FACTOR = 25           # illustrative stocks: 25 x the highest level, so every level is a whole number of droplets
NEIGHBOURS = 5
#          column in the file, name, unit
COMPONENTS = [('mg_gluta', 'Mg-glutamate', 'mM'), ('K_gluta', 'K-glutamate', 'mM'), ('aa', 'Amino acids', 'mM'),
              ('trna', 'tRNA', 'mg/mL'), ('coa', 'CoA', 'mM'), ('nad', 'NAD', 'mM'), ('camp', 'cAMP', 'mM'),
              ('folinic_acid', 'Folinic acid', 'mM'), ('spermidin', 'Spermidine', 'mM'), ('pga', '3-PGA', 'mM'),
              ('nucleo_mix', 'NTPs', 'mM')]


def load():
    """Every measured recipe: the 11 components under their names, ``Yield`` and the study's ``Plate``."""
    raw = pd.read_csv(DATA)
    frame = raw.rename(columns={column: name for column, name, _ in COMPONENTS})
    frame = frame.rename(columns={'yield': TARGET, 'day': 'Plate'})
    return frame[[name for _, name, _ in COMPONENTS] + [TARGET, 'Plate']]


def variables(data=None):
    """The 11 components as discrete variables with the study's four levels each. The stock
    concentrations are illustrative (25 x the highest level), not the study's."""
    data = load() if data is None else data
    out = []
    for _, name, unit in COMPONENTS:
        levels = sorted(float(x) for x in data[name].unique())
        out.append(Discrete(name, levels, unit=unit, stock=STOCK_FACTOR * levels[-1]))
    return out


def liquid_handling():
    return LiquidHandling(sample_volume_ul=SAMPLE_VOLUME_UL)


def day0(data=None):
    """The study's first plate: the measurements a tutorial campaign starts from."""
    data = load() if data is None else data
    return data.loc[data['Plate'] == 1].drop(columns='Plate').reset_index(drop=True)


def campaign(seed=0, model=None):
    """A campaign on the tutorial's design with the first plate as day 0 data."""
    options = {} if model is None else {'model': model}
    out = Campaign(variables(), target=TARGET, liquid_handling=liquid_handling(), seed=seed, name=NAME, **options)
    out.add_data(day0())
    return out


def simulate(recipes, replicate=None, seed=0):
    """Stand-in measurements for ``recipes`` (a table with the 11 components).

    The value is the distance-weighted mean yield of the 5 most similar recipes the study
    measured; a recipe the study measured gets its own measured mean. With ``replicate``
    numbers, replicates after the first get noise of the study's typical replicate spread,
    so that they differ as real ones do. The answer is fixed for a given input and seed.
    """
    from sklearn.neighbors import KNeighborsRegressor
    data = load()
    names = [name for _, name, _ in COMPONENTS]
    measured = data.groupby(names, as_index=False)[TARGET].mean()
    space = RecipeSpace(variables(data))
    model = KNeighborsRegressor(n_neighbors=NEIGHBOURS, weights='distance')
    model.fit(space.embedding(space.encode(measured[names])), measured[TARGET].to_numpy())
    recipes = pd.DataFrame(recipes).reset_index(drop=True)
    values = model.predict(space.embedding(space.encode(recipes[names])))
    if replicate is not None:
        spread = float(pd.read_csv(DATA)['yield_std'].median())
        noise = np.random.default_rng(seed).normal(0.0, spread, size=len(values))
        values = np.where(np.asarray(replicate) > 1, values + noise, values)
    return np.round(np.maximum(values, 0.0), 4)
