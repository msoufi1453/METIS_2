"""Active-learning campaigns in a few lines: the engine behind the METIS app (gui/app.py).

    from metis.campaign import Campaign, Categorical, Continuous, Discrete, LiquidHandling

    campaign = Campaign(
        [Continuous('Mg-glutamate', 0, 10, unit='mM', stock=500),
         Discrete('K-glutamate', [40, 80, 120, 160], unit='mM', stock=3000),
         Categorical('Promoter', ['T7', 'T3'])],
        target='sfGFP', liquid_handling=LiquidHandling(sample_volume_ul=10))

    batch = campaign.propose(10)            # round 1: a Latin hypercube
    campaign.add_round(batch, replicates=2)
    campaign.save('round1.xlsx')            # fill in the results, then:

    campaign = Campaign.load('round1.xlsx')
    batch = campaign.propose(10, penalty=0.0, hill_climb=True)    # model-guided
    batch.table                             # recipes, predictions, 80% intervals
    campaign.add_round(batch)
    campaign.pipetting().csv()              # the pipetting sheet; .idot('MoS', 'round2') for the iDOT file
    campaign.save('round2.xlsx')

Modules: ``variables`` (what a recipe can change), ``liquid`` (droplets and volumes),
``space`` (drawing and encoding recipes), ``models`` (the model registry), ``batch``
(batch rules), ``analysis`` (PCA map, importance, interactions), ``workbook`` (the Excel
file), ``pipetting`` (pipetting sheet and iDOT protocol), ``day0`` (data from before the
campaign) and ``tutorial`` (the Borkowski example). docs/APP.md describes the app.
"""
from . import analysis, day0, pipetting, tutorial, workbook
from .campaign import Campaign, CampaignError, Fitted, Proposal
from .liquid import Issue, LiquidHandling
from .models import (DEFAULT_MODEL, MODELS, ModelSpec, default_pool_size, has_gpu, make_model, model_spec,
                     register_model)
from .pipetting import IdotFile, PipettingPlan
from .space import RecipeSpace
from .variables import MAX_LEVELS, Categorical, Continuous, Discrete, VariableError
from .workbook import WorkbookError

__all__ = ['Campaign', 'CampaignError', 'Categorical', 'Continuous', 'DEFAULT_MODEL', 'Discrete', 'Fitted',
           'IdotFile', 'Issue', 'LiquidHandling', 'MAX_LEVELS', 'MODELS', 'ModelSpec', 'PipettingPlan', 'Proposal',
           'RecipeSpace', 'VariableError', 'WorkbookError', 'analysis', 'day0', 'default_pool_size', 'has_gpu',
           'make_model', 'model_spec', 'pipetting', 'register_model', 'tutorial', 'workbook']
