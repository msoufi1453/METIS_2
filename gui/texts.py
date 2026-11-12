"""The app's longer texts: the tutorial's instructions and the explanations next to the controls."""

INTRO = ('METIS proposes the next experiments from the ones already measured. Each round, a model learns from every '
         'result so far and picks the recipes to try next.')

LATIN_HYPERCUBE = ('**This round is not chosen by a model: there is no data to learn from yet.** Its recipes are '
                   'spread over the whole design as a Latin hypercube: each variable\'s range is cut into as many '
                   'slices as there are recipes and every slice is used once, so no region is left out. The ratio '
                   'below takes effect from the first model-guided round.')

EXPLOIT_HELP = ('Each candidate recipe is scored as ratio x predicted result + (1 - ratio) x uncertainty. 0 picks the '
                'recipes the model is least sure about (explore), 1 the ones it predicts to be best (exploit).')

EXPLOIT_DEFAULT = ('Default: {:.1f}. In our replays of measured datasets the best schedule was 0.5 in the first '
                   'model-guided round and 1.0 afterwards.')

PENALTY_HELP = ('After each pick, candidates near it lose score, so the round is not ten near-copies of one recipe. '
                'Low: within about one typical spacing between candidates. High: three times that distance. In our '
                'tests on measured data a low penalty helped and a high one did not help more.')

FANTASY_HELP = ('Recipes are picked one by one: after each pick the model pretends it was measured at its predicted '
                'value and is consulted again, so later picks go where it is still unsure. Slow without a GPU: one '
                'more model call per recipe.')

HILL_HELP = ('The random candidates rarely include the very best recipe. Hill-climbing starts from the best-scoring '
             'candidates and changes one variable by one step at a time while the score improves. Switch it off to '
             'go back to the recipes chosen without it.')

ROUNDING = ('Liquid handling is on, so concentrations are rounded to what can be pipetted: a whole number of droplets '
            'of the stock. A value you type is rounded straight away, and a change that would not fit the sample '
            'volume is undone.')

DOWNLOAD_STEPS = '''
1. Download the workbook and run the experiments in its **Experiments** sheet (the highlighted rows).
2. Type each result into the **{target}** column. One row is one measurement: replicates have their own rows.
3. Every row needs a result. If a measurement failed, delete its row.
4. Save the file and upload it under **Continue a campaign** to get round {next}.

The workbook also holds a **Metadata** sheet (seed, ranges, model, settings, software versions, the PCA map's
parameters, a timestamp) and a **State** sheet the app reads the campaign from: leave those as they are.
'''

IMPORTANCE = ('How much the model relies on each variable: the drop in its R² on the measured recipes when that '
              'variable is shuffled between recipes. It describes what the model has learned so far, not a proven cause.')

INTERACTIONS = ('How much the effect of one variable depends on another, in the model. For two measured recipes, '
                'swap in variable A, variable B, then both: the value is how far the change from both is from the '
                'sum of the two single changes (in units of {target}, averaged over pairs of recipes). 0 means the '
                'two simply add up.')

PCA = ('Each point is a recipe; recipes that are alike sit close together. The two axes are the directions along '
       'which the mapped recipes differ most. Hover a point to identify it, click it for all its values.')

TUTORIAL_SETUP = '''
**Tutorial, step 1 of 3: set up the campaign.** This is the screen you get from *Start a new campaign*, filled in
with a real study: Borkowski et al. (2020) optimised a cell-free reaction buffer of 11 components.

* **Day 0 data** is what you have measured before the campaign. Here it is the study's first plate: 99 recipes
  with their yield. *Download a template* shows the file format: one column per variable, one for the result,
  and one row per measurement (replicates are extra rows). You can start without day 0 data; round 1 is then a
  Latin hypercube instead of a model's choice.
* **Recipes per round** and **replicates**: 10 recipes, each measured once.
* **Variables**: each row is one thing a recipe can change. All 11 are *discrete* here (four concentrations
  each). *Continuous* takes a minimum and a maximum; *categorical* takes names (a strain, a promoter).
* **Liquid handling** is on: every variable has a stock concentration, and the app keeps each recipe pipettable
  and inside the 10 µL sample. The stocks here are made up for the tutorial (25 x the highest step).
* **Exploration / exploitation** is at the recommended 0.5 for the first model-guided round.

Nothing needs changing. Scroll down and press **Generate round 1 data**. TabPFN takes about a minute without a
GPU.
'''

TUTORIAL_ROUND = '''
**Tutorial, step 2 of 3: the proposed round.**

* **Recipes**: the 10 recipes to run, with the model's predicted yield and the range it expects 80% of
  measurements to fall in. Change any cell: the prediction updates. Try the **Hill-climbing** switch to see the
  recipes move and switch it off to go back.
* **Pipetting information**: the volume of every stock for every well, as a table, a CSV and an iDOT protocol.
* **PCA map**: the proposed recipes in green among the measured ones (darker = higher yield).
* **Progress**, **Importance**, **Interactions**: what the data and the model say so far.

In a real campaign you now download the workbook, run the experiments and type in the results. Here, press
**Simulate the lab**: the results are filled in from the recipes the study measured (each proposed recipe gets
the weighted mean of the 5 most similar measured ones), and you continue as if you had uploaded the workbook.
'''

TUTORIAL_NEXT = '''
**Tutorial, step 3 of 3: continue the campaign.** This is the screen you get after uploading a filled-in workbook
under *Continue a campaign*. The simulated results of round {last} are in; the tabs above show them.

* The ratio now defaults to **{exploit:.1f}** (the tuned schedule) and can be set between 0.5 and 1.
* **Distance penalty** and **Fantasy batch** keep a round from being near-copies of one recipe.
* You can change the variables for the coming rounds: widen a range, add a step, add or remove a variable.

Press **Generate round {next} data** to go on, as often as you like. To practise the real hand-over, download a
round's workbook, fill in the results yourself and upload it under *Continue a campaign*.
'''
