# The METIS app and `metis.campaign`

The app plans an active-learning campaign round by round: it proposes recipes, you measure
them, it learns and proposes the next ones. Everything it computes comes from the Python
package `metis.campaign`, so the same campaign can be run from a script or a notebook
([In Python](#in-python-metiscampaign)).

## Starting the app

```powershell
uv sync --extra tabpfn35 --extra cpu --extra gui      # once; cu128 instead of cpu with an NVIDIA RTX 50 GPU
.\.venv\Scripts\python.exe -m streamlit run gui/app.py
```

(`.venv/bin/python` on macOS and Linux.) The page opens in the browser. The first screen
takes about 10 seconds to appear after starting, while the scientific libraries load.

* **TabPFN-3.5** needs its (non-commercial) licence accepted once, or `TABPFN_TOKEN` set in
  the shell that starts the app (docs/REPLAY.md). The first round also loads the model
  (about 30 seconds).
* **Only this computer can open the app.** `.streamlit/config.toml` binds it to
  `localhost`; Streamlit's default would let every computer on the network open it, and
  with it the campaign's data. Remove the `address` line there to share the app.
* **Without a GPU** TabPFN scores about 25 candidate recipes a second (8 estimators, 100
  measured recipes, 11 variables, on a 4-core laptop CPU), so the app scores 1,000
  candidates per round by default instead of 10,000. Both numbers are under *Advanced
  settings*.

## The three ways in

| Button | What it opens |
|---|---|
| **Start a new campaign** | The set-up form, empty |
| **Continue a campaign** | An upload for the workbook of the last round, with its results filled in |
| **Tutorial** | The same screens filled in with the cell-free study of Borkowski et al. (2020), with instructions on each screen and a simulated lab |

## Setting up a round

The same form serves a new campaign and every later round.

| Part | What to enter |
|---|---|
| Model | TabPFN-3.5 (the only one for now; see [Adding a model](#adding-a-model)) |
| Day 0 data (optional) | Results you already have: a CSV or Excel table with one column per variable and one for the measured value. Every measurement is a row; replicates of a recipe are rows with the same recipe. *Download a template* gives the right header for the variables in the form. Uploading fills in the variable rows from the columns, for you to adjust |
| Size of the round | Recipes per round (default 10) and replicates of each (default 1) |
| Liquid handling | On by default: the smallest droplet (4 nL), the sample volume, the pipettable volume (empty: the whole sample) and the diluent. Each numeric variable then needs its stock concentration |
| Variables | One row each: **continuous** (minimum, maximum), **discrete** (a number of steps, then that many values; at most 100) or **categorical** (a number of categories, then their names). A unit per numeric variable; units may differ between variables. Up to 6,000 variables with TabPFN |
| Exploration and exploitation | The ratio in the score `ratio x prediction + (1 - ratio) x uncertainty`, and the two batch modules |
| Advanced settings | Candidates to score, TabPFN estimators, the random seed, the cap on hill-climbing moves |

**Generate round N data** stays disabled until at least one variable is complete and
nothing blocks the round; the form says what is missing.

### What the app decides for you

* **The first round is a Latin hypercube** when there is no data: each variable's range is
  cut into as many slices as there are recipes and every slice is used once. Of 20 random
  hypercubes the one whose two closest recipes are farthest apart is kept. With liquid
  handling, recipes that do not fit the sample are replaced by drawing a larger hypercube
  and keeping the fitting recipes that spread best.
* **With day 0 data the model can choose round 1** (a switch, on by default). Day 0 data
  is round 0 in every table and plot.
* **The ratio defaults to the tuned schedule** (docs/REPLAY.md, amendment 4): 0.5 in the
  first model-guided round, 1.0 afterwards. In a new campaign the slider runs from 0 to 1;
  when continuing, from 0.5 to 1.
* **Replicates are separate measurements for the model**: it sees every row, so it also
  sees how much replicates scatter. Plots and "best so far" use the mean per recipe.

### Liquid handling

A dispenser adds whole droplets. With a stock of concentration `S`, droplets of `d` nL and
a sample of `V` µL, one droplet adds `S x d / (1000 x V)` to the sample.

* Every proposed concentration is a whole number of droplets, and a recipe's droplets
  together never exceed the pipettable volume. The diluent tops each sample up to it.
* **Pipettable volume**: how much of each sample the dispenser fills. Leave it empty when
  the dispenser fills the whole sample. When part of every sample is taken by liquids the
  dispenser does not add, enter the rest: for 10 µL samples of which 1.75 µL are taken,
  enter 8.25. Concentrations stay per the whole 10 µL, recipes must fit in 8.25 µL, and the
  pipetting table fills 8.25 µL per well (in whole droplets: 8.248 µL with 4 nL droplets,
  which the table notes). A fixed component that the dispenser does add is a discrete
  variable with one step instead.
* Discrete steps are rounded to whole droplets; the form lists steps that move by more
  than 0.1%.
* **The form checks the stocks**: a maximum above the stock concentration, a range in which
  no droplet count falls, a step below half a droplet, and two steps that become the same
  droplet count all block the round, with what to change. If the components' maximum
  volumes together exceed the sample, a warning lists the largest volumes: either lower
  those maximums or use more concentrated stocks.
* *Pipetting check* (an expander under the variables) shows, per variable, what one
  droplet adds and the volume at its minimum and maximum.
* A fixed component that the dispenser adds (an extract, a buffer) is a discrete variable
  with one step.
* A categorical variable can have a pipetted volume: each category is then its own liquid
  (for example a DNA template). With volume 0 it is set by hand.

## The round screen

| Tab | Content |
|---|---|
| **Recipes** | The proposed recipes in an editable table: steps and categories from a list, continuous values by typing. Model-guided rounds show the predicted result and the range the model expects 80% of measurements to fall in; both update when a recipe is changed. With liquid handling an edit is rounded to whole droplets at once, and an edit that would not fit the sample is undone with a message. Below the table: the workbook download with its instructions |
| **Pipetting information** | With liquid handling: one row per liquid, one column per well, in µL, in the layout of `iDOT_pipetting_sheet.csv` (1A = column 1, row A; wells fill column by column; 96 per plate). Downloads: the table as CSV and, after entering your initials, an iDOT protocol (`.idotx`, an XML file). Also the source plate (what to load) and the plate layout |
| **PCA map** | Every recipe on the two directions along which the mapped recipes differ most. Measured recipes are blue, darker for a higher result, with a shape per round; proposed ones are bright green. Hovering a point identifies it ("Round 1 · Data point 3"); clicking it lists all its values, its result and the model's prediction with the 80% interval |
| **Progress** | From two rounds with results on: every recipe's result by round and the best so far |
| **Importance** | The drop in the model's R² on the measured recipes when a variable is shuffled between recipes |
| **Interactions** | For each pair of variables, how far the effect of changing both is from the sum of the two single effects, in units of the result. 0 means they simply add up |

The last four tabs are also on the set-up screen once there is data. Importance and
interactions need a model call, so they are computed when their button is pressed.

**Hill-climbing** is a switch above the table in model-guided rounds. On: the best-scoring
candidates move one step at a time to better-scoring neighbours and the round is chosen
again; the map shows where the recipes were (hollow), where they are (filled) and dotted
trails between. Off: back to the recipes chosen without it. Both versions are kept.

**Distance penalty** and **fantasy batch** are set before generating (set-up screen). The
penalty slider runs from low (radius of 1 x the typical spacing between candidates) to high
(3 x); see docs/MODULES.md for what they do and how they performed.

### The iDOT protocol

The file is built by `idot_builder.py` of
[idot-protocol-builder](https://github.com/msoufi1453/idot-protocol-builder), copied
unchanged into `metis/idot/` with its template. It names each liquid
`<initials>_<name>`, lays out the source plate, and re-reads and verifies the file before
the app offers it.

## The workbook

Each round is downloaded as one Excel file, `METIS_<campaign>_round<N>.xlsx`:

| Sheet | Content |
|---|---|
| **Experiments** | Every recipe so far, one row per measurement: Sample ID, Round, Data point, Replicate, Plate and Well (liquid handling), the variables with their units, the model's prediction and 80% interval, and the result. The new round's result cells are empty and highlighted |
| **Instructions** | How to fill it in |
| **Metadata** | Readable: seed, variables and ranges, liquid handling, and for every round its settings, software versions, timestamp and the PCA map's parameters |
| **State** | The same, machine-readable. The app reads the campaign from it: leave it as it is |

To continue: type each result into the result column, save, upload under **Continue a
campaign**.

* Every row needs a result; otherwise the upload is refused and the message lists the rows.
  Delete the row of a measurement that failed.
* If what was pipetted differs from the recipe, correct the concentrations in the sheet
  (in all replicates of that Sample ID). The sheet is what the model learns from.
* Extra columns (notes) are kept.

### Changing the design between rounds

* **Ranges, steps and categories** can change freely; measured recipes keep their values.
* **A new variable** needs the value it had in the earlier rounds (for example 0 if it was
  absent); the form asks for it.
* **A removed variable** stays in the workbook, but the model no longer sees it. To stop
  varying a component and keep it in the recipe, make it discrete with one step.

## In Python: `metis.campaign`

```python
from metis.campaign import Campaign, Categorical, Continuous, Discrete, LiquidHandling

campaign = Campaign(
    [Continuous('Mg-glutamate', 0, 10, unit='mM', stock=500),
     Discrete('K-glutamate', [40, 80, 120, 160], unit='mM', stock=4000),
     Categorical('Promoter', ['T7', 'T3'])],
    target='sfGFP', liquid_handling=LiquidHandling(sample_volume_ul=10))

campaign.check()                         # problems with the stocks and volumes, if any
batch = campaign.propose(10)             # round 1: a Latin hypercube
campaign.add_round(batch, replicates=2)
campaign.save('round1.xlsx')             # fill in the sfGFP column, then:

campaign = Campaign.load('round1.xlsx')
batch = campaign.propose(10, penalty=0.0, hill_climb=True)     # model-guided
batch.table                              # recipes, predictions, 80% intervals
campaign.add_round(batch)
plan = campaign.pipetting()              # plan.csv(), plan.idot('MoS', 'round2').data
campaign.save('round2.xlsx')
```

Also: `campaign.add_data(table)` (day 0 data), `campaign.redesign(variables, fill=...)`,
`campaign.predict(recipes)`, `campaign.pca()`, `campaign.importance()`,
`campaign.interactions()`, `campaign.progress()`, `campaign.snap(recipes)`.
`metis.campaign.tutorial.campaign()` is the tutorial's campaign.

| Module | Contents |
|---|---|
| `variables` | `Continuous`, `Discrete`, `Categorical` |
| `liquid` | `LiquidHandling`: droplets, volumes and the checks of the stocks |
| `space` | `RecipeSpace`: drawing recipes (uniform, Latin hypercube), neighbours, encodings |
| `campaign` | `Campaign` and `Proposal` |
| `models` | The model registry |
| `batch` | Batch rules: distance penalty, fantasy batch, both together |
| `analysis` | PCA map, importance, interactions, progress |
| `workbook` | Reading and writing the Excel file |
| `pipetting` | The pipetting sheet and the iDOT protocol |
| `day0` | Reading day 0 tables, guessing variables, the template |
| `tutorial` | The Borkowski campaign and its simulated lab |

### Adding a model

Anything with the `metis.surrogates.Surrogate` interface (`fit`, `predict` returning a mean
and a standard deviation) can be registered; it then appears in the app's model list:

```python
from metis.campaign import register_model
from metis.surrogates import XGBEnsemble

register_model('xgboost', 'XGBoost ensemble (METIS v1)',
               lambda seed=0, **options: XGBEnsemble(random_state=seed, **options),
               max_variables=1000, options={'n_iter': 200, 'ensemble_len': 20})
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_campaign_api.py tests/test_gui.py
```

* `tests/test_campaign_api.py`: the package, with a fast stand-in model; 40 random designs
  check that every draw is distinct, allowed, pipettable and inside the sample; one test
  (marked `tabpfn`) runs a round with the real TabPFN-3.5.
* `tests/test_gui.py`: the app, run headless through its three flows. Streamlit's test
  client cannot drive file uploaders or the editable table, so uploads are handed over the
  way the uploaders do it and table edits are tested through the function that handles them.

## Notes for developers

* `gui/` only draws: `app.py` (entry point), `state.py` (what is remembered between
  reruns), `form.py` (the variable rows), `setup_page.py`, `round_page.py`, `insight.py`
  (the analysis tabs), `charts.py` (Altair) and `texts.py`.
* Streamlit forgets which tab is open when the tabs move on the page, so nothing that only
  sometimes appears is drawn above them; waits and messages have fixed slots.
* A chart with click-selection must be a single layer in Streamlit 1.45, so the PCA map
  draws the hill-climbing paths as rows of small dots.
* The tutorial's stock concentrations are illustrative (25 x each component's highest
  step), chosen so that every step is a whole number of droplets. Its simulated lab
  answers with the distance-weighted mean yield of the 5 most similar recipes the study
  measured.
