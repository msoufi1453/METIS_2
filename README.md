# METIS
* Publication: 
Pandi A*‡, Diehl C*, Yazdizadeh Kharrazi A, Scholz SA, Bobkova E, Faure L, Nattermann M, Adam D, Chapin N, Foroughijabbari Y, Moritz C, Paczia N, NS Cortina, Faulon J-L, Erb TJ‡. A versatile active learning workflow for optimization of genetic and metabolic networks. Nature Communications (2022), https://www.nature.com/articles/s41467-022-31245-z


![METIS_graphAbs](https://user-images.githubusercontent.com/55136474/174248711-66d5eccd-db31-4d1b-a444-024d24ca5543.png)




## Full Performance Can Be Achieved On Google Colab Notebooks:
* [Optimization NoteBook](https://colab.research.google.com/github/amirpandi/METIS/blob/main/METIS_Optimization_Notebook.ipynb)
* [Prediction NoteBook](https://colab.research.google.com/github/amirpandi/METIS/blob/main/METIS_Prediction_Notebook.ipynb)



## Package layout (this fork)
`utils.py` is now a thin shim over the `metis` package, so `from utils import *` in every notebook keeps working with the same function signatures. Run notebooks locally from the repository root (or `pip install -e .`), since the shim imports `metis`.

| Module | Contents |
| --- | --- |
| `metis.design`, `metis.io`, `metis.echo` | The v1 functions (grids, random designs, volumes, Results files, Echo plate layouts) |
| `metis.surrogates` | Pluggable surrogates: `fit(X, y, noise_var=None)` / `predict(X) -> (mean, std)`. `XGBEnsemble` (the incumbent), `RegressorEnsemble`, `GPSurrogate` (Matern), `TabPFNSurrogate` |
| `metis.acquisition` | Pluggable acquisition: `UCB`, `ExpectedImprovement`, and batch selectors `TopK`, `SequentialFantasy`, `LocalPenalization`, `GPBUCB` |
| `metis.space` | `DesignSpace`, the droplet grid as an explicit, vectorised space |
| `metis.optimize` | `bayesian_optimization` / `batch_ucb` (v1 wrappers) and `suggest_combinations` (pluggable entry point) |

| `metis.initialization` | Initial designs on the droplet grid: random (v1), scrambled Sobol, maximin LHS, fill-the-unsampled-region for imported data |
| `metis.schedules` | Exploration schedules: v1 list, constant, linear/exponential decay, GP-UCB rate, adaptive |

Environment: `uv sync --all-extras` creates the pinned environment from `uv.lock` (Python 3.11). `uv run pytest` runs the tests; `tests/test_regression_legacy.py` checks that one seeded METIS round reproduces the original output byte for byte.

## Benchmark (`metis_bench`)
A benchmark harness for improving METIS, with exact regret:

* **Oracles**: gold-regressor oracles from the Borkowski cell-free data (Fig. 1b recipe; the exact maximum comes from enumerating all 4^11 grid recipes) and the PURE data. There is also a synthetic suite of 90 functions: 9 landscape subtypes x 5 calibrated specs x 12/27 factors. Each has an exact global maximum on its droplet grid under the volume budget.
* **Calibration**: the same descriptors (yield distribution, Sobol split, ruggedness, thresholds, learnability, noise ratio) for every synthetic function and both real datasets. Each function gets a realism score and a place on a 2D map (`suite/`).
* **Measurement layer**: triplicate wells, heteroscedastic noise fitted to Borkowski's replicates, outlier wells and plate (batch) effects.
* **Campaign runner**: 32 recipes x 10 rounds, paired seeds, resumable, shardable for clusters.
* **Preregistered** metrics and statistics ([docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)), frozen by hash.

The runbook is in [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Guide
* Apart from the two Google Colab notebooks above that can be tailored for custom applications, all example notebooks used in this work along with their data (combinations and yields) are provided in the Examples/Applications folder of the METIS repository. 

* To tailor METIS for a custom application, check Fig. 2 of the above paper (see below), and a detailed step-by-step guide in Supplementary Note 2 and Supplementary Fig. 4-6.



![Fig2  METIS](https://user-images.githubusercontent.com/55136474/173783016-43c756bd-0f14-4a66-b00e-743660e4bdba.png)




## Examples
This folder contains METIS applications as well as other scripts used in the paper:

## Folders Description:
### Cell_Free_System:
* Data: Data for 10 Days of active learning optimization (Fig. 1e).
* Code Description: This folder uses old version of METIS algorithms and codes. The adapted METIS notebook for lysate cell-free systems is also available under Code_METIS.

### Cell_Free_System_Simulation:
* Data: Data for the simulation on Borkowski et al. (2020) data (Fig. 1b).
* Code Description: This folder uses an old version of METIS algorithms and codes.

### Enzyme_Engineering:
* Data: Data for 800 enzyme mutants from Nattermann et al (2021) (Supplementary Fig. 19).
* Codes: METIS optimization and simulation notebooks. 

### LacI (Gene Circuit):
* Data: Data for 10 Days of active learning optimization (Fig. 3c) and 2 rounds of "20 Most Informative" experiments (Fig. 3k).
* Codes: METIS optimization notebook. 

### PURE (Cell-Free System):
* Data: Data for 10 Days of simulation (Supplementary Fig. 19).
* Codes: METIS prediction notebook.

### TTU (Transcription & Translation Unit):
* Data: Data for 4 Days of active learning optimization (Fig. 4c) and 2 rounds of "20 Most Informative" experiments (Fig. 4e).
* Codes: METIS optimization notebook. 

### CETCH
* Data: Data for 5 Days of "yield" active learning optimization (Fig. 5c) and 5 Days of "efficiency" active learning optimization (Fig. 5e).
* Codes: METIS optimization notebook.





## Documentation
* All functions have Help as DocString (i.e. Help(Name of Function))

### Support
* Feel free to open an issue.
