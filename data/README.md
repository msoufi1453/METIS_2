# Data

* `replay/`: the measured datasets of the replay experiments (PURE, Borkowski, CETCH),
  concatenated from the files in `Examples/` (see `metis_bench/oracles/measured.py`).
* `external/` (not in the repository): third-party data that may not be redistributed here.

## Cell growth (Aida & Ying 2024)

Aida H, Ying B-W. Data-driven discovery of the interplay between genetic and environmental
factors in bacterial growth. *Communications Biology* 7, 1691 (2024).
https://doi.org/10.1038/s42003-024-07347-3

Saturated population density (K) of 115 *E. coli* strains in 135 synthetic media (the
paper's Supplementary Data). The article and its supplementary material are licensed
CC BY-NC-ND 4.0: non-commercial use and unmodified sharing with attribution are allowed,
but adapted versions may not be shared. This repository is public, so the spreadsheet is
not committed, and the cleaning (dropping the three ID columns, the paper's SD < 0.1
filter) happens in code at load time (`metis_bench/oracles/cell_growth.py`).

To use it, download the growth-profile spreadsheet from the article's Supplementary
Information and save it, unchanged, as

    data/external/cell_growth_Data_Data.xlsx

or set `METIS_CELL_GROWTH` to its path. Expected columns: `Unique ID`, `Combination ID`,
`Medium ID`, `Condition ID_unique`, `Strain (gene)`, 46 `log10[...] (mM)` concentrations,
`K_ave` and `K_ave_sd`; 15,525 rows, of which 13,944 pass the filter.
