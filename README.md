# CPSC 322 Final Project: Meteorite Landing Classification

- **Dataset**: NASA Meteorite Landings (`Meteorite_Landings.csv`), ~38k rows. Cleaned with pure Python (no pandas), then stratified sample of 1500 rows for EDA. Target label: `fall` (Fell vs Found).
- **Environment**: Python 3.9. Standard library + `matplotlib`/`seaborn` for plots. All algorithms implemented in `mysklearn` (KNN, Naive Bayes, Decision Tree, Random Forest, Dummy).
- **Repo layout**:
  - `FinalReport.ipynb` — full report/notebook (cleaning, EDA, modeling).
  - `ProjectProposal*.ipynb` — proposal drafts.
  - `mysklearn/` — custom classifiers; random forest supports stratified split, node random features, validation top-k.
  - `tests/` — unit tests for KNN, Decision Tree, Random Forest, evaluation helpers.
  - `Meteorite_Landings.csv` — input data.

## How to run

1) Ensure Python 3.9 and install plotting libs if missing:
```bash
pip install matplotlib seaborn
```
2) Open `FinalReport.ipynb` and Run All. Notebook already loads CSV via `csv.DictReader`, cleans rows, stratifies 1/3 test set, fits NB/DT/RF/KNN, prints accuracies and confusion matrices, and renders EDA figures.
3) Random forest parameter sweep is inside the notebook (cells near the end). Modify `param_sets` to try different N/M/F if desired.

## How to test

Run all unit tests:
```bash
python -m pytest tests
```
Key coverage:
- `test_myrandomforest.py` — stratified split, bootstrap/feature subsample, voting.
- `test_mydecisiontree.py` — entropy/gain splits with random feature cap.
- `test_myknn.py`, `test_myevaluation.py` — KNN and evaluation helpers.

## Notes

- No pandas dependency; all preprocessing uses csv/standard library.
- DecisionTree/RandomForest include `random_state` and `max_features` for reproducible feature subsampling.
- Class imbalance is large (Found >> Fell); interpret accuracy with confusion matrices and balanced accuracy.

## Data source

NASA Open Data: Meteorite Landings (public domain). Downloaded once and stored as `Meteorite_Landings.csv`.
