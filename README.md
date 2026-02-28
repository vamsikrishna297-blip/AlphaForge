# AlphaForge(AFF)


### Data Preparation
Similar to [AlphaGen](https://github.com/RL-MLDM/alphagen), We Use [Qlib](https://github.com/microsoft/qlib#data-preparation) as data save tool and download data from free & open-source data source  [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5).

Please install Qlib [Qlib](https://github.com/microsoft/qlib) first

Then download stock data through running `data_collection/fetch_baostock_data.py`

The next, Modify the correspoding `/path/for/qlib_data` in `gan.utils.data.py` to the data you downloaded (the dafault setting is `~/.qlib/qlib_data/cn_data_rolling`)


#### Price adjustment factor (`factor`) in exported Qlib data

For CN data exported by `data_collection/fetch_baostock_data.py`, the column `factor` in Qlib files is sourced from Baostock `foreAdjustFactor` (renamed during export).

In `alphagen_qlib/stock_data.py` daily raw features use this convention:
- `open`, `close`, `high`, `low`, and `vwap` are multiplied by `$factor`
- `volume` is divided by `$factor`

This keeps price/volume features consistent under corporate-action adjustments (e.g., splits/dividends).


### Run Our Model

#### stage1: Minning alpha factors
```shell
python train_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --zoo_size=100
```

Here,
- `instruments` is the dataset to use, e.g., `csi300`,`csi500`.
- `seeds` is random seed list, e.g., `[0,1,2]` or `[0]`. 
- `train_end_year` is the last year of training set, when train_end_year is 2020,the train,valid and test set is seperately: `2010-01-01 to 2020-12-31`,`2021-01-01 to 2021-12-31`,`2022-01-01 to 2022-12-31`
- `save_name` is the prefix when saving running results. `zoo_size` is the num of factors to save at stage 1 mining model.

Sanity-check files per seed (`out/<save_name>_<instruments>_<train_end_year>_<seed>/`):
- `training_sanity.csv`
- `training_summary.json`

#### stage2: Combining alpha factors
```shell
python combine_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --n_factors=10 --window=inf --sanity_sample_n=5 --bucket_percentiles="[0,10,20,30,40,50,60,70,80,90,100]"
```
Here `instruments,train_end_year,seeds,save_name`,` must be the same as it in stage 1
- `n_factors` is the num of factors used at each day, it should be less than or equal to `zoo_size` in stage 1.
- `window` is the slicing window that is used to evaluate the alpha factors in order to dynamicly select and cobine.
- `bucket_percentiles` controls extra return-bucket CSV edges (in percent).

Sanity-check files per seed (`out/<save_name>_<instruments>_<train_end_year>_<seed>/`):
- `combine_sanity_<train_end>_<n_factors>_<window>_<seed>.csv`
- `combine_summary_<train_end>_<n_factors>_<window>_<seed>.json`
- `combine_quintiles_<train_end>_<n_factors>_<window>_<seed>.csv` (q1..q5 and q5-q1 time series)
- `combine_bucket_returns_<train_end>_<n_factors>_<window>_<seed>.csv` (custom percentile bucket returns)
- `combine_samples_<train_end>_<n_factors>_<window>_<seed>.json` (sample day-level coefficient/factor details + previews, including VWAP(t+1,t+21) used in target)

`combine_sanity_*.csv` now also includes:
- equal-weighted index return (`equal_weighted_index_ret`)
- cross-sectional return dispersion (`cs_return_dispersion`)
- top selected factor formulas in `top_formula_1` ... `top_formula_10`
- `pred_valid_<train_end>_<n_factors>_<window>_<seed>.pt`
- `pred_<train_end>_<n_factors>_<window>_<seed>.pt`



Filtered-stock combine (using a custom instruments-like file such as `nifty300.txt`):
```shell
python combine_AFF_filtered.py --instruments=nifty500 --stock_list_file=path/to/nifty300.txt --train_end_year=2020 --seeds='[0,1,2]' --save_name=test --n_factors=10 --window=inf --sanity_sample_n=5
```
This writes the same `combine_sanity/combine_summary/combine_quintiles/combine_samples/pred*` artifacts, with `_filtered` suffix in prediction/output names.

Equal-weight combine (no dynamic regression weights):
```shell
python combine_AFF_equal.py --instruments=csi300 --train_end_year=2020 --seeds='[0,1,2,3,4]' --save_name=test --n_factors=10 --window=inf --sanity_sample_n=5 --sanity_sample_date=2022-06-15
```
Outputs per seed under `out/<save_name>_<instruments>_<train_end_year>_<seed>/`:
- `pred_valid_equal_<train_end>_<n_factors>_<window>_<seed>.pt`
- `pred_equal_<train_end>_<n_factors>_<window>_<seed>.pt`
- `combine_equal_sanity_<train_end>_<n_factors>_<window>_<seed>.csv`
- `combine_equal_quintiles_<train_end>_<n_factors>_<window>_<seed>.csv` (q1..q5 and q5-q1 time series)
- `combine_equal_summary_<train_end>_<n_factors>_<window>_<seed>.json`
- `combine_equal_samples_<train_end>_<n_factors>_<window>_<seed>.json` (sample-date/day factor coefficients, IC stats, prediction preview + stock ids + VWAP(t+1,t+21) used in target)

#### stage3: Show the results

You could run the ipython notebook file 

```shell
exp_AFF_calc_result.ipynb
```

to generate and concat experiment result.

Or run a script that writes stage-3 sanity outputs to files:
```shell
python exp_AFF_calc_result.py --instruments=csi300 --train_end_year=2020 --seeds='[0,1,2,3,4]' --save_name=test --n_factors=10 --window=inf
```
This creates files under `out/results/` including:
- `stage3_seed_metrics_<save_name>_<instruments>_<train_end_year>.csv`
- `stage3_agg_metrics_<save_name>_<instruments>_<train_end_year>.csv`
- `stage3_predictions_<save_name>_<instruments>_<train_end_year>.csv`
- `stage3_summary_<save_name>_<instruments>_<train_end_year>.json`


### Run baseline experiments

The experiment process of other models is similar to running our AFF model, Except that none of the other models have a combine step.

#### GP:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### RL:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### DSO:

train: `train_RL.py`, show result: `exp_RL_calc_result.ipynb`

#### ML models including XGBoost, LightGBM and MLP:

train & show results: `exp_ML_train_and_result.ipynb`





### Generate Indian (NSE) data in Qlib format

By default, the script uses a local symbol file at `data_collection/nse_symbols.txt` (URL bypass).

```shell
python data_collection/fetch_india_data.py \
  --save_path=~/.qlib/tmp_nse \
  --qlib_export_path=~/.qlib/qlib_data/in_data_rolling
```

To use your own local list:

```shell
python data_collection/fetch_india_data.py \
  --save_path=~/.qlib/tmp_nse \
  --qlib_export_path=~/.qlib/qlib_data/in_data_rolling \
  --symbols_file=path/to/EQUITY_L.csv
```

`symbols_file` can be either:
- yfinance-style tickers (`RELIANCE.NS`, `TCS.NS`) in txt/csv, or
- NSE `EQUITY_L.csv` format with a `SYMBOL` column (the script auto-converts to `<SYMBOL>.NS`).

A URL option still exists (`--symbols_url`) as fallback only if local file is unavailable.

Outputs are similar to the CN script:
- `save_path/k_data/*.pkl`
- `save_path/export/*.csv`
- `save_path/symbol_map.csv`
- `qlib_export_path/{calendars,features,instruments}`

The script also writes `instruments/all.txt` and `instruments/nifty500.txt` so the training pipeline can use:

```shell
export QLIB_PATH_DAY=~/.qlib/qlib_data/in_data_rolling
export QLIB_REGION=cn
python train_AFF.py --instruments=nifty500 --train_end_year=2020 --seeds=[0] --save_name=test
```

> Note: `nifty500.txt` is generated from the symbols you provide in `--symbols_file`. If that file is the actual NIFTY 500 constituent list, `--instruments=nifty500` will match NIFTY 500.


Troubleshooting (`No data returned from qlib`):
- Sample JSON day `date` is now aligned to evaluated tensor indices (accounts for Qlib backtrack offset), so it matches the price fields used for target construction.
- If your dumped dataset does not include `factor`, the loader now falls back to non-factor raw expressions automatically.
- If your requested date range starts before the dataset calendar start, upgrade to the latest code (calendar bounds are now clamped to avoid empty loads from negative indexing).
- In notebooks, run env exports in the same shell invocation as training (or use `%env`) because separate `!export ...` lines do not persist across commands.
- If your `instruments/*.txt` uses uppercase codes but `features/` folders are lowercase (or vice-versa), rebuild the dataset so both use the same code format. The exporter now writes lowercase qlib codes (e.g., `nsreliance`) consistently.
- Ensure `QLIB_PATH_DAY` points to the real dumped dataset root (the folder containing `calendars/`, `features/`, `instruments/`).
- Ensure `instruments/<name>.txt` exists for the value passed to `--instruments` (for example `instruments/all.txt` or `instruments/nifty500.txt`).
- For NSE data, set `QLIB_REGION=cn` (Qlib region keys like `in` are not supported by pyqlib).
