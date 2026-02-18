# AlphaForge(AFF)


### Data Preparation
Similar to [AlphaGen](https://github.com/RL-MLDM/alphagen), We Use [Qlib](https://github.com/microsoft/qlib#data-preparation) as data save tool and download data from free & open-source data source  [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5).

Please install Qlib [Qlib](https://github.com/microsoft/qlib) first

Then download stock data through running `data_collection/fetch_baostock_data.py`

The next, Modify the correspoding `/path/for/qlib_data` in `gan.utils.data.py` to the data you downloaded (the dafault setting is `~/.qlib/qlib_data/cn_data_rolling`)


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

#### stage2: Combining alpha factors
```shell
python combine_AFF.py --instruments=csi300 --train_end_year=2020 --seeds=[0,1,2,3,4] --save_name=test --n_factors=10 --window=inf
```
Here `instruments,train_end_year,seeds,save_name`,` must be the same as it in stage 1
- `n_factors` is the num of factors used at each day, it should be less than or equal to `zoo_size` in stage 1.
- `window` is the slicing window that is used to evaluate the alpha factors in order to dynamicly select and cobine.

#### stage3: Show the results

You could run the ipython notebook file 

```shell
exp_AFF_calc_result.ipynb
```

to generate and concat experiment result.


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
