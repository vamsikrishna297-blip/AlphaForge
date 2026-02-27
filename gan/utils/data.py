from alphagen_generic.features import open_
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *

import os
from pathlib import Path


def _has_qlib_structure(root: str) -> bool:
    base = Path(root)
    return (
        (base / "calendars" / "day.txt").exists()
        and (base / "features").exists()
        and (base / "instruments").exists()
    )


def _resolve_qlib_path(freq: str, instruments: str | None = None):
    env_key = f"QLIB_PATH_{freq.upper()}"

    candidates = []
    env_value = os.environ.get(env_key)
    if env_value:
        candidates.append(os.path.expanduser(env_value))

    # Backward compatible: keep supporting the repo-local placeholder path when present.
    # Keep this after env var so custom datasets are preferred.
    candidates.append(os.path.expanduser("path/for/qlib"))

    # Keep the default consistent with README and data_collection script outputs.
    candidates.append(os.path.expanduser("~/.qlib/qlib_data/cn_data_rolling"))

    missing_layout = []
    missing_instruments = []

    for resolved in candidates:
        if not os.path.exists(resolved):
            continue
        if not _has_qlib_structure(resolved):
            missing_layout.append(resolved)
            continue

        if instruments:
            inst_file = Path(resolved) / "instruments" / f"{instruments}.txt"
            if not inst_file.exists():
                missing_instruments.append(str(inst_file))
                continue

        return {freq: resolved}

    raise FileNotFoundError(
        f"No usable Qlib dataset found for freq='{freq}' and instruments='{instruments}'. "
        f"Tried paths: {candidates}. "
        f"Missing qlib layout at: {missing_layout or '[]'}. "
        f"Missing instrument file(s): {missing_instruments or '[]'}. "
        f"Set {env_key} to a valid qlib root (with calendars/features/instruments)."
    )


def get_data_by_year(
    train_start = 2010,train_end=2019,valid_year=2020,test_year =2021,
    instruments=None, target=None,freq=None,
                    ):
    QLIB_PATH = _resolve_qlib_path(freq, instruments=instruments)
    
    from gan.utils import load_pickle,save_pickle
    # from gan.utils.qlib import get_data_my
    get_data_my = StockData

    train_dates=(f"{train_start}-01-01", f"{train_end}-12-31")
    val_dates=(f"{valid_year}-01-01", f"{valid_year}-12-31")
    test_dates=(f"{test_year}-01-01", f"{test_year}-12-31")

    train_start,train_end = train_dates
    valid_start,valid_end = val_dates
    valid_head_start = f"{valid_year-2}-01-01"
    test_start,test_end = test_dates
    test_head_start = f"{test_year-2}-01-01"

    name = instruments + '_pkl_' + str(target).replace('/','_').replace(' ','') + '_' + freq
    name = f"{name}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"
    try:

        data = load_pickle(f'pkl/{name}/data.pkl')
        data_valid = load_pickle(f'pkl/{name}/data_valid.pkl')
        data_valid_withhead = load_pickle(f'pkl/{name}/data_valid_withhead.pkl')
        data_test = load_pickle(f'pkl/{name}/data_test.pkl')
        data_test_withhead = load_pickle(f'pkl/{name}/data_test_withhead.pkl')

    except:
        print('Data not exist, load from qlib')
        data = get_data_my(instruments, train_start, train_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_valid = get_data_my(instruments, valid_start, valid_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_valid_withhead = get_data_my(instruments,valid_head_start, valid_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_test = get_data_my(instruments, test_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        data_test_withhead = get_data_my(instruments, test_head_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)

        os.makedirs(f"pkl/{name}",exist_ok=True)
        save_pickle(data,f'pkl/{name}/data.pkl')
        save_pickle(data_valid,f'pkl/{name}/data_valid.pkl')
        save_pickle(data_valid_withhead,f'pkl/{name}/data_valid_withhead.pkl')
        save_pickle(data_test,f'pkl/{name}/data_test.pkl')
        save_pickle(data_test_withhead,f'pkl/{name}/data_test_withhead.pkl')
    
    try:
        data_all = load_pickle(f'pkl/{name}/data_all.pkl')
    except:
        data_all = get_data_my(instruments, train_start, test_end,raw = True,qlib_path = QLIB_PATH,freq=freq)
        save_pickle(data_all,f'pkl/{name}/data_all.pkl')
    return data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name
