from typing import List, Union, Optional, Tuple, Dict
import os
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    
def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        else:
            result.append(feature)
    return result

def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open','$close','$high','$low','$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        else:
            raise ValueError(f"feature {feature} not supported")
    return result



def _normalize_qlib_region(region: str) -> str:
    normalized = str(region).strip().lower()
    # qlib only supports built-in region keys (cn/us/tw).
    # For custom dumped datasets (e.g. NSE), use CN config for expression/parsing defaults.
    if normalized in {"in", "india", "nse"}:
        return "cn"
    return normalized or "cn"
class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0'),
                 raw:bool = False,
                 qlib_path:Union[str,Dict] = "",
                 freq:str = 'day',
                 ) -> None:
        self._init_qlib(qlib_path)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.data, self._dates, self._stock_ids = self._get_data()


    @classmethod
    def _init_qlib(cls, qlib_path) -> None:
        if cls._qlib_initialized:
            return
        import qlib

        region = _normalize_qlib_region(os.environ.get("QLIB_REGION", "cn"))
        qlib.init(provider_uri=qlib_path, region=region)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar(freq=self.freq)
        if len(cal) == 0:
            raise ValueError(f"Qlib calendar is empty for freq='{self.freq}'.")

        start_ts = pd.Timestamp(self._start_time)
        end_ts = pd.Timestamp(self._end_time)

        start_index = int(cal.searchsorted(start_ts))  # type: ignore
        end_index = int(cal.searchsorted(end_ts))  # type: ignore

        # Clamp to valid calendar range to avoid negative / overflow indexing.
        start_index = min(max(start_index, 0), len(cal) - 1)
        end_index = min(max(end_index, 0), len(cal) - 1)

        if cal[end_index] > end_ts and end_index > 0:
            end_index -= 1

        real_start_idx = max(start_index - self.max_backtrack_days, 0)
        real_end_idx = min(end_index + self.max_future_days, len(cal) - 1)

        real_start_time = cal[real_start_idx]
        real_end_time = cal[real_end_idx]

        if real_start_time > real_end_time:
            raise ValueError(
                "Invalid calendar window after indexing. "
                f"start={self._start_time}, end={self._end_time}, "
                f"real_start={real_start_time}, real_end={real_end_time}, "
                f"calendar_range=({cal[0]}, {cal[-1]})."
            )

        result = (QlibDataLoader(config=exprs, freq=self.freq)  # type: ignore
                  .load(self._instrument, real_start_time, real_end_time))
        return result
    
    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        if self.raw and self.freq == 'day':
            features = change_to_raw(features)
        elif self.raw:
            features = change_to_raw_min(features)
        df = self._load_exprs(features)
        self.df_bak = df
        if df is None or df.empty:
            raise ValueError(
                "No data returned from qlib. Check instruments/date range and qlib_path "
                f"(instrument={self._instrument}, start={self._start_time}, "
                f"end={self._end_time}, freq={self.freq})."
            )
        # print(df)
        df = df.stack().unstack(level=1)
        if df.empty or len(df.columns) == 0:
            raise ValueError(
                "Qlib returned an empty dataframe after stacking; ensure instrument universe "
                "(e.g. csi300/csi500/all) exists in your qlib dataset and includes the "
                "requested date range."
            )
        dates = df.index.levels[0]                                      # type: ignore
        stock_ids = df.columns
        values = df.values
        values = values.reshape((-1, len(features), values.shape[-1]))  # type: ignore
        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def add_data(self,data:torch.Tensor,dates:pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data,data],dim=0)
        self._dates = pd.Index(self._dates.append(dates))


    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
    
    