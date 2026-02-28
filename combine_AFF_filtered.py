import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from alphagen.utils.correlation import batch_pearsonr, batch_ret, batch_spearmanr
from alphagen_generic.features import *
from alphagen.data.expression import *
from gan.utils import get_blds_list_df, load_pickle
from gan.utils.data import get_data_by_year


def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list = []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i + chunk_size], y[i:i + chunk_size]))
    return torch.cat(spearmanr_list, dim=0)


def get_tensor_metrics_raw(x, y):
    ic_s = torch.nan_to_num(batch_pearsonr(x, y), nan=0)
    ric_s = torch.nan_to_num(chunk_batch_spearmanr(x, y, chunk_size=400), nan=0)
    ret_s = torch.nan_to_num(batch_ret(x, y), nan=0)
    return ic_s, ric_s, ret_s


def _quintile_bucket_returns(pred: torch.Tensor, y_true: torch.Tensor, n_buckets: int = 5) -> dict:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    y_np = y_true.detach().cpu().numpy().reshape(-1)
    valid = np.isfinite(pred_np) & np.isfinite(y_np)
    pred_np, y_np = pred_np[valid], y_np[valid]
    if len(pred_np) < n_buckets:
        out = {f"q{i+1}": 0.0 for i in range(n_buckets)}
        out["q5_q1"] = 0.0
        out["n_assets"] = int(len(pred_np))
        return out
    order = np.argsort(pred_np)
    chunks = np.array_split(y_np[order], n_buckets)
    vals = {f"q{i+1}": float(np.nanmean(ch)) if len(ch) else 0.0 for i, ch in enumerate(chunks)}
    vals["q5_q1"] = vals["q5"] - vals["q1"]
    vals["n_assets"] = int(len(pred_np))
    return vals


def _date_from_eval_index(data_all, eval_idx: int) -> pd.Timestamp:
    if not hasattr(data_all, "_dates"):
        return pd.Timestamp(eval_idx)
    offset = int(getattr(data_all, "max_backtrack_days", 0))
    raw_idx = min(eval_idx + offset, len(data_all._dates) - 1)
    return pd.Timestamp(data_all._dates[raw_idx])


def _load_stock_intervals(stock_list_file: str) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    p = os.path.expanduser(stock_list_file)
    df = pd.read_csv(p, sep='\t', header=None)
    if df.shape[1] < 3:
        raise ValueError(f"stock_list_file must have at least 3 tab-separated columns: code, start, end. got {df.shape[1]}")
    df = df.iloc[:, :3]
    df.columns = ["code", "start", "end"]
    df["code"] = df["code"].astype(str).str.strip().str.lower()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, r in df.iterrows():
        out.setdefault(r["code"], []).append((r["start"], r["end"]))
    return out


def _active_stock_mask(stock_ids, date_ts: pd.Timestamp, intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]], device) -> torch.Tensor:
    keep = []
    for sid in stock_ids:
        code = str(sid).lower()
        active = False
        for s, e in intervals.get(code, []):
            if s <= date_ts <= e:
                active = True
                break
        keep.append(active)
    return torch.tensor(keep, dtype=torch.bool, device=device)


def _window_xy(fct_tensor, tgt_tensor, begin: int, end_exclusive: int, good_idx: list, masks: Dict[int, torch.Tensor]):
    xs, ys = [], []
    for d in range(max(begin, 0), max(end_exclusive, 0)):
        mask = masks.get(d)
        if mask is None or int(mask.sum().item()) == 0:
            continue
        xs.append(fct_tensor[d, mask][:, good_idx])
        ys.append(tgt_tensor[d, mask])
    if not xs:
        return None, None
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y


def main(
    instruments: str = "csi500",
    stock_list_file: str = "",
    train_end_year: int = 2020,
    freq: str = 'day',
    seeds: str = '[0]',
    cuda: int = 0,
    save_name: str = 'test',
    n_factors: int = 10,
    window: Union[int, str] = 'inf',
    sanity_sample_n: int = 5,
    sanity_sample_date: Union[str, None] = None,
):
    if not stock_list_file:
        raise ValueError("--stock_list_file is required (tab-separated code/start/end like instruments/*.txt)")

    if isinstance(seeds, str):
        seeds = eval(seeds)

    if isinstance(window, str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    train_end = train_end_year

    returned = get_data_by_year(
        train_start=2010,
        train_end=train_end,
        valid_year=train_end + 1,
        test_year=train_end + 2,
        instruments=instruments,
        target=target,
        freq=freq,
    )
    data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = returned

    intervals = _load_stock_intervals(stock_list_file)

    for seed in seeds:
        path = f"out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl"
        tensor_save_path = f"out/{save_name}_{instruments}_{train_end}_{seed}/"
        name = f"{train_end}_{n_factors}_{window}_{seed}_filtered"
        zoo = load_pickle(path)

        df = get_blds_list_df([zoo]).sort_values('score', ascending=False, key=lambda x: abs(x))
        expr_col = "exprs_str" if "exprs_str" in df.columns else "exprs"

        from gan.utils.builder import exprs2tensor
        fct_tensor = exprs2tensor(df['exprs'], data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)
        vwap_t1_tensor = exprs2tensor([Ref(vwap, -1)], data_all, normalize=False)
        vwap_t21_tensor = exprs2tensor([Ref(vwap, -21)], data_all, normalize=False)

        ic_list, ric_list, ret_list = [], [], []
        from tqdm import tqdm
        for cur in tqdm(range(fct_tensor.shape[-1])):
            ic_s, ric_s, ret_s = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
            ic_list.append(ic_s)
            ric_list.append(ric_s)
            ret_list.append(ret_s)

        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        ret_s = torch.stack(ret_list, dim=-1)

        shift = 21
        pred_list, ics_list, rics_list = [], [], []
        sanity_rows, detailed_samples = [], []

        eval_begin = len(fct_tensor) - data_test.n_days - data_valid.n_days
        eval_end = len(fct_tensor)

        masks = {
            d: _active_stock_mask(data_all._stock_ids, _date_from_eval_index(data_all, d), intervals, fct_tensor.device)
            for d in range(eval_begin, eval_end)
        }

        pbar = tqdm(range(eval_begin, eval_end))
        for cur in pbar:
            begin = cur - window - shift if np.isfinite(window) else 0
            cur_ic = ic_s[begin:cur - shift]
            cur_ric = ric_s[begin:cur - shift]
            cur_ret = ret_s[begin:cur - shift]

            metrics = dict(
                ic=cur_ic.mean(dim=0).detach().cpu().numpy(),
                ic_std=cur_ic.std(dim=0).detach().cpu().numpy(),
                icir=(cur_ic.mean(dim=0) / cur_ic.std(dim=0)).detach().cpu().numpy(),
                ric=cur_ric.mean(dim=0).detach().cpu().numpy(),
                ric_std=cur_ric.std(dim=0).detach().cpu().numpy(),
                ricir=(cur_ric.mean(dim=0) / cur_ric.std(dim=0)).detach().cpu().numpy(),
                ret=cur_ret.mean(dim=0).detach().cpu().numpy(),
                ret_std=cur_ret.std(dim=0).detach().cpu().numpy(),
                retir=(cur_ret.mean(dim=0) / cur_ret.std(dim=0)).detach().cpu().numpy(),
            )
            tmp = pd.DataFrame(metrics).sort_values('ricir', ascending=False, key=lambda x: abs(x))
            selected = tmp[(tmp['ric'] > 0.02) & (tmp['ricir'] > 0.2)]
            if len(selected) < 1:
                selected = tmp.iloc[:1]
            good_idx = selected.iloc[:n_factors].index.to_list()

            x, y = _window_xy(fct_tensor, tgt_tensor, begin, cur - shift, good_idx, masks)
            mask_cur = masks[cur]
            pred_full = torch.full((tgt_tensor.shape[1],), float('nan'), device=fct_tensor.device)

            day_date = str(_date_from_eval_index(data_all, cur))
            if x is None or int(mask_cur.sum().item()) == 0:
                step_ic = np.nan
                step_ric = np.nan
                qret = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0, "q5": 0.0, "q5_q1": 0.0, "n_assets": 0}
            else:
                to_pred = torch.nan_to_num(fct_tensor[cur, mask_cur][:, good_idx], nan=0)
                y_true = tgt_tensor[cur, mask_cur]
                y = y.reshape(-1, y.shape[-1])
                x = x.reshape(-1, x.shape[-1])
                valid = torch.isfinite(y)[:, 0]
                y = y[valid]
                x = x[valid]
                ones = torch.ones_like(x[..., 0:1])
                x = torch.cat([x, ones], dim=-1)
                ones = torch.ones_like(to_pred[..., 0:1])
                to_pred = torch.cat([to_pred, ones], dim=-1)
                coef = torch.linalg.lstsq(x, y).solution
                pred = to_pred @ coef
                pred_full[mask_cur] = pred[:, 0]

                step_ic = float(batch_pearsonr(pred.T, y_true.T)[0].detach().cpu().item())
                step_ric = float(batch_spearmanr(pred.T, y_true.T)[0].detach().cpu().item())
                qret = _quintile_bucket_returns(pred, y_true, n_buckets=5)
                ics_list.append(step_ic)
                rics_list.append(step_ric)

                capture_by_n = sanity_sample_n > 0 and cur >= eval_end - sanity_sample_n
                capture_by_date = sanity_sample_date is not None and str(day_date).startswith(str(sanity_sample_date))
                if capture_by_n or capture_by_date:
                    pred_preview = pred[:5, 0].detach().cpu().numpy().tolist()
                    tgt_preview = y_true[:5, 0].detach().cpu().numpy().tolist()
                    vwap_t1_preview = vwap_t1_tensor[cur, mask_cur][:5, 0].detach().cpu().numpy().tolist()
                    vwap_t21_preview = vwap_t21_tensor[cur, mask_cur][:5, 0].detach().cpu().numpy().tolist()
                    sid = np.array(data_all._stock_ids)[mask_cur.detach().cpu().numpy()][:5]
                    stock_preview = [{
                        "stock": str(sid[i]),
                        "prediction": float(pred_preview[i]) if np.isfinite(pred_preview[i]) else None,
                        "target": float(tgt_preview[i]) if np.isfinite(tgt_preview[i]) else None,
                        "vwap_t1": float(vwap_t1_preview[i]) if np.isfinite(vwap_t1_preview[i]) else None,
                        "vwap_t21": float(vwap_t21_preview[i]) if np.isfinite(vwap_t21_preview[i]) else None,
                    } for i in range(len(pred_preview))]
                    detailed_samples.append({
                        "day_index": int(cur),
                        "date": day_date,
                        "selected_factors": int(len(good_idx)),
                        "prediction_preview": pred_preview,
                        "target_preview": tgt_preview,
                        "stock_preview": stock_preview,
                        "selected_details": [
                            {"factor_idx": int(idx), "expr": str(df.loc[idx][expr_col])}
                            for idx in good_idx
                        ],
                    })

            top_formulas = [str(df.loc[idx][expr_col]) for idx in good_idx[:10]]
            step_row = {
                "day_index": int(cur),
                "date": day_date,
                "selected_factors": int(len(good_idx)),
                "step_ic": float(step_ic) if np.isfinite(step_ic) else None,
                "step_ric": float(step_ric) if np.isfinite(step_ric) else None,
                "running_ic_mean": float(np.nanmean(ics_list)) if len(ics_list) else None,
                "running_ric_mean": float(np.nanmean(rics_list)) if len(rics_list) else None,
                "q1_ret": qret["q1"],
                "q2_ret": qret["q2"],
                "q3_ret": qret["q3"],
                "q4_ret": qret["q4"],
                "q5_ret": qret["q5"],
                "q5_q1_ret": qret["q5_q1"],
                "n_assets": qret["n_assets"],
            }
            for i in range(10):
                step_row[f"top_formula_{i + 1}"] = top_formulas[i] if i < len(top_formulas) else ""
            sanity_rows.append(step_row)
            pred_list.append(pred_full)

        num_1 = data_valid.n_days
        num_2 = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        torch.save(all_pred[-num_2 - num_1:-num_1].detach().cpu(), f"{tensor_save_path}/pred_valid_{name}.pt")
        torch.save(all_pred[-data_test.n_days:].detach().cpu(), f"{tensor_save_path}/pred_{name}.pt")

        sanity_df = pd.DataFrame(sanity_rows)
        sanity_df.to_csv(f"{tensor_save_path}/combine_sanity_{name}.csv", index=False)
        sanity_df[["day_index", "date", "q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret", "q5_q1_ret", "n_assets"]].to_csv(
            f"{tensor_save_path}/combine_quintiles_{name}.csv", index=False
        )
        with open(f"{tensor_save_path}/combine_summary_{name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "seed": int(seed),
                "stock_list_file": stock_list_file,
                "n_steps": int(len(sanity_rows)),
                "final_running_ic": float(np.nanmean(ics_list)) if len(ics_list) else 0.0,
                "final_running_ric": float(np.nanmean(rics_list)) if len(rics_list) else 0.0,
            }, f, indent=2)
        with open(f"{tensor_save_path}/combine_samples_{name}.json", "w", encoding="utf-8") as f:
            json.dump(detailed_samples, f, indent=2)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
