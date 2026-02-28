import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from alphagen.utils.correlation import batch_pearsonr, batch_ret, batch_spearmanr
from alphagen_generic.features import *
from alphagen.data.expression import *
from gan.utils import get_blds_list_df, load_pickle
from gan.utils.data import get_data_by_year


def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    weights = raw['weights']
    return exprs, weights


def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
        return load_alpha_pool(raw)


def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list = []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i + chunk_size], y[i:i + chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list


def get_tensor_metrics_raw(x, y):
    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0)
    ric_s = torch.nan_to_num(ric_s, nan=0)
    ret_s = torch.nan_to_num(ret_s, nan=0)

    return ic_s, ric_s, ret_s


def _quintile_bucket_returns(pred: torch.Tensor, y_true: torch.Tensor, n_buckets: int = 5) -> dict:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    y_np = y_true.detach().cpu().numpy().reshape(-1)

    valid_mask = np.isfinite(pred_np) & np.isfinite(y_np)
    pred_np = pred_np[valid_mask]
    y_np = y_np[valid_mask]

    if len(pred_np) < n_buckets:
        out = {f"q{i+1}": 0.0 for i in range(n_buckets)}
        out["q5_q1"] = 0.0
        out["n_assets"] = int(len(pred_np))
        return out

    order = np.argsort(pred_np)
    y_sorted = y_np[order]
    chunks = np.array_split(y_sorted, n_buckets)

    vals = {}
    for i, ch in enumerate(chunks, start=1):
        vals[f"q{i}"] = float(np.nanmean(ch)) if len(ch) else 0.0

    vals["q5_q1"] = vals["q5"] - vals["q1"]
    vals["n_assets"] = int(len(pred_np))
    return vals


def main(
    instruments: str = "csi500",
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
    if isinstance(seeds, str):
        seeds = eval(seeds)
    assert isinstance(seeds, list)

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

    for seed in seeds:
        path = f"out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl"
        tensor_save_path = f"out/{save_name}_{instruments}_{train_end}_{seed}/"
        name = f"{train_end}_{n_factors}_{window}_{seed}"
        zoo = load_pickle(path)

        df = get_blds_list_df([zoo]).sort_values('score', ascending=False, key=lambda x: abs(x))
        expr_col = "exprs_str" if "exprs_str" in df.columns else "exprs"

        from gan.utils.builder import exprs2tensor
        fct_tensor = exprs2tensor(df['exprs'], data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)
        vwap_t1_tensor = exprs2tensor([Ref(vwap, -1)], data_all, normalize=False)
        vwap_t21_tensor = exprs2tensor([Ref(vwap, -21)], data_all, normalize=False)

        ic_list = []
        ric_list = []
        ret_list = []

        from tqdm import tqdm
        for cur in tqdm(range(fct_tensor.shape[-1])):
            ic_s, ric_s, ret_s = get_tensor_metrics_raw(fct_tensor[..., cur], tgt_tensor[..., 0])
            ic_list.append(ic_s)
            ric_list.append(ric_s)
            ret_list.append(ret_s)

        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        ret_s = torch.stack(ret_list, dim=-1)
        torch.cuda.empty_cache()

        shift = 21

        pred_list = []
        ics_list = []
        rics_list = []
        sanity_rows = []
        detailed_samples = []

        eval_begin = len(fct_tensor) - data_test.n_days - data_valid.n_days
        eval_end = len(fct_tensor)
        pbar = tqdm(range(eval_begin, eval_end))
        for cur in pbar:
            begin = cur - window - shift if np.isfinite(window) else 0

            cur_ic = ic_s[begin:cur - shift]
            cur_ric = ric_s[begin:cur - shift]
            cur_ret = ret_s[begin:cur - shift]

            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)
            ret_mean = cur_ret.mean(dim=0)
            ret_std = cur_ret.std(dim=0)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            retir = ret_mean / ret_std

            metrics = dict(
                ic=ic_mean.detach().cpu().numpy(),
                ic_std=ic_std.detach().cpu().numpy(),
                icir=icir.detach().cpu().numpy(),
                ric=ric_mean.detach().cpu().numpy(),
                ric_std=ric_std.detach().cpu().numpy(),
                ricir=ricir.detach().cpu().numpy(),
                ret=ret_mean.detach().cpu().numpy(),
                ret_std=ret_std.detach().cpu().numpy(),
                retir=retir.detach().cpu().numpy(),
            )
            tmp = pd.DataFrame(metrics).sort_values('ricir', ascending=False, key=lambda x: abs(x))

            selected = tmp[(tmp['ric'] > 0.02) & (tmp['ricir'] > 0.2)]
            if len(selected) < 1:
                selected = tmp.iloc[:1]

            good_idx = selected.iloc[:n_factors].index.to_list()

            x = fct_tensor[begin:cur - shift, :, good_idx]
            y = tgt_tensor[begin:cur - shift]
            to_pred = fct_tensor[cur, :, good_idx]
            y_true = tgt_tensor[cur]

            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])

            to_select = torch.isfinite(y)[:, 0]
            y = y[to_select]
            x = x[to_select]

            to_pred = torch.nan_to_num(to_pred, nan=0)

            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones], dim=-1)

            coef = torch.linalg.lstsq(x, y).solution
            pred = to_pred @ coef

            cur_ic = batch_pearsonr(pred.T, y_true.T)[0]
            cur_ric = batch_spearmanr(pred.T, y_true.T)[0]
            ics_list.append(cur_ic.detach().cpu().numpy())
            rics_list.append(cur_ric.detach().cpu().numpy())

            pbar.set_description(f"ic:{np.nanmean(ics_list):.3f} ric:{np.nanmean(rics_list):.3f} n:{len(good_idx)}")

            day_date = str(data_all._dates[cur]) if hasattr(data_all, "_dates") else str(cur)
            qret = _quintile_bucket_returns(pred, y_true, n_buckets=5)
            y_true_1d = y_true[:, 0]
            finite_mask = torch.isfinite(y_true_1d)
            if finite_mask.any():
                index_ret_ew = float(y_true_1d[finite_mask].mean().detach().cpu().item())
                cs_dispersion = float(y_true_1d[finite_mask].std().detach().cpu().item())
            else:
                index_ret_ew = 0.0
                cs_dispersion = 0.0

            top_formulas = [str(df.loc[idx][expr_col]) for idx in good_idx[:10]]

            step_row = {
                "day_index": int(cur),
                "date": day_date,
                "selected_factors": int(len(good_idx)),
                "step_ic": float(cur_ic.detach().cpu().item()),
                "step_ric": float(cur_ric.detach().cpu().item()),
                "running_ic_mean": float(np.nanmean(ics_list)),
                "running_ric_mean": float(np.nanmean(rics_list)),
                "q1_ret": qret["q1"],
                "q2_ret": qret["q2"],
                "q3_ret": qret["q3"],
                "q4_ret": qret["q4"],
                "q5_ret": qret["q5"],
                "q5_q1_ret": qret["q5_q1"],
                "n_assets": qret["n_assets"],
                "equal_weighted_index_ret": index_ret_ew,
                "cs_return_dispersion": cs_dispersion,
            }
            for i in range(10):
                step_row[f"top_formula_{i + 1}"] = top_formulas[i] if i < len(top_formulas) else ""
            sanity_rows.append(step_row)

            capture_by_n = sanity_sample_n > 0 and cur >= eval_end - sanity_sample_n
            capture_by_date = sanity_sample_date is not None and str(day_date).startswith(str(sanity_sample_date))
            if capture_by_n or capture_by_date:
                coef_np = coef.detach().cpu().numpy().reshape(-1)
                selected_detail = []
                for local_i, idx in enumerate(good_idx):
                    selected_detail.append({
                        "factor_idx": int(idx),
                        "expr": str(df.loc[idx][expr_col]),
                        "coef": float(coef_np[local_i]),
                        "hist_ic": float(tmp.loc[idx]["ic"]),
                        "hist_icir": float(tmp.loc[idx]["icir"]),
                        "hist_ric": float(tmp.loc[idx]["ric"]),
                        "hist_ricir": float(tmp.loc[idx]["ricir"]),
                    })

                pred_preview = pred[:5, 0].detach().cpu().numpy().tolist()
                tgt_preview = y_true[:5, 0].detach().cpu().numpy().tolist()
                vwap_t1_preview = vwap_t1_tensor[cur, :5, 0].detach().cpu().numpy().tolist()
                vwap_t21_preview = vwap_t21_tensor[cur, :5, 0].detach().cpu().numpy().tolist()
                stock_ids = list(data_all._stock_ids[:5]) if hasattr(data_all, "_stock_ids") else list(range(len(pred_preview)))
                stock_preview = [
                    {
                        "stock": str(stock_ids[i]),
                        "prediction": float(pred_preview[i]) if np.isfinite(pred_preview[i]) else None,
                        "target": float(tgt_preview[i]) if np.isfinite(tgt_preview[i]) else None,
                        "vwap_t1": float(vwap_t1_preview[i]) if np.isfinite(vwap_t1_preview[i]) else None,
                        "vwap_t21": float(vwap_t21_preview[i]) if np.isfinite(vwap_t21_preview[i]) else None,
                    }
                    for i in range(len(pred_preview))
                ]

                detailed_samples.append({
                    **step_row,
                    "intercept": float(coef_np[-1]),
                    "prediction_preview": pred_preview,
                    "target_preview": tgt_preview,
                    "stock_preview": stock_preview,
                    "vwap_t1_preview": vwap_t1_preview,
                    "vwap_t21_preview": vwap_t21_preview,
                    "quintile_returns": qret,
                    "selected_details": selected_detail,
                })

            pred_list.append(pred[:, 0])

        num_1 = data_valid.n_days
        num_2 = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        all_pred = all_pred[-num_2 - num_1:-num_1]
        torch.save(all_pred.detach().cpu(), f"{tensor_save_path}/pred_valid_{name}.pt")

        num_ = data_test.n_days
        all_pred = torch.stack(pred_list, dim=0)
        all_pred = all_pred[-num_:]
        torch.save(all_pred.detach().cpu(), f"{tensor_save_path}/pred_{name}.pt")

        sanity_df = pd.DataFrame(sanity_rows)
        sanity_df.to_csv(f"{tensor_save_path}/combine_sanity_{name}.csv", index=False)
        sanity_df[["day_index", "date", "q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret", "q5_q1_ret", "n_assets"]].to_csv(
            f"{tensor_save_path}/combine_quintiles_{name}.csv", index=False
        )

        with open(f"{tensor_save_path}/combine_summary_{name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "seed": int(seed),
                "n_factors_requested": int(n_factors),
                "n_steps": int(len(sanity_rows)),
                "sanity_sample_n": int(sanity_sample_n),
                "sanity_sample_date": sanity_sample_date,
                "weighting": "linear_regression",
                "final_running_ic": float(np.nanmean(ics_list)) if len(ics_list) else 0.0,
                "final_running_ric": float(np.nanmean(rics_list)) if len(rics_list) else 0.0,
                "avg_q1_ret": float(sanity_df["q1_ret"].mean()) if len(sanity_df) else 0.0,
                "avg_q5_ret": float(sanity_df["q5_ret"].mean()) if len(sanity_df) else 0.0,
                "avg_q5_q1_ret": float(sanity_df["q5_q1_ret"].mean()) if len(sanity_df) else 0.0,
                "avg_equal_weighted_index_ret": float(sanity_df["equal_weighted_index_ret"].mean()) if len(sanity_df) else 0.0,
                "avg_cs_return_dispersion": float(sanity_df["cs_return_dispersion"].mean()) if len(sanity_df) else 0.0,
            }, f, indent=2)

        with open(f"{tensor_save_path}/combine_samples_{name}.json", "w", encoding="utf-8") as f:
            json.dump(detailed_samples, f, indent=2)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
