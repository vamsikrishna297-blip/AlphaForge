import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_generic.features import target
from gan.utils.data import get_data_by_year


def main(
    instruments: str = "csi300",
    train_end_year: int = 2020,
    freq: str = "day",
    seeds: str = "[0,1,2,3,4]",
    save_name: str = "test",
    n_factors: int = 10,
    window: str = "inf",
    out_dir: str = "out/results",
):
    seed_list = eval(seeds) if isinstance(seeds, str) else seeds
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_rows = []
    pred_frames = []

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for seed in seed_list:
        returned = get_data_by_year(
            train_start=2010,
            train_end=train_end_year,
            valid_year=train_end_year + 1,
            test_year=train_end_year + 2,
            instruments=instruments,
            target=target,
            freq=freq,
        )
        data_all, data, data_valid, data_valid_withhead, data_test, data_test_withhead, _ = returned

        pred_path = Path(f"out/{save_name}_{instruments}_{train_end_year}_{seed}/pred_{train_end_year}_{n_factors}_{window}_{seed}.pt")
        if not pred_path.exists():
            print(f"[WARN] missing prediction file for seed {seed}: {pred_path}")
            continue

        pred = torch.load(pred_path).to(device)
        tgt = target.evaluate(data_all)

        ones = torch.ones_like(tgt) * torch.nan
        ones[-data_test.n_days:] = pred
        cur_df = data_all.make_dataframe(ones)
        pred_frames.append(cur_df.unstack().iloc[-data_test.n_days:].stack().rename(f"seed_{seed}"))

        tgt = tgt[-data_test.n_days:].to(device)
        ic_s = torch.nan_to_num(batch_pearsonr(pred, tgt), nan=0)
        ric_s = torch.nan_to_num(batch_spearmanr(pred, tgt), nan=0)

        ic_mean = ic_s.mean().item()
        ric_mean = ric_s.mean().item()
        ic_std = ic_s.std().item()
        ric_std = ric_s.std().item()

        result_rows.append(
            {
                "seed": int(seed),
                "n_factors": int(n_factors),
                "ic": ic_mean,
                "ric": ric_mean,
                "icir": ic_mean / ic_std if ic_std != 0 else 0.0,
                "ricir": ric_mean / ric_std if ric_std != 0 else 0.0,
            }
        )

    if not result_rows:
        raise RuntimeError("No stage-3 results were generated. Ensure combine_AFF.py prediction files exist.")

    result_df = pd.DataFrame(result_rows)
    agg_df = result_df.groupby("n_factors").agg(["mean", "std"])

    result_df.to_csv(out_path / f"stage3_seed_metrics_{save_name}_{instruments}_{train_end_year}.csv", index=False)
    agg_df.to_csv(out_path / f"stage3_agg_metrics_{save_name}_{instruments}_{train_end_year}.csv")

    if pred_frames:
        pd.concat(pred_frames, axis=1).to_csv(out_path / f"stage3_predictions_{save_name}_{instruments}_{train_end_year}.csv")

    summary = {
        "save_name": save_name,
        "instruments": instruments,
        "train_end_year": train_end_year,
        "seeds": [int(s) for s in seed_list],
        "n_factors": int(n_factors),
        "window": window,
        "rows": len(result_rows),
    }
    with open(out_path / f"stage3_summary_{save_name}_{instruments}_{train_end_year}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(result_df)
    print(agg_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", default="csi300")
    parser.add_argument("--train_end_year", type=int, default=2020)
    parser.add_argument("--freq", default="day")
    parser.add_argument("--seeds", default="[0,1,2,3,4]")
    parser.add_argument("--save_name", default="test")
    parser.add_argument("--n_factors", type=int, default=10)
    parser.add_argument("--window", default="inf")
    parser.add_argument("--out_dir", default="out/results")
    args = parser.parse_args()
    main(**vars(args))
