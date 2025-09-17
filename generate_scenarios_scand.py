#!/usr/bin/env python3
"""
Select "best" negative samples from SCAND metadata with simple fitness scoring.

Workflow
--------
1) Load a CSV like <save_folder>/scand_<split>.csv.
2) For each row, compute:
   - y_dev / x_dev from the label_waypoints column (stringified arrays)
   - proximity statistics by reading localized scene JSON derived from config
3) Score each row with a weighted linear combination ("fitness").
4) Within each sequence, keep only the max fitness in each sliding index range
   (e.g., 0–24, 25–49, ...), to encourage temporal diversity.
5) Save the top-N rows by fitness to <save_folder>/<output_subdir>/best_samples_<split>.csv.

Notes
-----
- Global-ish paths like the split, CSV name pattern, output subdir, and range size
  are now configurable via CLI args (and still overridable via the YAML).
- Uses tqdm progress bars. Safe directory creation/removal.
"""

import argparse
import json
import os
import re
import shutil
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import tqdm

from finetune_utils import get_autolabel_base_filepath


# --------------------------- Utility helpers ---------------------------

def str_to_npy(input_str: str) -> np.ndarray:
    """
    Convert a stringified numpy array (possibly with nested 'array([...])' text)
    into a numeric numpy array. Returns shape:
      - (N, 2) if one subarray
      - (K, N, 2) if multiple subarrays are embedded.
    """
    n_subarrays = input_str.count('array') if 'array' in input_str else 1
    cleaned = input_str.replace('array', '\n')
    # match decimals or scientific notation
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    nums = [float(x) for x in numbers]
    num_columns = 2
    if n_subarrays == 1:
        return np.array(nums).reshape(-1, num_columns)
    return np.array(nums).reshape(n_subarrays, -1, num_columns)


def compute_deviation(row: pd.Series) -> Tuple[float, float]:
    """Return (x_dev, y_dev) from the 'label_waypoints' column."""
    wps = str_to_npy(row['label_waypoints'])
    # flatten if needed
    wps2d = wps if wps.ndim == 2 else wps.reshape(-1, wps.shape[-1])
    x_dev = float(wps2d[:, 0].max() - wps2d[:, 0].min())
    y_dev = float(wps2d[:, 1].max() - wps2d[:, 1].min())
    return x_dev, y_dev


def get_num_close_and_very_close_people(row: pd.Series) -> Tuple[float, float, float, float, int]:
    """
    Read localized scene objects and compute proximity stats in robot frame.
    Returns:
      avg_n_close_people, avg_n_very_close_people, avg_n_close_and_consistent_people,
      avg_n_total_people, total_unique_people
    """
    with open(row['dataset_cfg_fp'], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    localized_objs_dir = config['lart_localized_scene_objects']
    imagepath = row['filepath']
    localized_info_fp = get_autolabel_base_filepath(imagepath, localized_objs_dir, config) + '_localized_obj_coords.json'

    with open(localized_info_fp, 'r') as f:
        localized_info = json.load(f)

    track_id_to_object_trajs = localized_info.get('track_id_to_object_trajectory', {})
    total_n_very_close_people = 0
    total_n_close_people = 0
    total_n_people = 0
    total_n_close_and_consistent_people = 0  # appears in at least 2 consecutive timesteps
    last_appeared = {}                       # track id -> last timestep seen
    total_unique_people = 0

    for track_id, object_history in track_id_to_object_trajs.items():
        total_unique_people += 1
        for timestep, obs in enumerate(object_history):
            total_n_people += 1
            y, x = float(obs['center_y']), float(obs['center_x'])
            if -5 <= y <= 5 and 0 <= x <= 10:
                total_n_close_people += 1
                if track_id in last_appeared and last_appeared[track_id] == timestep - 1:
                    total_n_close_and_consistent_people += 1
                last_appeared[track_id] = timestep
            if -3 <= y <= 3 and 0 <= x <= 5:
                total_n_very_close_people += 1

    # assume 10 timesteps per clip in this metadata
    denom = 10 if 10 > 0 else 1
    avg_total = total_n_people / denom
    avg_very_close = total_n_very_close_people / denom
    avg_close = total_n_close_people / denom
    avg_close_consistent = total_n_close_and_consistent_people / denom
    return avg_close, avg_very_close, avg_close_consistent, avg_total, total_unique_people


def keep_largest_fitness_in_range(df: pd.DataFrame, range_size: int) -> pd.DataFrame:
    """
    Within each (sequence_name, range_group), keep only the max fitness row;
    others get fitness = -inf to be filtered out by nlargest().
    """
    tmp = df.copy()
    tmp['range_group'] = tmp['idx_in_sequence'] // max(1, range_size)
    tmp['max_in_group'] = tmp.groupby(['sequence_name', 'range_group'])['fitness'].transform('max')
    tmp['fitness'] = tmp.apply(
        lambda x: x['fitness'] if x['fitness'] == x['max_in_group'] else float('-inf'),
        axis=1
    )
    return tmp.drop(columns=['range_group', 'max_in_group'])


# --------------------------- Core logic ---------------------------

def negative_sample(
    save_folder: str,
    *,
    split: str = "val",
    csv_pattern: str = "scand_{split}.csv",
    output_subdir: str = "negative_samples",
    n_negatives: int = 0,
    range_size: int = 25,
    # fitness weights
    w_dev_y: float = 3.0,
    w_close: float = 0.15,
    w_very_close: float = 0.5,
    w_close_consistent: float = 0.75,
    w_people_penalty_per_over6: float = -2.0,
    filter_out_jackal: bool = True,
) -> pd.DataFrame:
    """
    Compute fitness per frame and save top-K results.

    Returns the dataframe of selected rows (also written to CSV).
    """
    csv_fp = os.path.join(save_folder, csv_pattern.format(split=split))
    assert os.path.exists(csv_fp), f"Input CSV not found: {csv_fp}"

    df = pd.read_csv(csv_fp)

    if filter_out_jackal and 'filepath' in df.columns:
        df = df[~df['filepath'].str.contains('Jackal')]

    # Compute deviations
    tqdm.tqdm.write("Computing waypoint deviations...")
    devs = df.progress_apply(compute_deviation, axis=1)
    df['x_dev'], df['y_dev'] = zip(*devs)

    # Proximity stats (requires dataset_cfg_fp and filepath)
    required_cols = {'dataset_cfg_fp', 'filepath'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for proximity stats: {missing}")

    tqdm.tqdm.write("Computing proximity statistics...")
    prox = df.progress_apply(get_num_close_and_very_close_people, axis=1)
    (df['avg_n_close_people'],
     df['avg_n_very_close_people'],
     df['avg_n_close_consistent_people'],
     df['avg_n_total_people'],
     df['n_unique_ppl']) = zip(*prox)

    # Fitness score
    def compute_fitness(row: pd.Series) -> float:
        penalty = w_people_penalty_per_over6 * max(0.0, row['avg_n_total_people'] - 6.0)
        return (
            w_dev_y * row['y_dev']
            + w_close * row['avg_n_close_people']
            + w_very_close * row['avg_n_very_close_people']
            + w_close_consistent * row['avg_n_close_consistent_people']
            + penalty
        )

    tqdm.tqdm.write("Scoring samples...")
    df['fitness'] = df.progress_apply(compute_fitness, axis=1)

    # Sequence + index metadata
    def get_sequence_name(fp: str) -> str:
        return fp.split('/')[-2]

    def get_idx(fp: str) -> int:
        stem = fp.split('/')[-1].split('.')[0]
        return int(stem)

    df['sequence_name'] = df['filepath'].apply(get_sequence_name)
    df['idx_in_sequence'] = df['filepath'].apply(get_idx)

    # Keep only the max per sliding window
    df = keep_largest_fitness_in_range(df, range_size=range_size)

    # Output dir
    out_dir = os.path.join(save_folder, output_subdir)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Select top-K (if n_negatives==0, just save the whole scored df)
    if n_negatives and n_negatives > 0:
        selected = df.nlargest(n_negatives, 'fitness')
    else:
        selected = df.sort_values('fitness', ascending=False)

    out_fp = os.path.join(out_dir, f'best_samples_{split}.csv')
    selected.to_csv(out_fp, index=False)
    tqdm.tqdm.write(f"Saved {len(selected)} rows → {out_fp}")
    return selected


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--finetune_config', type=str, required=True,
                   help='Path to YAML config file (seeds, defaults).')
    # Overrides / extras (prefer args over config if both provided)
    p.add_argument('--save_folder', type=str, default=None,
                   help='Folder containing the CSV and where outputs are written. '
                        'Overrides config.eval_filepath.')
    p.add_argument('--split', type=str, default='val',
                   help='Data split name (used in file pattern). Default: val')
    p.add_argument('--csv_pattern', type=str, default='scand_{split}.csv',
                   help='CSV filename pattern relative to save_folder. Default: scand_{split}.csv')
    p.add_argument('--output_subdir', type=str, default='negative_samples',
                   help='Subdirectory under save_folder to write results.')
    p.add_argument('--n_negatives', type=int, default=None,
                   help='How many top rows to keep. If omitted, uses config.n_negatives.')
    p.add_argument('--range_size', type=int, default=25,
                   help='Index window size per sequence for max-fitness filtering.')
    # Weights
    p.add_argument('--w_dev_y', type=float, default=3.0)
    p.add_argument('--w_close', type=float, default=0.15)
    p.add_argument('--w_very_close', type=float, default=0.5)
    p.add_argument('--w_close_consistent', type=float, default=0.75)
    p.add_argument('--w_people_penalty_per_over6', type=float, default=-2.0)
    # Misc
    p.add_argument('--no_filter_jackal', action='store_true',
                   help='If set, do NOT filter out rows with "Jackal" in filepath.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tqdm.tqdm.pandas()

    # Load config
    with open(args.finetune_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Reproducibility
    random_seed = int(config.get('random_seed', 0))
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Resolve paths / numbers with CLI override precedence
    save_folder = args.save_folder or config.get('eval_filepath')
    if not save_folder:
        raise ValueError("save_folder is required (via --save_folder or config.eval_filepath).")
    n_negatives = args.n_negatives if args.n_negatives is not None else int(config.get('n_negatives', 0))

    negative_sample(
        save_folder=save_folder,
        split=args.split,
        csv_pattern=args.csv_pattern,
        output_subdir=args.output_subdir,
        n_negatives=n_negatives,
        range_size=args.range_size,
        w_dev_y=args.w_dev_y,
        w_close=args.w_close,
        w_very_close=args.w_very_close,
        w_close_consistent=args.w_close_consistent,
        w_people_penalty_per_over6=args.w_people_penalty_per_over6,
        filter_out_jackal=not args.no_filter_jackal,
    )


if __name__ == "__main__":
    main()