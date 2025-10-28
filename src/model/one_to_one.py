import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def recommend_lots_for_buyer(buyer_id, buyer_lots_df, upcoming_df, top_k=6):
    results = []
    used_lots = set()

    # Step 1: Loop over each lot the buyer has seen
    for _, row in buyer_lots_df.iterrows():
        acv, repair = row['acv'], row['repair_cost']
        year, make, model = row['lot_year'], row['lot_make_cd'], row['grp_model']
        input_vec = np.array([[acv, repair]])

        ymm_candidates = upcoming_df[
            (upcoming_df['lot_year'] == year) &
            (upcoming_df['lot_make_cd'] == make) &
            (upcoming_df['grp_model'] == model)
        ][['lot_nbr', 'acv', 'repair_cost']].dropna()

        if ymm_candidates.empty:
            ymm_candidates = upcoming_df[['lot_nbr', 'acv', 'repair_cost']].dropna()

        ymm_candidates = ymm_candidates[~ymm_candidates['lot_nbr'].isin(used_lots)]
        if ymm_candidates.empty:
            continue

        ymm_candidates = ymm_candidates.copy()
        ymm_candidates['manhattan_dist'] = manhattan_distances(
            ymm_candidates[['acv', 'repair_cost']].values, input_vec
        ).flatten()

        best_match = ymm_candidates.sort_values('manhattan_dist').iloc[0]

        results.append({
            'input_buyer_nbr': buyer_id,
            'original_lot': int(row['recommended_lot']),
            'recommended_lot': int(best_match['lot_nbr']),
            'manhattan_distance': float(best_match['manhattan_dist']),
            'source': 'Step 1 - YMM/Manhattan'
        })
        used_lots.add(int(best_match['lot_nbr']))

    # Step 2: Recent YMM
    if len(results) < top_k:
        most_recent = buyer_lots_df.sort_values('inv_dt', ascending=False).iloc[0]
        acv, repair = most_recent['acv'], most_recent['repair_cost']
        year, make, model = most_recent['lot_year'], most_recent['lot_make_cd'], most_recent['grp_model']
        input_vec = np.array([[acv, repair]])

        ymm_candidates = upcoming_df[
            (upcoming_df['lot_year'] == year) &
            (upcoming_df['lot_make_cd'] == make) &
            (upcoming_df['grp_model'] == model)
        ][['lot_nbr', 'acv', 'repair_cost']].dropna()

        ymm_candidates = ymm_candidates[~ymm_candidates['lot_nbr'].isin(used_lots)]
        if not ymm_candidates.empty:
            ymm_candidates = ymm_candidates.copy()
            ymm_candidates['manhattan_dist'] = manhattan_distances(
                ymm_candidates[['acv', 'repair_cost']].values, input_vec
            ).flatten()

            for _, r in ymm_candidates.sort_values('manhattan_dist').iterrows():
                results.append({
                    'input_buyer_nbr': buyer_id,
                    'original_lot': int(most_recent['recommended_lot']),
                    'recommended_lot': int(r['lot_nbr']),
                    'manhattan_distance': float(r['manhattan_dist']),
                    'source': 'Step 2 - Recent YMM/Manhattan'
                })
                used_lots.add(int(r['lot_nbr']))
                if len(results) >= top_k:
                    break

    # Step 3: Make-level fallback
    if len(results) < top_k:
        make = most_recent['lot_make_cd']
        input_vec = np.array([[acv, repair]])

        make_candidates = upcoming_df[
            (upcoming_df['lot_make_cd'] == make)
        ][['lot_nbr', 'acv', 'repair_cost']].dropna()
        make_candidates = make_candidates[~make_candidates['lot_nbr'].isin(used_lots)]

        if not make_candidates.empty:
            make_candidates = make_candidates.copy()
            make_candidates['manhattan_dist'] = manhattan_distances(
                make_candidates[['acv', 'repair_cost']].values, input_vec).flatten()

            for _, r in make_candidates.sort_values('manhattan_dist').iterrows():
                results.append({
                    'input_buyer_nbr': buyer_id,
                    'original_lot': int(most_recent['recommended_lot']),
                    'recommended_lot': int(r['lot_nbr']),
                    'manhattan_distance': float(r['manhattan_dist']),
                    'source': 'Step 3 - Global Make/Manhattan'
                })
                used_lots.add(int(r['lot_nbr']))
                if len(results) >= top_k:
                    break

    # Step 4: Global fallback
    if len(results) < top_k:
        input_vec = np.array([[acv, repair]])
        global_candidates = upcoming_df[['lot_nbr', 'acv', 'repair_cost']].dropna()
        global_candidates = global_candidates[~global_candidates['lot_nbr'].isin(used_lots)]

        if not global_candidates.empty:
            global_candidates = global_candidates.copy()
            global_candidates['manhattan_dist'] = manhattan_distances(
                global_candidates[['acv', 'repair_cost']].values, input_vec
            ).flatten()

            for _, r in global_candidates.sort_values('manhattan_dist').iterrows():
                results.append({
                    'input_buyer_nbr': buyer_id,
                    'original_lot': int(most_recent['recommended_lot']),
                    'recommended_lot': int(r['lot_nbr']),
                    'manhattan_distance': float(r['manhattan_dist']),
                    'source': 'Step 4 - Global Fallback Manhattan'
                })
                used_lots.add(int(r['lot_nbr']))
                if len(results) >= top_k:
                    break
    return results


def refine_recommendations_parallel_per_buyer(reco_df, upcoming_df, max_workers=4):
    # 🛠 Rename buyer_nbr and lot_nbr to match expected inputs
    reco_df.columns = reco_df.columns.str.strip().str.lower()

    reco_df = reco_df.rename(columns={
        'buyer_nbr': 'input_buyer_nbr',
        'lot_nbr': 'recommended_lot'
    })

    results = []
    futures = []
    grouped = list(reco_df.groupby('input_buyer_nbr'))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for buyer_id, group_df in grouped:
            futures.append(executor.submit(recommend_lots_for_buyer, buyer_id, group_df, upcoming_df))

        for f in tqdm(as_completed(futures), total=len(futures), desc="Refining recos"):
            try:
                results.extend(f.result())
            except Exception as e:
                print(f"⚠️ Skipped buyer due to error: {e}")

    return pd.DataFrame(results)

# ==============================================================
# Save to Excel Helper (Always Excel)
# ==============================================================
def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save a DataFrame to Excel (.xlsx) in a given path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    print(f"\nSaved processed data to {output_path} ({len(df):,} rows)")

if __name__ == "__main__":

    upcoming_lots = pd.read_csv("data/processed/upcoming_lots.csv")

    # TEST data low group
    data_low_test = pd.read_csv("data/split/one_to_one_test.csv")
    recommended_upcoming_df_lt6_test = refine_recommendations_parallel_per_buyer(data_low_test, upcoming_lots, max_workers=8)
    save_processed_data(recommended_upcoming_df_lt6_test, "data/results/onetoone_test_reco.xlsx")

    # HOLDOUT data low group (would have)
    data_low_holdout = pd.read_csv("data/split/one_to_one_holdout.csv")
    recommended_upcoming_df_lt6_holdout = refine_recommendations_parallel_per_buyer(data_low_holdout, upcoming_lots, max_workers=8)
    save_processed_data(recommended_upcoming_df_lt6_holdout, "data/results/onetoone_holdout_would_have_reco.xlsx")

    # TEST data high group
    data_cf_test = pd.read_excel("data/past_reco/cf_test_reco.xlsx")
    recommended_upcoming_df_gt6 = refine_recommendations_parallel_per_buyer(data_cf_test, upcoming_lots, max_workers=8)
    save_processed_data(recommended_upcoming_df_gt6, "data/results/cf_test_reco.xlsx")

    # HOLDOUT data high group (would have)
    data_cf_holdout = pd.read_excel("data/past_reco/cf_holdout_would_have_reco.xlsx")
    recommended_upcoming_df_gt6_holdout = refine_recommendations_parallel_per_buyer(data_cf_holdout, upcoming_lots, max_workers=8)
    save_processed_data(recommended_upcoming_df_gt6_holdout, "data/results/cf_holdout_would_have_reco.xlsx")