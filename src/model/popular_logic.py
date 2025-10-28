import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

def generate_final_recommendations(data, popular_lots_top6):

    #data = data[['mbr_lic_type','mbr_state','buyer_nbr','mbr_email']]

    if 'mbr_lic_type' in data.columns:
        data = data.rename(columns={'mbr_lic_type': 'buyer_type'})

    if 'buyer_nbr' in data.columns:
        data = data.rename(columns={'buyer_nbr': 'mbr_nbr'})

    if 'acv' in data.columns:
        data = data.rename(columns={'acv': 'median_acv'})

    if 'repair_cost' in data.columns:
        data = data.rename(columns={'repair_cost': 'median_repair_cost'})

    data = data.drop_duplicates(subset=['mbr_nbr'])

    # Step 1: Merge based on buyer_type and mbr_state

    merged = data.merge(
        popular_lots_top6,
        on=['buyer_type', 'mbr_state'],
        how='inner'
    )

    # Step 2: Format initial recommendations
    initial_recommendations = merged[[
        'mbr_nbr', 'mbr_email', 'buyer_type', 'mbr_state',
        'lot_make_cd', 'grp_model', 'rank', 'rank_clean',
        'median_acv', 'median_repair_cost'
    ]]

    final_reco_list = []
    processed_buyers = set()

    # Step 3: For buyers with initial matches
    for mbr_nbr, group in initial_recommendations.groupby('mbr_nbr'):
        buyer_type = group['buyer_type'].iloc[0]
        mbr_email = group['mbr_email'].iloc[0]
        mbr_state = group['mbr_state'].iloc[0]

        recos = group.sort_values('rank_clean').to_dict('records')
        processed_buyers.add(mbr_nbr)

        if len(recos) < 6:
            needed = 6 - len(recos)

            fallback_pool = (
                popular_lots_top6[popular_lots_top6['buyer_type'] == buyer_type]
                .sort_values('rank_clean')
            )

            already_recoed = {(r['lot_make_cd'], r['grp_model']) for r in recos}

            for _, row in fallback_pool.iterrows():
                key = (row['lot_make_cd'], row['grp_model'])
                if key in already_recoed:
                    continue

                recos.append({
                    'mbr_nbr': mbr_nbr,
                    'mbr_email': mbr_email,
                    'buyer_type': buyer_type,
                    'mbr_state': mbr_state,
                    'lot_make_cd': row['lot_make_cd'],
                    'grp_model': row['grp_model'],
                    'rank': row.get('rank'),
                    'rank_clean': row.get('rank_clean'),
                    'median_acv': row.get('median_acv'),
                    'median_repair_cost': row.get('median_repair_cost')
                })

                already_recoed.add(key)
                if len(recos) == 6:
                    break

        final_reco_list.extend(recos)

    # Step 4: Handle buyers with no initial match
    missing_mbrs = set(data['mbr_nbr'].unique()) - processed_buyers
    fallback_missing = data[data['mbr_nbr'].isin(missing_mbrs)]

    for _, row in fallback_missing.iterrows():
        mbr_nbr = row['mbr_nbr']
        mbr_email = row['mbr_email']
        buyer_type = row['buyer_type']
        mbr_state = row['mbr_state']

        fallback_pool = (
            popular_lots_top6[popular_lots_top6['buyer_type'] == buyer_type]
            .sort_values('cnt', ascending=False)
            .drop_duplicates(subset=['lot_make_cd', 'grp_model'])
            .head(6)
        )

        for _, lot in fallback_pool.iterrows():
            final_reco_list.append({
                'mbr_nbr': mbr_nbr,
                'mbr_email': mbr_email,
                'buyer_type': buyer_type,
                'mbr_state': mbr_state,
                'lot_make_cd': lot['lot_make_cd'],
                'grp_model': lot['grp_model'],
                'rank': lot.get('rank'),
                'rank_clean': lot.get('rank_clean'),
                'median_acv': lot.get('median_acv'),
                'median_repair_cost': lot.get('median_repair_cost')
            })

    # Step 5: Return final DataFrame
    return pd.DataFrame(final_reco_list).sort_values(by=['mbr_nbr', 'rank_clean'])



def match_recommendations_fast(final_recommendations, future_lots):
    # Build lookup for YMM exact matches
    future_groups = {
        k: v.reset_index(drop=True)
        for k, v in future_lots.groupby(["lot_make_cd", "grp_model"])
    }

    # Pre-store full arrays for fallback
    fallback_df = future_lots.reset_index(drop=True)
    fallback_vals = fallback_df[['acv','repair_cost']].values.astype(np.float32)

    results = []

    for _, row in tqdm(final_recommendations.iterrows(), total=len(final_recommendations)):
        make = row['lot_make_cd']
        model = row['grp_model']
        acv = row['median_acv']
        repair = row['median_repair_cost']

        # Step 1: Try fast group lookup
        match_df = future_groups.get((make, model), None)

        if match_df is not None and len(match_df) > 0:
            arr = match_df[['acv','repair_cost']].values.astype(np.float32)
            dist = np.abs(arr[:,0] - acv) + np.abs(arr[:,1] - repair)
            i = dist.argmin()
            selected = match_df.iloc[i]
            fallback_reason = "YMM"
        else:
            # 🔥 Only compute fallback if YMM missing
            dist = np.abs(fallback_vals[:,0] - acv) + np.abs(fallback_vals[:,1] - repair)
            i = dist.argmin()
            selected = fallback_df.iloc[i]
            fallback_reason = "Global"

        results.append({
            "mbr_nbr": row['mbr_nbr'],
            "recommended_lot_nbr": selected["lot_nbr"],
            "distance": float(dist[i]),
            "fallback_reason": fallback_reason
        })

    return pd.DataFrame(results)


# ------------------------ Save Utility ------------------------

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    print(f"\nSaved processed data to {output_path} ({len(df):,} rows)")


# ----------------------- Main Flow --------------------------

if __name__ == "__main__":
    upcoming_lots = pd.read_csv("data/processed/upcoming_lots.csv")
    popular_lots = pd.read_csv("data/processed/popular_lots.csv")

    nonactive_test = pd.read_csv("data/split/nonactive_test.csv")
    nonactive_holdout = pd.read_csv("data/split/nonactive_holdout.csv")
    cf_holdout = pd.read_csv("data/split/cf_holdout.csv")
    one_to_one_holdout = pd.read_csv("data/split/one_to_one_holdout.csv")

    nonactive_test_past_reco = generate_final_recommendations(nonactive_test, popular_lots)
    nonactive_test_reco = match_recommendations_fast(nonactive_test_past_reco, upcoming_lots)
    save_processed_data(nonactive_test_reco, "data/results/nonactive_test_reco.xlsx")

    nonactive_holdout_past_reco = generate_final_recommendations(nonactive_holdout, popular_lots)
    nonactive_holdout_reco = match_recommendations_fast(nonactive_holdout_past_reco, upcoming_lots)
    save_processed_data(nonactive_holdout_reco, "data/results/nonactive_holdout_reco.xlsx")

    cf_holdout = cf_holdout[['mbr_lic_type','buyer_nbr','mbr_state','mbr_email']]
    cf_holdout_past_reco = generate_final_recommendations(cf_holdout, popular_lots)
    cf_holdout_reco = match_recommendations_fast(cf_holdout_past_reco, upcoming_lots)
    save_processed_data(cf_holdout_reco, "data/results/cf_holdout_reco.xlsx")

    one_to_one_holdout = one_to_one_holdout[['mbr_lic_type','buyer_nbr','mbr_state','mbr_email']]
    one_to_one_holdout_past_reco = generate_final_recommendations(one_to_one_holdout, popular_lots)
    one_to_one_holdout_reco = match_recommendations_fast(one_to_one_holdout_past_reco, upcoming_lots)
    save_processed_data(one_to_one_holdout_reco, "data/results/one_to_one_holdout_reco.xlsx")




