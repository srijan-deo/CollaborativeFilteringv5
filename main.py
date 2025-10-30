import os
import time
import pandas as pd
from datetime import datetime
from src.data.data_ingestion import get_bq_client, ingest_dataset
from src.data.data_preprocessing import preprocess_all
from src.data.data_splitting import action as run_split
from src.model.collaborativefiltering import run_batch_recommendations, save_processed_data, format_and_concat_two_groups
from src.model.one_to_one import refine_recommendations_parallel_per_buyer, save_processed_data as save_one_to_one_data
from src.model.popular_logic import (
    generate_final_recommendations,
    match_recommendations_fast,
    save_processed_data as save_popular_data
)
from src.merger.data_merging import rename_tag_concat_and_pivot, upload_to_bigquery, save_processed_data as save_merged_data

def log_time(step_name, start_time):
    duration = time.time() - start_time
    minutes = duration // 60
    seconds = duration % 60
    print(f"⏱️ {step_name} completed in {int(minutes)}m {seconds:.2f}s\n")

def main():
    print("🚀 Starting full recommendation pipeline...\n")
    overall_start = time.time()

    # ───────────────────────────────────────────────────────────────
    step = "STEP 1️⃣: BigQuery Ingestion"
    print(f"\n{step}")
    start = time.time()
    cred_path = "/Users/srdeo/OneDrive - Copart, Inc/secrets/cprtpr-datastewards-sp1-614d7e297848 (1).json"
    client = get_bq_client(cred_path)

    tasks = [
        ("Active Buyers", "src/queries/active_buyers.sql", "data/raw/active_buyers.csv"),
        ("Non-Active Buyers", "src/queries/non_active_buyers.sql", "data/raw/non_active_buyers.csv"),
        ("Popular Lots", "src/queries/popular_lots.sql", "data/raw/popular_lots.csv"),
        ("Upcoming Lots", "src/queries/upcoming_lots.sql", "data/raw/upcoming_lots.csv"),
    ]
    for name, query_path, output_path in tasks:
        print(f"   Ingesting: {name}")
        #ingest_dataset(client, query_path, output_path)
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 2️⃣: Preprocessing"
    print(f"\n{step}")
    start = time.time()
    active = pd.read_csv("data/raw/active_buyers.csv")
    nonactive = pd.read_csv("data/raw/non_active_buyers.csv")
    popular = pd.read_csv("data/raw/popular_lots.csv")
    upcoming = pd.read_csv("data/raw/upcoming_lots.csv")
    active_clean, nonactive_clean, popular_clean, upcoming_clean = preprocess_all(active, nonactive, popular, upcoming)
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 3️⃣: Splitting"
    print(f"\n{step}")
    start = time.time()
    test_cf, holdout_cf, test_1to1, holdout_1to1, test_nonactive, holdout_nonactive = run_split(active_clean, nonactive_clean)
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 4️⃣: ALS + FAISS Recommendations"
    print(f"\n{step}")
    start = time.time()

    test_cf = pd.read_csv("data/split/cf_test.csv")
    holdout_cf = pd.read_csv("data/split/cf_holdout.csv")

    cf_test_reco = run_batch_recommendations(test_cf)
    save_processed_data(cf_test_reco, "data/past_reco/cf_test_reco.xlsx")

    cf_holdout_would_have_reco = run_batch_recommendations(holdout_cf)
    save_processed_data(cf_holdout_would_have_reco, "data/past_reco/cf_holdout_would_have_reco.xlsx")

    combined_cf = format_and_concat_two_groups(df1=cf_test_reco, df2=cf_holdout_would_have_reco, group1="test",
                                               group2="would_have", identifier=1)

    upload_to_bigquery(
        dataframe=combined_cf,
        table_id="member_reco.test_past_reco",
        project_id="cprtqa-strategicanalytics-sp1",
        credentials_path="/Users/srdeo/OneDrive - Copart, Inc/cprtqa-strategicanalytics-sp1-8b7a00c4fbae.json"
    )

    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 5️⃣: One-to-One Refinement"
    print(f"\n{step}")
    start = time.time()
    refined_test_1to1 = refine_recommendations_parallel_per_buyer(test_1to1, upcoming_clean, max_workers=8)
    save_one_to_one_data(refined_test_1to1, "data/results/onetoone_test_reco.xlsx")

    refined_holdout_1to1 = refine_recommendations_parallel_per_buyer(holdout_1to1, upcoming_clean, max_workers=8)
    save_one_to_one_data(refined_holdout_1to1, "data/results/onetoone_holdout_would_have_reco.xlsx")

    refined_test_cf = refine_recommendations_parallel_per_buyer(
        pd.read_excel("data/past_reco/cf_test_reco.xlsx"), upcoming_clean, max_workers=8
    )
    save_one_to_one_data(refined_test_cf, "data/results/cf_test_reco.xlsx")

    refined_holdout_cf = refine_recommendations_parallel_per_buyer(
        pd.read_excel("data/past_reco/cf_holdout_would_have_reco.xlsx"), upcoming_clean, max_workers=8
    )
    save_one_to_one_data(refined_holdout_cf, "data/results/cf_holdout_would_have_reco.xlsx")
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 6️⃣: Popular-Based Fallback"
    print(f"\n{step}")
    start = time.time()
    final_nonactive_test = generate_final_recommendations(test_nonactive, popular_clean)
    final_nonactive_test_reco = match_recommendations_fast(final_nonactive_test, upcoming_clean)
    save_popular_data(final_nonactive_test_reco, "data/results/nonactive_test_reco.xlsx")

    final_nonactive_holdout = generate_final_recommendations(holdout_nonactive, popular_clean)
    final_nonactive_holdout_reco = match_recommendations_fast(final_nonactive_holdout, upcoming_clean)
    save_popular_data(final_nonactive_holdout_reco, "data/results/nonactive_holdout_reco.xlsx")

    cf_holdout_fallback = holdout_cf[['mbr_lic_type', 'buyer_nbr', 'mbr_state', 'mbr_email']]
    cf_holdout_past = generate_final_recommendations(cf_holdout_fallback, popular_clean)
    cf_holdout_reco = match_recommendations_fast(cf_holdout_past, upcoming_clean)
    save_popular_data(cf_holdout_reco, "data/results/cf_holdout_reco.xlsx")

    onetoone_holdout_fallback = holdout_1to1[['mbr_lic_type', 'buyer_nbr', 'mbr_state', 'mbr_email']]
    onetoone_holdout_past = generate_final_recommendations(onetoone_holdout_fallback, popular_clean)
    onetoone_holdout_reco = match_recommendations_fast(onetoone_holdout_past, upcoming_clean)
    save_popular_data(onetoone_holdout_reco, "data/results/one_to_one_holdout_reco.xlsx")
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    step = "STEP 7️⃣: Merge All & Final Pivot"
    print(f"\n{step}")
    start = time.time()
    final_pivoted = rename_tag_concat_and_pivot(
        pd.read_excel("data/results/cf_test_reco.xlsx"),
        pd.read_excel("data/results/onetoone_test_reco.xlsx"),
        pd.read_excel("data/results/nonactive_test_reco.xlsx"),
        pd.read_excel("data/results/cf_holdout_reco.xlsx"),
        pd.read_excel("data/results/one_to_one_holdout_reco.xlsx"),
        pd.read_excel("data/results/nonactive_holdout_reco.xlsx"),
        pd.read_excel("data/results/cf_holdout_would_have_reco.xlsx"),
        pd.read_excel("data/results/onetoone_holdout_would_have_reco.xlsx")
    )
    save_merged_data(final_pivoted)
    upload_to_bigquery(
        dataframe=final_pivoted,
        table_id="member_reco.test_future_reco",
        project_id="cprtqa-strategicanalytics-sp1",
        credentials_path="/Users/srdeo/OneDrive - Copart, Inc/cprtqa-strategicanalytics-sp1-8b7a00c4fbae.json"
    )
    log_time(step, start)

    # ───────────────────────────────────────────────────────────────
    print("🏁 ALL STEPS COMPLETED")
    log_time("TOTAL PIPELINE", overall_start)

if __name__ == "__main__":
    main()
