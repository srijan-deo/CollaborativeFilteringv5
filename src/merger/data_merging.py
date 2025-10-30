import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from google.cloud import bigquery

def rename_tag_concat_and_pivot(
    cf_test_reco,
    one_to_one_test_reco,
    nonactive_test_reco,
    cf_holdout_reco,
    one_to_one_holdout_reco,
    nonactive_holdout_reco,
    cf_holdout_would_have,
    one_to_one_holdout_would_have
):
    """
    Full pipeline:
    1. Renames:
       - 'mbr_nbr' → 'input_buyer_nbr'
       - 'recommended_lot_nbr' → 'recommended_lot'
    2. Keeps ['input_buyer_nbr', 'recommended_lot']
    3. Adds 'identifier' (1=CF, 2=One-to-One, 3=Nonactive)
    4. Adds 'group' (test / holdout / would_have)
    5. Concatenates all 9 DataFrames (keeps duplicates)
    6. Reorders columns: identifier → group → input_buyer_nbr
    7. Pivots to create 6 columns for recommended lots per buyer
    8. Renames columns to lot_1 ... lot_6
    9. Converts lot columns to int
    10. Adds 'created_at' (current CST) and 'sent_at' (next day 7 AM CST)
    """

    def _rename_and_trim(df):
        rename_map = {}
        if 'mbr_nbr' in df.columns:
            rename_map['mbr_nbr'] = 'input_buyer_nbr'
        if 'recommended_lot_nbr' in df.columns:
            rename_map['recommended_lot_nbr'] = 'recommended_lot'
        df = df.rename(columns=rename_map)

        keep_cols = [col for col in ['input_buyer_nbr', 'recommended_lot'] if col in df.columns]
        return df[keep_cols]

    # --- Apply rename & trim ---
    cf_test_reco = _rename_and_trim(cf_test_reco)
    one_to_one_test_reco = _rename_and_trim(one_to_one_test_reco)
    nonactive_test_reco = _rename_and_trim(nonactive_test_reco)
    cf_holdout_reco = _rename_and_trim(cf_holdout_reco)
    one_to_one_holdout_reco = _rename_and_trim(one_to_one_holdout_reco)
    nonactive_holdout_reco = _rename_and_trim(nonactive_holdout_reco)
    cf_holdout_would_have = _rename_and_trim(cf_holdout_would_have)
    one_to_one_holdout_would_have = _rename_and_trim(one_to_one_holdout_would_have)

    # ✅ Create a separate copy for nonactive_holdout_would_have
    nonactive_holdout_would_have = nonactive_holdout_reco.copy(deep=True)
    #nonactive_holdout_would_have.to_excel('../data/results/nonactive_holdout_would_have_reco.xlsx')

    # --- Add identifier and group columns ---
    mapping = [
        # TEST
        (cf_test_reco, 1, 'test'),
        (one_to_one_test_reco, 2, 'test'),
        (nonactive_test_reco, 3, 'test'),

        # HOLDOUT
        (cf_holdout_reco, 1, 'holdout'),
        (one_to_one_holdout_reco, 2, 'holdout'),
        (nonactive_holdout_reco, 3, 'holdout'),

        # WOULD_HAVE
        (cf_holdout_would_have, 1, 'would_have'),
        (one_to_one_holdout_would_have, 2, 'would_have'),
        (nonactive_holdout_would_have, 3, 'would_have')
    ]

    for df, id_val, grp in mapping:
        df['identifier'] = id_val
        df['group'] = grp

    # --- Concatenate all together ---
    combined_recos = pd.concat([
        cf_test_reco,
        one_to_one_test_reco,
        nonactive_test_reco,
        cf_holdout_reco,
        one_to_one_holdout_reco,
        nonactive_holdout_reco,
        cf_holdout_would_have,
        one_to_one_holdout_would_have,
        nonactive_holdout_would_have
    ], ignore_index=True)

    # --- Reorder columns (identifier first, group second) ---
    cols = ['identifier', 'group', 'input_buyer_nbr', 'recommended_lot']
    combined_recos = combined_recos[cols]

    # --- Rank and pivot to create 6 columns for recommended lots ---
    combined_recos['rank'] = combined_recos.groupby(['identifier', 'group', 'input_buyer_nbr']).cumcount() + 1
    pivoted = combined_recos.pivot(
        index=['identifier', 'group', 'input_buyer_nbr'],
        columns='rank',
        values='recommended_lot'
    ).reset_index()

    # ✅ Rename recommendation columns to lot_1 ... lot_6
    pivoted.columns = [
        f'lot_{int(col)}' if isinstance(col, int) else col
        for col in pivoted.columns
    ]

    # ✅ Ensure all 6 lot columns exist
    lot_cols = [f'lot_{i}' for i in range(1, 7)]
    for col in lot_cols:
        if col not in pivoted.columns:
            pivoted[col] = 0

    pivoted = pivoted[['identifier', 'group', 'input_buyer_nbr'] + lot_cols]

    # ✅ Convert lot columns to int
    pivoted[lot_cols] = pivoted[lot_cols].fillna(0).astype(int)

    # ✅ Add created_at and sent_at timestamps
    cst = pytz.timezone('US/Central')
    now_cst = datetime.now(cst)
    next_day_7am_cst = (now_cst + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)

    pivoted['created_at'] = now_cst
    pivoted['sent_at'] = next_day_7am_cst

    return pivoted

# ==============================================================
# Save to Excel Helper (Always Excel)
# ==============================================================
def save_processed_data(df: pd.DataFrame):
    """
    Save the final merged recommendations file with tomorrow’s CST date.
    Output example: ../data/final/recommendations_2025-10-29.xlsx
    """

    # Get tomorrow’s date in CST (YYYY-MM-DD format)
    cst = pytz.timezone('US/Central')
    now_cst = datetime.now(cst)
    tomorrow_date = (now_cst + timedelta(days=1)).strftime("%Y-%m-%d")

    # Build file path (relative to root)
    file_path = f"data/final/recommendations_{tomorrow_date}.xlsx"

    # Ensure folder exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)

    # Save DataFrame
    df.to_excel(file_path, index=False)

    print(f"✅ File saved successfully as: {file_path}")


# ==============================================================
# Upload merged recommendations to BigQuery
# ==============================================================
def upload_to_bigquery(dataframe, table_id, project_id, credentials_path):

    print(f"\n📤 Uploading data to BigQuery table `{table_id}`...")

    # Initialize BigQuery client
    client = bigquery.Client.from_service_account_json(credentials_path)

    # Define job configuration (append mode + schema autodetect)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        autodetect=True
    )

    # Start upload job
    job = client.load_table_from_dataframe(
        dataframe,
        destination=f"{project_id}.{table_id}",
        job_config=job_config
    )

    job.result()  # Wait for load to complete

    print(f"✅ Data successfully appended to `{table_id}` in project `{project_id}`.")


def main():
    final_pivoted_recos = rename_tag_concat_and_pivot(cf_test_reco, one_to_one_test_reco, nonactive_test_reco, cf_holdout_reco,
        one_to_one_holdout_reco, nonactive_holdout_reco, cf_holdout_would_have, one_to_one_holdout_would_have)

if __name__ == "__main__":
    cf_test_reco = pd.read_excel('../../data/results/cf_test_reco.xlsx')
    one_to_one_test_reco = pd.read_excel('../../data/results/onetoone_test_reco.xlsx')
    nonactive_test_reco = pd.read_excel('../../data/results/nonactive_test_reco.xlsx')

    cf_holdout_reco = pd.read_excel('../../data/results/cf_holdout_reco.xlsx')
    one_to_one_holdout_reco = pd.read_excel('../../data/results/onetoone_holdout_reco.xlsx')
    nonactive_holdout_reco = pd.read_excel('../../data/results/nonactive_holdout_reco.xlsx')

    cf_holdout_would_have = pd.read_excel('../../data/results/cf_holdout_would_have_reco.xlsx')
    one_to_one_holdout_would_have = pd.read_excel('../../data/results/onetoone_holdout_would_have_reco.xlsx')

    final_pivoted_recos = rename_tag_concat_and_pivot(cf_test_reco, one_to_one_test_reco, nonactive_test_reco, cf_holdout_reco,
                                one_to_one_holdout_reco, nonactive_holdout_reco, cf_holdout_would_have,
                                one_to_one_holdout_would_have)

    save_processed_data(final_pivoted_recos)

    upload_to_bigquery(
        dataframe=final_pivoted_recos,
        table_id="member_reco.test",
        project_id="cprtqa-strategicanalytics-sp1",
        credentials_path="/Users/srdeo/OneDrive - Copart, Inc/secrets/cprtqa-strategicanalytics-sp1-8b7a00c4fbae.json"
    )