import os
import pandas as pd

# ==============================================================
# Fill Missing grp_model (Hierarchical Logic)
# ==============================================================

def _fill_grp_model_year_make(group: pd.DataFrame) -> pd.DataFrame:
    """Helper: fill grp_model within (lot_year, lot_make_cd)."""
    mode_val = group['grp_model'].mode()
    if not mode_val.empty:
        group['grp_model'] = group['grp_model'].fillna(mode_val[0])
    return group

def _fill_grp_model_make(group: pd.DataFrame) -> pd.DataFrame:
    """Helper: fill grp_model within lot_make_cd."""
    mode_val = group['grp_model'].mode()
    if not mode_val.empty:
        group['grp_model'] = group['grp_model'].fillna(mode_val[0])
    return group

def fill_missing_grp_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'grp_model' using hierarchical mode logic:
    1. Within (lot_year, lot_make_cd)
    2. Within (lot_make_cd)
    3. Drop remaining rows where grp_model is still NaN
    """
    df = df.groupby(['lot_year', 'lot_make_cd'], group_keys=False).apply(_fill_grp_model_year_make)
    df = df.groupby(['lot_make_cd'], group_keys=False).apply(_fill_grp_model_make)
    df = df.dropna(subset=['grp_model'])
    return df


# ==============================================================
# Clean Active buyers
# ==============================================================
def clean_active_buyers(df: pd.DataFrame) -> pd.DataFrame:

    df['mbr_lic_type'] = df['mbr_lic_type'].fillna(df['mbr_lic_type'].mode()[0])
    df['mbr_state'] = df['mbr_state'].fillna(df['mbr_state'].mode()[0])
    df['mbr_lic_type'] = df['mbr_lic_type'].replace('Automotive Related Business', 'General Business')

    df = df.groupby(['lot_year', 'lot_make_cd'], group_keys=False).apply(_fill_grp_model_year_make)
    df = df.groupby(['lot_make_cd'], group_keys=False).apply(_fill_grp_model_make)
    df = df.dropna(subset=['grp_model'])

    df['acv'] = df['acv'].mask(df['acv']<=0, df['plug_lot_acv'])

    return df

# ==============================================================
# Clean Non Active buyers
# ==============================================================
def clean_non_active_buyers(df: pd.DataFrame) -> pd.DataFrame:

    df['mbr_lic_type'] = df['mbr_lic_type'].fillna(df['mbr_lic_type'].mode()[0])
    df['mbr_state'] = df['mbr_state'].fillna(df['mbr_state'].mode()[0])
    df['mbr_lic_type'] = df['mbr_lic_type'].replace('Automotive Related Business', 'General Business')
    df = df.rename(columns={'mbr_lic_type': 'buyer_type'})

    return df

# ==============================================================
# 3️⃣ Clean Popular Lots
# ==============================================================

def clean_popular_lots(popular_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and ranks popular lots data:
    - Replace 'Automotive Related Business' → 'General Business'
    - Fill missing buyer_type with mode
    - Fill grp_model hierarchically
    - Deduplicate and keep top 6 per (buyer_type, mbr_state)
    """
    # Replace business type
    popular_df['buyer_type'] = popular_df['buyer_type'].replace(
        'Automotive Related Business', 'General Business'
    )

    # Fill buyer_type with mode
    mode_val = popular_df['buyer_type'].mode()
    if not mode_val.empty:
        popular_df['buyer_type'] = popular_df['buyer_type'].fillna(mode_val[0])

    # Fill grp_model using make-level mode
    popular_df = popular_df.groupby('grp_model', group_keys=False).apply(_fill_grp_model_make)

    # Replace acv with plug_lot_acv where acv is 0 or negative
    popular_df['median_acv'] = popular_df['median_acv'].mask(popular_df['median_acv']<=0, popular_df['median_plug_lot_acv'])

    # Sort + deduplicate
    popular_df_sorted = (
        popular_df
        .sort_values(['buyer_type', 'mbr_state', 'cnt'], ascending=[True, True, False])
        .drop_duplicates(subset=['buyer_type', 'mbr_state', 'lot_make_cd', 'grp_model'])
    )

    # Rank within buyer_type + state
    popular_df_sorted['rank_clean'] = (
        popular_df_sorted.groupby(['buyer_type', 'mbr_state']).cumcount() + 1
    )

    # Keep only top 6
    return popular_df_sorted[popular_df_sorted['rank_clean'] <= 6]


# ==============================================================
# 4️⃣ Clean Upcoming lots
# ==============================================================

def clean_upcoming_lots(df: pd.DataFrame) -> pd.DataFrame:

    mode_val = df['damage_type_desc'].mode()
    if not mode_val.empty:
        df['damage_type_desc'] = df['damage_type_desc'].fillna(mode_val[0])

    df = df.groupby(['lot_year', 'lot_make_cd'], group_keys=False).apply(_fill_grp_model_year_make)
    df = df.groupby(['lot_make_cd'], group_keys=False).apply(_fill_grp_model_make)
    df = df.dropna(subset=['grp_model'])

    df['acv'] = df['acv'].mask(df['acv']<=0, df['plug_lot_acv'])

    return df

# ==============================================================
#  Helper: Save DataFrame
# ==============================================================

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed DataFrame to CSV in data/processed folder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Saved: {output_path} ({len(df):,} rows)")

# ==============================================================
# 5️⃣ Full Pipeline Wrapper
# ==============================================================

def preprocess_all(active_buyers: pd.DataFrame,
                   non_active_buyers: pd.DataFrame,
                   popular_lots: pd.DataFrame,
                   upcoming_lots: pd.DataFrame,
                   output_dir: dir = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function will run preprocessing steps for active buyers, non-active buyers, popular lots and upcoming lots.
    """
    print("\n Starting preprocessing pipeline...\n")

    # Active buyers
    print("\nCleaning Active buyers\n")
    active_buyers = clean_active_buyers(active_buyers)
    save_processed_data(active_buyers, os.path.join(output_dir, "active_buyers.csv"))

    # Non-Active buyers
    print("\nCleaning Non-Active buyers\n")
    non_active_buyers = clean_non_active_buyers(non_active_buyers)
    save_processed_data(non_active_buyers, os.path.join(output_dir, "non_active_buyers.csv"))

    # Popular lots
    print("\nCleaning Popular Lots\n")
    popular_lots = clean_popular_lots(popular_lots)
    save_processed_data(popular_lots, os.path.join(output_dir, "popular_lots.csv"))

    # Upcoming Lots
    print("\nCleaning Upcoming Lots\n")
    upcoming_lots = clean_upcoming_lots(upcoming_lots)
    save_processed_data(upcoming_lots, os.path.join(output_dir, "upcoming_lots.csv"))

    return active_buyers, non_active_buyers, popular_lots, upcoming_lots


def main():
    active_buyers = pd.read_csv("data/raw/active_buyers.csv")
    nonactive_buyers = pd.read_csv("data/raw/non_active_buyers.csv")
    popular_lots = pd.read_csv("data/raw/popular_lots.csv")
    upcoming_lots = pd.read_csv("data/raw/upcoming_lots.csv")

    preprocess_all(active_buyers, nonactive_buyers, popular_lots, upcoming_lots)

if __name__ == "__main__":
    main()


