import pathlib

import pandas as pd
import os

def divide_in_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function split the data frame into Collaborative and 1-1 filtering groups
    """
    data_high = df[df['total_unique_lots_bid_by_buyers']>=7]
    data_low = df[df['total_unique_lots_bid_by_buyers']<7]

    return data_high, data_low

def odd_even_split(df: pd.DataFrame, buyer_col) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function split the data frame into Test and Holdout groups based on odd and even buyer numbers
    """
    df = df.copy()
    df['last_digit'] = df[buyer_col] % 10
    holdout_df = df[df['last_digit'] % 2 == 0].drop(columns='last_digit')
    test_df = df[df['last_digit'] % 2 != 0].drop(columns='last_digit')

    return holdout_df, test_df

def save_split_data(df: pd.DataFrame, output_path: str) -> None:
    """This function will save the split data into csv files"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Saved: {output_path} ({len(df):,} rows)")


def action(active_buyers: pd.DataFrame,
           nonactive_buyers: pd.DataFrame,
           output_dir: str = "data/split") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print(f"Number of active buyers: {active_buyers['buyer_nbr'].nunique()}")

    ## Group split
    data_high, data_low = divide_in_groups(active_buyers)
    print(f"Number of active buyers in CF: {data_high['buyer_nbr'].nunique()}")
    print(f"Number of active buyers in one-to-one: {data_low['buyer_nbr'].nunique()}")

    ## Test vs Holdout split
    holdout_df_cf, test_df_cf = odd_even_split(data_high, buyer_col='buyer_nbr')
    print(f"Number of CF buyers in Test: {test_df_cf['buyer_nbr'].nunique()}")
    print(f"Number of CF buyers in Control: {holdout_df_cf['buyer_nbr'].nunique()}")
    save_split_data(test_df_cf, os.path.join(output_dir, "cf_test.csv"))
    save_split_data(holdout_df_cf, os.path.join(output_dir, "cf_holdout.csv"))

    holdout_df_onetoone, test_df_onetoone = odd_even_split(data_low, buyer_col='buyer_nbr')
    print(f"Number of one-to-one buyers in Test: {test_df_onetoone['buyer_nbr'].nunique()}")
    print(f"Number of one-to-one buyers in Control: {holdout_df_onetoone['buyer_nbr'].nunique()}")
    save_split_data(test_df_onetoone, os.path.join(output_dir, "one_to_one_test.csv"))
    save_split_data(holdout_df_onetoone, os.path.join(output_dir, "one_to_one_holdout.csv"))

    ## Non-active buyers
    print(f"Number of non-active buyers: {nonactive_buyers['mbr_nbr'].nunique()}")
    holdout_df_nonactive, test_df_nonactive = odd_even_split(nonactive_buyers, buyer_col='mbr_nbr')
    print(f"Number of non-active buyers in Test: {test_df_nonactive['mbr_nbr'].nunique()}")
    print(f"Number of non-active buyers in Control: {holdout_df_nonactive['mbr_nbr'].nunique()}")
    save_split_data(test_df_nonactive, os.path.join(output_dir, "nonactive_test.csv"))
    save_split_data(holdout_df_nonactive, os.path.join(output_dir, "nonactive_holdout.csv"))

    return test_df_cf, holdout_df_cf, test_df_onetoone, holdout_df_onetoone, test_df_nonactive, test_df_nonactive


def main():
    active_buyers = pd.read_csv('data/processed/active_buyers.csv')
    nonactive_buyers = pd.read_csv('data/processed/non_active_buyers.csv')
    #popular_lots = pd.read_csv('data/processed/popular_lots.csv')
    #upcoming_lots = pd.read_csv('data/processed/upcoming_lots.csv')

    action(active_buyers, nonactive_buyers)

if __name__ == "__main__":
    main()



