import pandas as pd
import numpy as np

from newsreclib.utils import pylogger

log = pylogger.get_pylogger(__name__)


def aux_lst_f(series):
    """
    Define a custom aggregation function to create tuples of (history_news_id, num_clicks)
    """
    return list(series)

# --- User Engagement Variables
def _get_history_usr_eng(
    behaviors: pd.DataFrame, news_metrics_bucket: pd.DataFrame
) -> any:
    # Explode the 'history' column to individual rows for easier processing
    bhv_hist_explode = behaviors.explode("history")
    bhv_hist_explode = bhv_hist_explode.rename(columns={"history": "history_news_id"})
    bhv_hist_explode["time"] = pd.to_datetime(bhv_hist_explode["time"])

    # Filter the metrics bucket to include only news_ids present in bhv_hist_explode for efficiency
    unique_ids_metrics_bucket = news_metrics_bucket["news_id"].unique().tolist()
    bhv_hist_explode_filter = bhv_hist_explode[
        bhv_hist_explode["history_news_id"].isin(unique_ids_metrics_bucket)
    ]

    # Merge filtered behaviors with metrics on news_id
    merged_df = pd.merge(
        bhv_hist_explode_filter,
        news_metrics_bucket,
        left_on="history_news_id",
        right_on="news_id",
        how="left",
    )

    # Select entries where the behavioral event time is within the time bucket range
    valid_entries = merged_df[
        (merged_df["time"] >= merged_df["time_bucket_start_hour"])
        & (merged_df["time"] < merged_df["time_bucket_end_hour"])
    ]
    valid_entries = valid_entries.sort_values(by="time", ascending=False)
    final_df = valid_entries.drop_duplicates(
        subset=["history_news_id", "time"], keep="first"
    )

    # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
    bhv_hist_ctr = pd.merge(
        bhv_hist_explode_filter,
        final_df[
            [
                "history_news_id",
                "time",
                "num_clicks",
                "epi",
                "av",
                "epi_idx",
                "av_idx",
                "clicks_ratio",
            ]
        ],
        on=["history_news_id", "time"],
        how="left",
    )
    bhv_hist_ctr["num_clicks"] = bhv_hist_ctr["num_clicks"].fillna(0).astype(int)
    bhv_hist_ctr[["av", "epi", "clicks_ratio", "epi_idx", "av_idx"]] = (
        bhv_hist_ctr[["av", "epi", "clicks_ratio", "epi_idx", "av_idx"]]
        .fillna(0)
        .astype(float)
    )
    bhv_hist_ctr = bhv_hist_ctr.drop_duplicates(
        subset=["history_news_id", "time", "num_clicks"]
    )

    # Reaggregate to match the original data granularity and form a history column with CTR data
    final_df = pd.merge(
        bhv_hist_explode,
        bhv_hist_ctr[
            [
                "history_news_id",
                "impid",
                "uid",
                "user",
                "time",
                "num_clicks",
                "av",
                "epi",
                "clicks_ratio",
                "epi_idx",
                "av_idx",
            ]
        ],
        on=["history_news_id", "impid", "uid", "user", "time"],
        how="left",
    )

    final_df["num_clicks"] = final_df["num_clicks"].fillna(0).astype(int)
    final_df[["av", "epi", "clicks_ratio", "epi_idx", "av_idx"]] = (
        final_df[["av", "epi", "clicks_ratio", "epi_idx", "av_idx"]]
        .fillna(0)
        .astype(float)
    )

    result_df = (
        final_df.groupby(["impid", "uid", "user", "time"])
        .agg(
            {
                "history_news_id": list,
                "num_clicks": aux_lst_f,
                "av": aux_lst_f,
                "epi": aux_lst_f,
                "clicks_ratio": aux_lst_f,
                "epi_idx": aux_lst_f,
                "av_idx": aux_lst_f,
            }
        )
        .reset_index()
    )

    result_df["hist_usr_eng"] = result_df.apply(
        lambda x: list(
            zip(
                x["history_news_id"],
                x["num_clicks"],
                x["av"],
                x["av_idx"],
                x["epi"],
                x["epi_idx"],
                x["clicks_ratio"],
            )
        ),
        axis=1,
    )
    result_df = result_df.rename(columns={"history_news_id": "history"})

    # Validate that merged data matches original behaviors data
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    result_df["time"] = pd.to_datetime(result_df["time"])
    behaviors_ = pd.merge(
        behaviors, result_df, on=["impid", "uid", "user", "time"], how="inner"
    )
    diff_mask = behaviors_["history_x"] != behaviors_["history_y"]
    different_indexes = behaviors_.index[diff_mask].tolist()

    # Ensure no discrepancies exist
    assert len(different_indexes) == 0

    # Drop the 'history_y' column
    behaviors_ = behaviors_.drop(columns=["history_y"])

    # Rename 'history_x' to 'history'
    behaviors_ = behaviors_.rename(
        columns={
            "history_x": "history",
            "num_clicks": "hist_ctr",
            "av": "hist_av",
            "epi": "hist_epi",
            "clicks_ratio": "hist_clicks_ratio",
            "epi_idx": "hist_epi_idx",
            "av_idx": "hist_av_idx",
        }
    )

    return behaviors_


def _get_candidate_usr_eng(
    behaviors: pd.DataFrame,
    news_metrics_bucket: pd.DataFrame,
    article2published: any,
) -> any:
    # Explode the 'candidates' column to individual rows for easier processing
    bhv_cand_explode = behaviors.explode("candidates")
    bhv_cand_explode = bhv_cand_explode.rename(columns={"candidates": "cand_news_id"})
    bhv_cand_explode["pb_time"] = bhv_cand_explode["cand_news_id"].map(
        article2published
    )
    # -- In case of empty pb_time
    min_time = bhv_cand_explode["pb_time"].min()
    bhv_cand_explode["pb_time"] = bhv_cand_explode["pb_time"].fillna(min_time)
    bhv_cand_explode["time"] = pd.to_datetime(bhv_cand_explode["time"])

    # Filter the metrics bucket to include only news_ids present in bhv_cand_explode for efficiency
    unique_ids_metrics_bucket = news_metrics_bucket["news_id"].unique().tolist()
    bhv_cand_explode_filter = bhv_cand_explode[
        bhv_cand_explode["cand_news_id"].isin(unique_ids_metrics_bucket)
    ]

    # Merge filtered behaviors with metrics on news_id
    merged_df = pd.merge(
        bhv_cand_explode_filter,
        news_metrics_bucket,
        left_on="cand_news_id",
        right_on="news_id",
        how="left",
    )

    # Select entries where the behavioral event time is within the time bucket range
    valid_entries = merged_df[
        (merged_df["time"] >= merged_df["time_bucket_start_hour"])
        & (merged_df["time"] < merged_df["time_bucket_end_hour"])
    ]
    valid_entries = valid_entries.sort_values(by="time", ascending=False)
    final_df = valid_entries.drop_duplicates(
        subset=["cand_news_id", "time"], keep="first"
    )

    # Reintegrate CTR data back to behaviors, filling gaps with zero where no data was found
    bhv_cand_ctr = pd.merge(
        bhv_cand_explode_filter,
        final_df[
            [
                "cand_news_id",
                "time",
                "num_clicks",
                "av",
                "epi",
                "clicks_ratio",
                "av_idx",
                "epi_idx",
            ]
        ],
        on=["cand_news_id", "time"],
        how="left",
    )
    bhv_cand_ctr["num_clicks"] = bhv_cand_ctr["num_clicks"].fillna(0).astype(int)
    bhv_cand_ctr[["av", "epi", "clicks_ratio", "av_idx", "epi_idx"]] = (
        bhv_cand_ctr[["av", "epi", "clicks_ratio", "av_idx", "epi_idx"]]
        .fillna(0)
        .astype(float)
    )
    bhv_cand_ctr = bhv_cand_ctr.drop_duplicates(
        subset=["cand_news_id", "time", "num_clicks"]
    )

    # Reaggregate to match the original data granularity and form a cand column with CTR data
    final_df = pd.merge(
        bhv_cand_explode,
        bhv_cand_ctr[
            [
                "cand_news_id",
                "impid",
                "uid",
                "user",
                "time",
                "num_clicks",
                "av",
                "epi",
                "clicks_ratio",
                "av_idx",
                "epi_idx",
            ]
        ],
        on=["cand_news_id", "impid", "uid", "user", "time"],
        how="left",
    )
    final_df["num_clicks"] = final_df["num_clicks"].fillna(0).astype(int)
    final_df[["av", "epi", "clicks_ratio", "av_idx", "epi_idx"]] = (
        final_df[["av", "epi", "clicks_ratio", "av_idx", "epi_idx"]]
        .fillna(0)
        .astype(float)
    )

    # Get recency column
    final_df["time"] = pd.to_datetime(final_df["time"])
    final_df["cand_recency"] = (
        (final_df['time'] + pd.Timedelta(hours=1)) - final_df["pb_time"]
    ).dt.total_seconds() / 3600
    # Deal with inconsistencies of negative recency values
    final_df["cand_recency"] = final_df["cand_recency"].clip(lower=1)

    # Identify NaN values
    nan_mask = final_df.isna()

    # Identify inf values
    inf_mask = final_df.isin([np.inf, -np.inf])

    # Combine the masks to find rows with NaN or inf
    combined_mask = nan_mask | inf_mask

    # Display rows with NaN or inf values
    rows_with_nan_or_inf = final_df[combined_mask.any(axis=1)]

    final_df["cand_recency"] = final_df["cand_recency"].astype(int)

    # check if there's any negative value on the recency column
    assert False == (final_df["cand_recency"] < 0).any()

    result_df = (
        final_df.groupby(["impid", "uid", "user", "time"])
        .agg(
            {
                "cand_news_id": list,
                "num_clicks": aux_lst_f,
                "cand_recency": aux_lst_f,
                "av": aux_lst_f,
                "epi": aux_lst_f,
                "clicks_ratio": aux_lst_f,
                "av_idx": aux_lst_f,
                "epi_idx": aux_lst_f,
            }
        )
        .reset_index()
    )

    result_df = result_df.rename(columns={"cand_news_id": "candidates"})

    # compute candidates_usr_eng
    result_df["cand_usr_eng"] = result_df.apply(
        lambda x: list(
            zip(
                x["candidates"],
                x["num_clicks"],
                x["cand_recency"],
                x["av"],
                x["av_idx"],
                x["epi"],
                x["epi_idx"],
                x["clicks_ratio"],
            )
        ),
        axis=1,
    )

    # Validate that merged data matches original behaviors data
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    behaviors_ = pd.merge(
        behaviors, result_df, on=["impid", "uid", "user", "time"], how="inner"
    )
    diff_mask = behaviors_["candidates_x"] != behaviors_["candidates_y"]
    different_indexes = behaviors_.index[diff_mask].tolist()

    # Ensure no discrepancies exist
    assert len(different_indexes) == 0

    # Drop the 'candidates_y' column
    behaviors_ = behaviors_.drop(columns=["candidates_y"])

    # Rename 'cand_x' to 'cand'
    behaviors_ = behaviors_.rename(
        columns={
            "candidates_x": "candidates",
            "num_clicks": "cand_ctr",
            "av": "cand_av",
            "av_idx": "cand_av_idx",
            "epi": "cand_epi",
            "epi_idx": "cand_epi_idx",
            "clicks_ratio": "cand_clicks_ratio",
        }
    )

    return behaviors_


def _get_usr_eng(
    behaviors: pd.DataFrame,
    news_metrics_bucket: pd.DataFrame,
    article2published: any,
) -> any:
    """
    Calculate user engagement variables for each news article over its respective time buckets from the news_metrics_bucket DataFrame.
    It matches news articles by ID and checks that the behavioral event time falls within the designated time buckets.

    Parameters:
        behaviors (pd.DataFrame): DataFrame containing user behavior data.
        news_metrics_bucket (pd.DataFrame): DataFrame with metrics for each news article over specific time buckets.
        row (any): Unused in this snippet, but typically used for row-specific operations.
        article2published (any): Unused in this snippet, could be used for mapping articles to published info.

    Returns:
        pd.DataFrame: Behaviors DataFrame enriched with the CTR information and checks for data consistency.

    Example usage:
        - Input DataFrame row: {'news_id': 'N3128', 'time_bucket': '11/13/2019 14:00 to 15:00', 'num_clicks': 152}
        - Output: CTR values merged back into the original behaviors DataFrame.
    """
    # Assign impid to the index
    behaviors = behaviors.reset_index(names="impid")

    # Avoid the model to see the "future", each impression should access the time bucket 1 hour before the impression time
    behaviors["time"] = pd.to_datetime(behaviors["time"])
    timedelta = 5 # Adressa (5) | MIND (1)
    behaviors["time"] = behaviors["time"] - pd.Timedelta(hours=timedelta)

    # Get User engagment for history column
    df_history_usr_eng = _get_history_usr_eng(
        behaviors=behaviors, news_metrics_bucket=news_metrics_bucket
    )

    # Get User engagement for candidates column
    df_candidate_usr_eng = _get_candidate_usr_eng(
        behaviors=behaviors,
        news_metrics_bucket=news_metrics_bucket,
        article2published=article2published,
    )

    # Join informations
    behaviors = pd.merge(
        behaviors,
        df_history_usr_eng[
            [
                "impid",
                "hist_ctr",
                "hist_av",
                "hist_epi",
                "hist_clicks_ratio",
                "hist_usr_eng",
                "hist_av_idx",
                "hist_epi_idx",
            ]
        ],
        on="impid",
        how="left",
    )
    behaviors = pd.merge(
        behaviors,
        df_candidate_usr_eng[
            [
                "impid",
                "cand_ctr",
                "cand_recency",
                "cand_av",
                "cand_epi",
                "cand_clicks_ratio",
                "cand_usr_eng",
                "cand_av_idx",
                "cand_epi_idx",
            ]
        ],
        on="impid",
        how="left",
    )

    # Fix time column
    behaviors["time"] = behaviors["time"] + pd.Timedelta(hours=timedelta)

    return behaviors
