import os
import pandas as pd

from newsreclib.utils import pylogger

log = pylogger.get_pylogger(__name__)


def _get_nmb_acc(bucket_info):
    """Compute news metrics using accumulated (ACC) logic.
    
    ACC logic calculates metrics by considering all events that occurred from the start
    up until each time bucket tx. This means values are cumulative across time buckets.
    For example, clicks at time t3 would include all clicks from t1, t2, and t3.
    """
    # --- Compute ptb logic first
    # Compute num_clicks and exposures by grouping by 'time_bucket' and 'news_id'
    news_metrics_bucket = (
        bucket_info.groupby([
            "time_bucket", 
            "news_id", 
            "time_bucket_end_hour", 
            "time_bucket_start_hour"
        ])
        .agg(num_clicks=("clicked", "sum"), exposures=("clicked", "count"))
        .reset_index()
    )

    # Compute total number of impressions per time bucket
    total_impressions = (
        bucket_info.groupby("time_bucket").size().reset_index(name="total_impressions")
    )

    # Merge to get the total impressions per time bucket alongside the news_metrics
    news_metrics_bucket = pd.merge(
        news_metrics_bucket, total_impressions, on="time_bucket"
    )

    # --- Compute acc logic now
    # Sort the DataFrame by 'news_id' and 'time_bucket' to ensure correct order for cumulative sum
    news_metrics_bucket = news_metrics_bucket.sort_values(by=["news_id", "time_bucket"])

    # Calculate cumulative sums for 'num_clicks' and 'exposures' to get 'num_clicks' and 'exposures'
    news_metrics_bucket["num_clicks"] = news_metrics_bucket.groupby("news_id")[
        "num_clicks"
    ].cumsum()
    news_metrics_bucket["exposures"] = news_metrics_bucket.groupby("news_id")[
        "exposures"
    ].cumsum()

    # Compute cumulative total impressions as well
    news_metrics_bucket["total_impressions"] = news_metrics_bucket.groupby("news_id")[
        "total_impressions"
    ].cumsum()

    # Exposure Per Impression (EPI)
    news_metrics_bucket["epi"] = (
        news_metrics_bucket["exposures"] / news_metrics_bucket["total_impressions"]
    )

    # Avoidance Rate (AR)
    news_metrics_bucket["av"] = (
        1 - news_metrics_bucket["num_clicks"] / news_metrics_bucket["exposures"]
    )

    return news_metrics_bucket


def _get_nmb_ptb(bucket_info):
    """Compute news metrics using per-time-bucket (PTB) logic.
    
    PTB logic calculates metrics independently for each time bucket [t1, t2].
    Values are specific to what happened within each bucket's time window,
    without considering previous buckets. For example, clicks at time t3 would
    only include clicks that occurred during t3's time window.
    """
    # Compute num_clicks and exposures by grouping by 'time_bucket' and 'news_id'
    news_metrics_bucket = (
        bucket_info.groupby([
            "time_bucket", 
            "news_id", 
            "time_bucket_end_hour", 
            "time_bucket_start_hour"
        ])
        .agg(num_clicks=("clicked", "sum"), exposures=("clicked", "count"))
        .reset_index()
    )

    # Compute total number of impressions per time bucket
    total_impressions = (
        bucket_info.groupby("time_bucket").size().reset_index(name="total_impressions")
    )

    # Merge article metrics with total impressions to get all information together
    news_metrics_bucket = pd.merge(
        news_metrics_bucket, total_impressions, on="time_bucket"
    )

    # exposure per impression (epi)
    news_metrics_bucket["epi"] = (
        news_metrics_bucket["exposures"] / news_metrics_bucket["total_impressions"]
    )

    # avoidance rate (ar)
    news_metrics_bucket["av"] = (
        1 - news_metrics_bucket["num_clicks"] / news_metrics_bucket["exposures"]
    )

    return news_metrics_bucket


def get_news_metrics_bucket(file_type, bucket_info, path, article2published, matrix_size=5):
    """Calculate news metrics using either PTB or ACC logic.
    
    Args:
        file_type (str): Type of calculation logic to use:
            - 'ptb': Per-time-bucket calculation (metrics isolated to each time window)
            - 'acc': Accumulated calculation (metrics accumulate across time windows)
        bucket_info (pd.DataFrame): DataFrame containing the news interaction data
        path (str): Path to save the resulting metrics
        article2published (dict): Mapping of article IDs to their publish times
        matrix_size (int, optional): Size of the classification matrix. Defaults to 5.
    """
    # -- Select logic type
    if file_type == "ptb":
        news_metrics_bucket = _get_nmb_ptb(bucket_info)
    elif file_type == "acc":
        news_metrics_bucket = _get_nmb_acc(bucket_info)

    # -- Clicks ratio
    total_clicks = (
        news_metrics_bucket.groupby("time_bucket")["num_clicks"]
        .sum()
        .reset_index(name="total_clicks")
    )

    # -- Merge this information back with the original DataFrame
    news_metrics_bucket = pd.merge(news_metrics_bucket, total_clicks, on="time_bucket")

    # -- Calculate 'clicks_ratio' as a percentage
    news_metrics_bucket["clicks_ratio"] = (
        news_metrics_bucket["num_clicks"] / news_metrics_bucket["total_clicks"]
    )

    # -- Replace any potential NaN values with 0 (in case there are time buckets with 0 clicks leading to division by 0)
    news_metrics_bucket["clicks_ratio"] = news_metrics_bucket["clicks_ratio"].fillna(0)

    # -- Normalize variables before processing
    news_metrics_bucket["epi"] = (
        news_metrics_bucket["epi"] - news_metrics_bucket["epi"].min()
    ) / (news_metrics_bucket["epi"].max() - news_metrics_bucket["epi"].min())

    news_metrics_bucket["clicks_ratio"] = (
        news_metrics_bucket["clicks_ratio"] - news_metrics_bucket["clicks_ratio"].min()
    ) / (
        news_metrics_bucket["clicks_ratio"].max()
        - news_metrics_bucket["clicks_ratio"].min()
    )

    news_metrics_bucket["av"] = (
        news_metrics_bucket["av"] - news_metrics_bucket["av"].min()
    ) / (news_metrics_bucket["av"].max() - news_metrics_bucket["av"].min())

    # -- Convert 'av' and 'epi' to indices for embedding computation later
    #    5x5 matrix to be created to categorized news articles (Default: 5x5)

    # EPI
    news_metrics_bucket["epi_idx"] = news_metrics_bucket["epi"].apply(
        lambda x: int(x * (matrix_size - 1))
    )
    news_metrics_bucket["epi_idx"] = news_metrics_bucket["epi_idx"].astype(int)

    # AV
    news_metrics_bucket["av_idx"] = news_metrics_bucket["av"].apply(
        lambda x: int(x * (matrix_size - 1))
    )
    news_metrics_bucket["av_idx"] = news_metrics_bucket["av_idx"].astype(int)

    # -- map publish time for articles
    news_metrics_bucket["news_pb_time"] = news_metrics_bucket["news_id"].map(
        article2published
    )

    # -- Save news metrics bucket into csv and pickle
    news_metrics_bucket.to_pickle(path)
    log.info(f"(News metric type ${file_type} bucket file created!")

    return news_metrics_bucket
