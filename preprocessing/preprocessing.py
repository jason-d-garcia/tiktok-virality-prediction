import numpy as np
import pandas as pd

def parse_metric(val):
    if pd.isna(val):
        return 0
    val = str(val).strip().lower()
    if 'k' in val:
        return float(val.replace('k', '')) * 1_000
    elif 'm' in val:
        return float(val.replace('m', '')) * 1_000_000
    return float(val.replace(',', ''))


def add_datetime_features(df, date_col="published date", time_col="published time"):
    df["published_datetime"] = pd.to_datetime(
        df[date_col] + " " + df[time_col],
        format="mixed",
        dayfirst=False
    )

    dt = df["published_datetime"]

    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df

def add_hashtag_features(df, text_col="caption"):
    df[text_col] = df[text_col].fillna("")
    df["caption_length"] = df[text_col].apply(len)
    df["hashtag_count"] = df[text_col].apply(lambda x: x.count("#"))
    df["hashtag_density"] = df["hashtag_count"] / df["caption_length"].clip(lower=1)
    return df

def parse_seconds(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().strip().replace("s", "")
    val = val.replace(",", ".")
    val = val.replace(":", ".")

    return float(val)

def add_target_columns(df):
    """
    Adds all derived target columns to the dataframe.
    Does NOT select the target.
    """

    # ----- Views Targets -----
    df["log_views"] = np.log1p(df["views"])

    # ----- Engagement Targets -----
    df["engagement_rate"] = (
        df["likes"] + df["comments"] + df["shares"] + df["saves"]
    ) / df["views"].replace(0, np.nan)

    df["log_engagement_rate"] = np.log1p(df["engagement_rate"])

    # ----- Other Potential Targets -----
    # df["growth_velocity"] = df["views"] / df["videos_last_7d"].replace(0, np.nan)
    # df["pct_female"] = df["pct_female"]
    # df["avg_watch_time"] = df["avg_watch_time"]
    # df["log_avg_watch_time"] = np.log1p(df["avg_watch_time"])
    # df["pct_US"] = df["pct_US"]

    return df

def add_video_length_features(df, length_col="video length"):
    # Parse raw seconds (e.g., "12s" → 12.0)
    df[length_col] = df[length_col].apply(parse_seconds)


    # Add nonlinear features
    df["log_video_length"] = np.log1p(df["video length"])
    df["video_length_sq"] = df["video length"] ** 2

    return df

def add_trailing_metrics(
    df,
    datetime_col="published_datetime",
    views_col="views",
    windows=(7, 14, 28),
):
    """
    Adds trailing posting + performance metrics based on the post date of the video.

    For each video V with timestamp T:
       - Count videos posted (T - w, T)
       - Sum views over (T - w, T)
       - Compute mean views/video over window

    IMPORTANT:
    df must contain a datetime column AND be sorted by it.
    """

    # Sort chronologically
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # We rely on array operations to avoid O(N^2) loops
    timestamps = df[datetime_col].values
    views = df[views_col].values

    for w in windows:
        video_counts = np.zeros(len(df))
        view_sums = np.zeros(len(df))
        view_means = np.zeros(len(df))

        w_delta = np.timedelta64(w, "D")

        start_idx = 0

        for i in range(len(df)):
            T = timestamps[i]
            lower_bound = T - w_delta

            # Move the start_idx forward while older videos fall outside the window
            while start_idx < i and timestamps[start_idx] < lower_bound:
                start_idx += 1

            # Window is (start_idx → i-1)
            count = (i - start_idx)
            if count > 0:
                window_views = views[start_idx:i]
                total_views = window_views.sum()
                mean_views = total_views / count
            else:
                total_views = 0
                mean_views = 0

            video_counts[i] = count
            view_sums[i] = total_views
            view_means[i] = mean_views

        df[f"videos_last_{w}d"] = video_counts
        df[f"views_last_{w}d"] = view_sums
        df[f"views_per_video_last_{w}d"] = view_means

    return df