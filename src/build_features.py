import numpy as np
import pandas as pd

from preprocessing.preprocessing import (
    parse_metric,
    add_datetime_features,
    add_hashtag_features,
    add_trailing_metrics,
    add_video_length_features,
    add_target_columns
)

from preprocessing.text_features import add_text_features


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

RAW_CSV_PATH = "../data/raw/tiktok_stats.csv"

VIDEO_CLIP_PATH = "../data/embeddings/X_clip.npy"
WHISPER_PATH = "../data/embeddings/X_whisper.npy"

OUTPUT_FEATURE_PATH = "../data/embeddings/X_features.npy"
OUTPUT_DF_PATH = "../data/embeddings/tiktok_full_features.pkl"

# ---------------------------------------------------------
# LOAD RAW DATA
# ---------------------------------------------------------

df = pd.read_csv(RAW_CSV_PATH)

# ---------------------------------------------------------
# METRIC CLEANING (K/M → numeric)
# ---------------------------------------------------------

for col in ["views", "likes", "comments", "shares", "saves"]:
    df[col] = df[col].apply(parse_metric)

# ---------------------------------------------------------
# ADD TARGET COLUMNS
# ---------------------------------------------------------

df = add_target_columns(df)

# ---------------------------------------------------------
# DATETIME FEATURES (FROM DATE + TIME)
# ---------------------------------------------------------

df = add_datetime_features(
    df,
    date_col="published date",
    time_col="published time"
)

# ---------------------------------------------------------
# VIDEO LENGTH FEATURES (YOUR NEW FUNCTION)
# ---------------------------------------------------------

df = add_video_length_features(df, length_col="video length")

# ---------------------------------------------------------
# HASHTAG COUNT + DENSITY (NUMERIC)
# ---------------------------------------------------------

df = add_hashtag_features(df, text_col="description")

# ---------------------------------------------------------
# TRAILING POST-BASED MOMENTUM FEATURES
# ---------------------------------------------------------

df = add_trailing_metrics(
    df,
    datetime_col="published_datetime",
    views_col="views",
    windows=(7, 14, 28)
)

# ---------------------------------------------------------
# CLIP TEXT FEATURES (DESC + HASHTAGS)
# ---------------------------------------------------------

df = add_text_features(
    df,
    caption_col="description",
    embed_desc=True,
    embed_tags=True,
    embed_full_caption=False
)

# ---------------------------------------------------------
# LOAD PRECOMPUTED MULTIMODAL EMBEDDINGS
# ---------------------------------------------------------

X_clip = np.load(VIDEO_CLIP_PATH)       # shape (N, 512)
X_whisper = np.load(WHISPER_PATH)       # shape (N, D)

assert len(df) == X_clip.shape[0], "Row mismatch: df vs X_clip"
assert len(df) == X_whisper.shape[0], "Row mismatch: df vs X_whisper"

# ---------------------------------------------------------
# FINAL NUMERIC FEATURE STACK ✅
# ---------------------------------------------------------

numeric_features = [

    # Video length
    "video length",
    "log_video_length",
    "video_length_sq",

    # Time encodings
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "month",

    # Hashtag stats
    "hashtag_count",
    "hashtag_density",

    # Trailing momentum (post-date based)
    "videos_last_7d",
    "views_last_7d",
    "views_per_video_last_7d",

    "videos_last_14d",
    "views_last_14d",
    "views_per_video_last_14d",

    "videos_last_28d",
    "views_last_28d",
    "views_per_video_last_28d",
]

X_numeric = df[numeric_features].fillna(0).values
X_text_desc = np.vstack(df["clip_desc_embedding"].values)
X_text_tags = np.vstack(df["clip_hashtag_embedding"].values)

# ---------------------------------------------------------
# FINAL MULTIMODAL CONCATENATION
# ---------------------------------------------------------

X_final = np.concatenate(
    [
        X_numeric,
        X_text_desc,
        X_text_tags,
        X_clip,
        X_whisper,
    ],
    axis=1
)

print("\n===== FEATURE BLOCK SHAPES =====")
print("X_numeric     :", X_numeric.shape)
print("X_text_desc   :", X_text_desc.shape)
print("X_text_tags   :", X_text_tags.shape)
print("X_clip        :", X_clip.shape)
print("X_whisper     :", X_whisper.shape)
print("X_final       :", X_final.shape)
print("================================\n")

# ---------------------------------------------------------
# SAVE OUTPUTS ✅
# ---------------------------------------------------------

np.save(OUTPUT_FEATURE_PATH, X_final)
df.to_pickle(OUTPUT_DF_PATH)

print("\n✅ BUILD COMPLETE")
print("✅ X saved to:", OUTPUT_FEATURE_PATH)
print("✅ Full df saved to:", OUTPUT_DF_PATH)
print("✅ Final Feature Shape:", X_final.shape)