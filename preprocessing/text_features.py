import re
import torch
import clip
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------
# Load CLIP Model
# ---------------------------------------------------

device = "mps" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# ---------------------------------------------------
# Hashtag Utilities
# ---------------------------------------------------

HASHTAG_REGEX = re.compile(r"#\w+")

def extract_hashtags(text: str):
    """
    Returns list of hashtags (without '#') from caption.
    """
    if not isinstance(text, str):
        return []
    return [tag[1:].lower() for tag in HASHTAG_REGEX.findall(text)]

def remove_hashtags(text: str):
    """
    Strips hashtags out of caption text, leaving pure description.
    """
    if not isinstance(text, str):
        return ""
    return HASHTAG_REGEX.sub("", text).strip()

# ---------------------------------------------------
# CLIP Embedding Utilities
# ---------------------------------------------------

@torch.no_grad()
def clip_embed_batch(text_list, batch_size=64):
    """
    Generates CLIP text embeddings for a list of strings.
    """
    embeddings = []

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]

        tokens = clip.tokenize(batch, truncate=True).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize

        embeddings.append(feats.cpu().numpy())

    return np.vstack(embeddings)

# ---------------------------------------------------
# Main Feature Builder
# ---------------------------------------------------

def add_text_features(
    df,
    caption_col="caption",
    embed_desc=True,
    embed_tags=True,
    embed_full_caption=False,
):
    """
    Produces:
    - hashtag_list
    - hashtag_count
    - description_no_hashtags
    - clip_desc_embedding (optional)
    - clip_hashtag_embedding (optional)
    - clip_full_caption_embedding (optional)
    """

    df[caption_col] = df[caption_col].fillna("")

    # -------- Extract hashtags --------
    df["hashtag_list"] = df[caption_col].apply(extract_hashtags)
    df["hashtag_count"] = df["hashtag_list"].apply(len)

    # -------- Description-only text --------
    df["description_no_hashtags"] = df[caption_col].apply(remove_hashtags)

    # -------- CLIP Embeddings --------

    if embed_desc:
        print("Embedding description-only text...")
        desc_emb = clip_embed_batch(df["description_no_hashtags"].tolist())
        df["clip_desc_embedding"] = list(desc_emb)

    if embed_tags:
        print("Embedding hashtags-only text...")
        tag_strings = df["hashtag_list"].apply(lambda tags: " ".join(tags)).tolist()
        tag_emb = clip_embed_batch(tag_strings)
        df["clip_hashtag_embedding"] = list(tag_emb)

    if embed_full_caption:
        print("Embedding full caption text...")
        full_emb = clip_embed_batch(df[caption_col].tolist())
        df["clip_full_caption_embedding"] = list(full_emb)

    return df