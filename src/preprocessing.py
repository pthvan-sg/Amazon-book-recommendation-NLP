"""
Data preparation utilities for the Amazon book review pipeline.

Covers three source datasets:
  DS1 — Amazon Books Reviews (star ratings + review text)
  DS4 — Amazon book metadata (title, author, category)
  DS6 — GoEmotions (fine-grained emotion labels, used to train the classifier)

Extracted from the project notebook (notebooks/amazon_book_recommendation.ipynb)
and reorganized into reusable functions.
"""

import re

import pandas as pd


# --------------------------------------------------------------------------
# DS1: Amazon review cleaning
# --------------------------------------------------------------------------

def standardize_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Amazon export columns to clean, consistent names and drop
    identifiers that aren't needed for sentiment/emotion analysis."""
    df = df.rename(columns={
        "review/score": "rating",
        "review/text": "text",
        "review/summary": "summary",
        "review/helpfulness": "helpfulness",
    })
    columns_to_drop = ["User_id", "profileName", "review/time"]
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])


def drop_missing_text_or_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing review or rating; cast rating to int."""
    df = df.dropna(subset=["text", "rating"])
    df["rating"] = df["rating"].astype(int)
    return df


def filter_min_reviews_per_book(df: pd.DataFrame, title_col: str = "Title",
                                 min_reviews: int = 3) -> pd.DataFrame:
    """Keep only books with at least `min_reviews` reviews, so per-book
    aggregation later isn't dominated by single-review noise."""
    counts = df.groupby(title_col).size().reset_index(name="review_count")
    keep_titles = counts[counts["review_count"] >= min_reviews][title_col]
    return df[df[title_col].isin(keep_titles)].copy()


def clean_text(text: str) -> str:
    """Lowercase, strip URLs and non-alphabetic characters, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(text.split())
    return text.strip()


def preprocess_review_text(df: pd.DataFrame, min_length: int = 30) -> pd.DataFrame:
    """Add a review_length column, clean the review text, and drop reviews
    too short to carry meaningful sentiment/emotion signal."""
    df["review_length"] = df["text"].str.len()
    df["clean_review"] = df["text"].apply(clean_text)
    return df[df["review_length"] > min_length]


def calculate_helpful_ratio(help_str: str) -> float:
    """Parse Amazon's 'x/y' helpfulness string into a 0-1 ratio."""
    try:
        numerator_str, denominator_str = str(help_str).split("/")
        numerator, denominator = int(numerator_str), int(denominator_str)
        return numerator / denominator if denominator > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def add_helpfulness_features(df: pd.DataFrame) -> pd.DataFrame:
    df["helpful_ratio"] = df["helpfulness"].apply(calculate_helpful_ratio)
    df["review_length"] = df["text"].str.len()
    return df


# --------------------------------------------------------------------------
# DS4: Book metadata cleaning
# --------------------------------------------------------------------------

def clean_book_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, drop rows with no title, standardize the title field,
    and fill missing author/category with 'unknown' rather than dropping."""
    df = df.drop_duplicates(subset=["Title"])
    df = df.dropna(subset=["Title"])
    df["Title"] = df["Title"].str.lower().str.strip()
    df["authors"] = df["authors"].fillna("unknown")
    df["categories"] = df["categories"].fillna("unknown")
    return df


def normalize_title(text: str) -> str:
    """Aggressive normalization (lowercase, alphanumeric-only) used purely
    as a join key between the review and metadata datasets."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    return re.sub(r"[^a-z0-9]", "", text).strip()


def merge_reviews_with_metadata(df_reviews: pd.DataFrame, df_books: pd.DataFrame,
                                 title_col: str = "Title") -> pd.DataFrame:
    """Left-join reviews to book metadata on a normalized title, so every
    review is retained even where a metadata match isn't found."""
    df_reviews = df_reviews.copy()
    df_books = df_books.copy()
    df_reviews["norm_title"] = df_reviews[title_col].apply(normalize_title)
    df_books["norm_title"] = df_books[title_col].apply(normalize_title)

    df_books_unique = df_books.drop_duplicates(subset=["norm_title"])
    merged = pd.merge(
        df_reviews, df_books_unique, on="norm_title", how="left",
        suffixes=("_review", "_meta"),
    )
    return merged


# --------------------------------------------------------------------------
# DS6: GoEmotions cleaning + label preparation
# --------------------------------------------------------------------------

EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def drop_unlabeled_rows(df: pd.DataFrame, emotion_cols=EMOTION_COLUMNS) -> pd.DataFrame:
    """Remove GoEmotions rows with no active emotion label at all."""
    return df[df[emotion_cols].sum(axis=1) > 0]


def filter_significant_emotions(df: pd.DataFrame, threshold: float = 0.05,
                                 emotion_cols=EMOTION_COLUMNS) -> pd.DataFrame:
    """Reduce GoEmotions' 27 (heavily imbalanced) emotion labels down to
    those appearing in at least `threshold` fraction of rows, then flatten
    each row's multi-label list to a single dominant label for training.

    A 5% threshold (following Demszky et al., 2020) keeps a manageable set
    of high-signal classes without collapsing to only "neutral".
    """
    counts = df[emotion_cols].sum()
    significant = counts[counts > threshold * len(df)].index.tolist()
    label_to_int = {emotion: i for i, emotion in enumerate(significant)}

    rows = []
    for _, row in df.iterrows():
        for emotion in significant:
            if row[emotion] == 1:
                rows.append({
                    "text": row["text"],
                    "emotion_label": emotion,
                    "emotion_label_int": label_to_int[emotion],
                })
                break

    return pd.DataFrame(rows), significant, label_to_int
