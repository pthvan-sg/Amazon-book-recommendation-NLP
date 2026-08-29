"""
Composite ranking system: blends star ratings with review-level emotion
signal into a single per-book score, then aggregates to a book-level rank.

    emotion_score  = (admiration + approval + gratitude)
                     - (annoyance + disapproval + neutral)

    star_rank      = percentile rank of star_rating
    emotion_rank   = percentile rank of emotion_score
    composite_score = 0.4 * star_rank + 0.6 * emotion_rank

The 0.4/0.6 weighting (rather than an equal split) was chosen after a
sensitivity check across weightings: star ratings are heavily skewed
(~61% five-star in this dataset), so emotion_rank is given more influence
to pull the resulting distribution back toward a symmetric, bell-shaped
curve without discarding the explicit rating signal. See
results/model_comparison.png and the notebook's sensitivity-analysis
section for the alternative weightings tested.
"""

import pandas as pd

EMOTIONS_FOR_SCORE = ["admiration", "annoyance", "approval", "disapproval",
                      "gratitude", "neutral"]
POSITIVE_EMOTIONS = ["admiration", "approval", "gratitude"]
NEGATIVE_EMOTIONS = ["annoyance", "disapproval", "neutral"]


def attach_emotion_scores(df: pd.DataFrame, emotion_predictions: list) -> pd.DataFrame:
    """Given DistilBERT's per-review, per-emotion probability scores,
    add one column per tracked emotion plus a single weighted emotion_score.
    """
    df = df.copy()
    for emotion in EMOTIONS_FOR_SCORE:
        df[f"{emotion}_score"] = 0.0

    for i, prediction_set in enumerate(emotion_predictions):
        for entry in prediction_set:
            label, score = entry["label"], entry["score"]
            if label in EMOTIONS_FOR_SCORE:
                df.loc[i, f"{label}_score"] = score

    df["emotion_score"] = (
        sum(df[f"{e}_score"] for e in POSITIVE_EMOTIONS)
        - sum(df[f"{e}_score"] for e in NEGATIVE_EMOTIONS)
    )
    return df


def compute_composite_score(df: pd.DataFrame, star_weight: float = 0.4,
                             emotion_weight: float = 0.6) -> pd.DataFrame:
    """Convert both signals to percentile ranks (0-1) so they're on a
    comparable scale, then blend with the given weights."""
    df = df.copy()
    df["star_rank"] = df["star_rating"].rank(pct=True)
    df["emotion_rank"] = df["emotion_score"].rank(pct=True)
    df["composite_score"] = df["star_rank"] * star_weight + df["emotion_rank"] * emotion_weight
    return df


def aggregate_to_book_level(df: pd.DataFrame, title_col: str = "Title_review") -> pd.DataFrame:
    """Average composite score per book — smooths outliers and gives
    books with more reviews a more reliable final ranking score."""
    book_ranking = df.groupby(title_col)["composite_score"].mean().reset_index()
    return book_ranking.rename(columns={"composite_score": "avg_composite_score"})
