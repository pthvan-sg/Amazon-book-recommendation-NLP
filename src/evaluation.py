"""
Statistical evaluation of (a) whether sentiment carries information beyond
star ratings, and (b) how much the composite ranking system actually
changes book rankings versus star ratings alone.
"""

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def test_sentiment_vs_helpfulness(df: pd.DataFrame, star_ratings=(4, 5)):
    """Mann-Whitney U test: within a fixed star-rating band, are
    high-sentiment reviews rated more helpful than low-sentiment ones?

    A non-parametric test is used because helpfulness ratio is not
    normally distributed. Splitting within a star-rating band controls
    for the rating itself, isolating the effect of sentiment.

    Returns (statistic, p_value).
    """
    subset = df[df["star_rating"].isin(star_ratings)]
    median_score = subset["vader_compound_score"].median()

    high_sentiment = subset[subset["vader_compound_score"] >= median_score]["helpful_ratio"]
    low_sentiment = subset[subset["vader_compound_score"] < median_score]["helpful_ratio"]

    return mannwhitneyu(high_sentiment, low_sentiment, alternative="greater")


def correlation_tables(df: pd.DataFrame, columns: list):
    """Pearson (linear) and Spearman (monotonic, rank-based) correlation
    matrices reported side by side — star ratings are ordinal and
    helpfulness/sentiment are often skewed, so both are needed for a
    robust read on the relationships between them."""
    pearson = df[columns].corr(method="pearson")
    spearman = df[columns].corr(method="spearman")
    return pearson, spearman


def compare_rankings(df_reviews: pd.DataFrame, df_book_ranking: pd.DataFrame,
                      title_col: str = "Title_review"):
    """Build a book-level comparison of the star-rating rank vs. the
    composite-score rank, and compute the metrics used to judge how much
    the new system actually changes the leaderboard:

      - Spearman correlation between the two rank orderings
      - percentage of books whose rank changed at all
      - mean absolute displacement (MAD) in rank position
      - percentage of star-rating ties that the composite score breaks
    """
    baseline = (
        df_reviews.groupby(title_col)["star_rating"].mean()
        .reset_index().rename(columns={"star_rating": "avg_star_rating"})
    )
    comparison = pd.merge(baseline, df_book_ranking, on=title_col, how="inner")

    comparison["star_rank"] = comparison["avg_star_rating"].rank(ascending=False)
    comparison["composite_rank"] = comparison["avg_composite_score"].rank(ascending=False)
    comparison["rank_difference"] = comparison["star_rank"] - comparison["composite_rank"]
    comparison["absolute_rank_difference"] = comparison["rank_difference"].abs()

    rho, p_value = spearmanr(comparison["star_rank"], comparison["composite_rank"])

    total_books = len(comparison)
    books_changed = (comparison["rank_difference"] != 0).sum()
    pct_changed = 100 * books_changed / total_books

    mad = comparison["absolute_rank_difference"].mean()

    unique_star_ranks = comparison["star_rank"].nunique()
    pct_ties_broken = 100 * (total_books - unique_star_ranks) / total_books

    metrics = {
        "spearman_rho": rho,
        "spearman_p": p_value,
        "pct_books_reranked": pct_changed,
        "mean_absolute_displacement": mad,
        "pct_ties_broken": pct_ties_broken,
    }
    return comparison, metrics
