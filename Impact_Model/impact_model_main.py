"""
impact_model_main.py
====================
Public interface for the YouTube Impact Score Model (v9).

This is the only file the root orchestrator needs to call.
It expects all data to be fetched and pre-assembled externally.
No API calls, no I/O, no fetching of any kind happens here.

Typical call from the root fetcher
-----------------------------------
    from impact_model_main import run_impact_model

    result = run_impact_model(
        video_metadata   = metadata_dict,       # from YouTube Data API
        comments         = list_of_comment_strs,# from YouTube commentThreads API
        transcript_text  = raw_transcript_str,  # from transcript fetcher
        sentiment_data   = sentiment_dict,       # pre-computed OR pass {} to compute here
    )

See run_impact_model() docstring for full parameter spec and return schema.
"""

import logging
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from feature_engineering import ImpactFeatureEngineering
from transcript_processor import TranscriptProcessor
from scoring_model import YouTubeImpactModel

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Module-level singletons (initialised once at import time) ─────────────────
_feature_engineer    = ImpactFeatureEngineering()
_transcript_processor = TranscriptProcessor()

_impact_model = YouTubeImpactModel()
_impact_model.initialize_population_stats()

logger.info("impact_model_main ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_impact_model(
    video_metadata:  dict,
    comments:        list,
    transcript_text: str,
    sentiment_data:  dict = None,
) -> dict:
    """
    Score a single YouTube video for impact.

    Parameters
    ----------
    video_metadata : dict
        Flat dictionary produced by the root fetcher from the YouTube Data API.
        Required keys (all others are used if present):

        Core counts
          view_count          int     total lifetime views
          like_count          float   total likes
          comment_count       float   total comments

        Temporal
          video_age_days      int     days since publish
          views_per_day       float   lifetime views / age
          likes_per_day       float   lifetime likes / age
          comments_per_day    float   lifetime comments / age
          views_per_day_log   float   log1p(views_per_day)
          publish_season      str     'Winter' | 'Spring' | 'Summer' | 'Fall'
          is_weekend          int     1 if published on weekend, else 0

        Title / description
          title               str     video title
          title_length        int     len(title)
          title_word_count    int     word count of title
          title_has_question  int     1 if '?' in title
          title_has_exclamation int   1 if '!' in title
          description_length  int     len(description)
          description_word_count int  word count of description
          has_description     int     1 if description non-empty

        Tags
          tag_count           int     number of tags

        Duration
          duration_seconds    float
          duration_minutes    float

        Optional enrichment (used if present, safe to omit)
          quality_engagement  float   any pre-computed quality-weighted engagement signal
          discussion_ratio    float   comments / likes  (for shareability)
          category_name       str     YouTube category label

    comments : list[str]
        Raw comment strings fetched by the root fetcher.
        Pass an empty list [] if comments are disabled or unavailable.

    transcript_text : str
        Raw transcript text (plain string, no timestamps).
        Pass '' if no transcript is available.

    sentiment_data : dict, optional
        Pre-computed sentiment signals. If provided, these values are merged
        directly into the feature row and the model skips re-computing them.
        Expected keys (all optional within the dict):
          avg_sentiment, sentiment_polarity, sentiment_subjectivity,
          positive_ratio, negative_ratio, neutral_ratio, sentiment_variance
        Pass None (default) to let this function compute sentiment from `comments`.

    Returns
    -------
    dict with keys:
        impact_score      float  0–100
        impact_level      str    'minimal' | 'low' | 'moderate' | 'high' | 'viral'
        dimension_scores  dict   {quality, engagement, sentiment, reach, virality}
        key_factors       list   up to 6 human-readable factor strings
        reasoning         str    pipe-separated reasoning narrative
        excellence_bonus  float  bonus points awarded for multi-dimension excellence
        transcript_analysis dict full transcript NLP signals
    """

    # ── 1. Compute sentiment if not pre-supplied ──────────────────────────────
    if sentiment_data is None:
        sentiment_data = _compute_sentiment(comments)

    # ── 2. Assemble feature row ───────────────────────────────────────────────
    row_dict = {**video_metadata, **sentiment_data}
    row_dict['has_transcript'] = len(transcript_text) > 0

    df_raw = pd.DataFrame([row_dict])

    # ── 3. Feature engineering ────────────────────────────────────────────────
    df_engineered = _feature_engineer.engineer_features(df_raw)

    # ── 4. Transcript NLP ─────────────────────────────────────────────────────
    transcript_features = _transcript_processor.process_single(transcript_text)
    for key, value in transcript_features.items():
        df_engineered[key] = value

    # ── 5. Refresh transcript-dependent quality signals ───────────────────────
    df_engineered = _feature_engineer.refresh_transcript_quality(df_engineered)

    # ── 6. Score ──────────────────────────────────────────────────────────────
    impact_result = _impact_model.predict_impact(df_engineered.iloc[0])

    return {
        **impact_result,
        'transcript_analysis': transcript_features,
    }


def run_impact_model_from_row(engineered_row: pd.Series) -> dict:
    """
    Score a pre-engineered pd.Series row directly.

    Use this variant when the root fetcher has already run feature engineering
    and transcript processing, and just needs the final impact score.

    Parameters
    ----------
    engineered_row : pd.Series
        A fully engineered row that already contains all feature columns
        (output of ImpactFeatureEngineering.engineer_features() +
         refresh_transcript_quality() + TranscriptProcessor.process_single()).

    Returns
    -------
    dict — same schema as run_impact_model().
    """
    return _impact_model.predict_impact(engineered_row)


def fit_model_on_dataset(df: pd.DataFrame) -> None:
    """
    (Optional) Replace population-level benchmarks with dataset-specific
    percentiles. Call this once at startup if you have a training dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered DataFrame (output of engineer_features()).
    """
    _impact_model.fit(df)
    logger.info("Impact model re-fitted on provided dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sentiment(comments: list) -> dict:
    """
    Compute sentiment signals from a list of comment strings.
    Returns a dict matching the keys expected by the feature engineer.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    if not comments:
        return {}

    analyzer = SentimentIntensityAnalyzer()
    sentiments, polarities, subjectivities = [], [], []
    positive_w = negative_w = neutral_w = total_w = 0.0

    for c in comments:
        if not isinstance(c, str) or not c.strip():
            continue
        word_len     = len(c.split())
        depth_weight = float(np.clip(np.log1p(word_len) / np.log1p(15), 0.25, 1.0))

        vs = analyzer.polarity_scores(c)
        sentiments.append(vs['compound'] * depth_weight)
        try:
            bl = TextBlob(c)
            polarities.append(bl.sentiment.polarity * depth_weight)
            subjectivities.append(bl.sentiment.subjectivity * depth_weight)
        except Exception:
            pass

        if   vs['compound'] >= 0.05:  positive_w += depth_weight
        elif vs['compound'] <= -0.05: negative_w += depth_weight
        else:                         neutral_w  += depth_weight
        total_w += depth_weight

    if total_w == 0:
        return {}

    raw_compounds = [
        analyzer.polarity_scores(c)['compound']
        for c in comments if isinstance(c, str) and c.strip()
    ]
    sent_variance = float(np.var(raw_compounds)) if raw_compounds else 0.0

    return {
        'avg_sentiment':          sum(sentiments)    / total_w,
        'sentiment_polarity':     sum(polarities)    / total_w if polarities     else 0.0,
        'sentiment_subjectivity': sum(subjectivities)/ total_w if subjectivities else 0.0,
        'positive_ratio':         positive_w / total_w,
        'negative_ratio':         negative_w / total_w,
        'neutral_ratio':          neutral_w  / total_w,
        'sentiment_variance':     sent_variance,
        'comments_extracted':     len(comments),
        'all_comments':           str(comments),
    }
