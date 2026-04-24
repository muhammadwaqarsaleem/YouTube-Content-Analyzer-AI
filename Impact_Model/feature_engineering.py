"""
Feature engineering for the YouTube Impact Score Model (v9).
No data fetching — expects a pre-built DataFrame row.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class ImpactFeatureEngineering:
    """
    Feature engineering for YouTube impact analysis.
    Handles missing data gracefully and creates robust features.
    """

    def __init__(self):
        self.scaler = RobustScaler()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for impact modelling."""
        logger.info("Starting feature engineering...")
        df = df.copy()
        df = self._create_engagement_features(df)
        df = self._create_sentiment_features(df)
        df = self._create_reach_features(df)
        df = self._create_quality_features(df)
        df = self._create_virality_features(df)
        df = self._enhance_temporal_features(df)
        logger.info(f"Feature engineering complete. Total features: {df.shape[1]}")
        return df

    def refresh_transcript_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute transcript-dependent quality signals AFTER transcript NLP
        columns are available.
        """
        logger.info("Refreshing quality features with transcript signals...")

        word_count_norm = np.clip(df.get('transcript_word_count',     0).fillna(0) / 2000, 0, 1)
        vocab_richness  = df.get('transcript_vocab_richness',          0).fillna(0)
        sent_len_norm   = np.clip(df.get('transcript_avg_sentence_len', 0).fillna(0) / 20,  0, 1)
        has_content     = df.get('transcript_has_content',             0).fillna(0)
        max_nonzero     = df.get('tfidf_nonzero_count', pd.Series([1])).max()
        tfidf_diversity = np.clip(df.get('tfidf_nonzero_count', 0).fillna(0) / (max_nonzero + 1e-10), 0, 1)

        df['transcript_richness_score'] = np.clip(
            word_count_norm * 0.35 +
            vocab_richness  * 0.35 +
            sent_len_norm   * 0.12 +
            has_content     * 0.18,
            0, 1
        )

        if 'transcript_quality_score' not in df.columns or (df['transcript_quality_score'] == 0).all():
            df['transcript_quality_score'] = df['transcript_richness_score']

        df['content_richness'] = (
            df['title_quality_score']      * 0.45 +
            df['description_completeness'] * 0.25 +
            df['metadata_quality']         * 0.30
        )

        logger.info("Quality features refreshed.")
        return df

    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating engagement features...")
        df['view_count']    = pd.to_numeric(df['view_count'],    errors='coerce').fillna(0)
        df['like_count']    = pd.to_numeric(df['like_count'],    errors='coerce').fillna(0)
        df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce').fillna(0)

        if 'like_rate' not in df.columns or df['like_rate'].isna().any():
            df['like_rate'] = np.where(df['view_count'] > 0,
                                       (df['like_count'] / df['view_count']) * 100, 0)
        if 'comment_rate' not in df.columns or df['comment_rate'].isna().any():
            df['comment_rate'] = np.where(df['view_count'] > 0,
                                          (df['comment_count'] / df['view_count']) * 100, 0)
        if 'engagement_rate' not in df.columns or df['engagement_rate'].isna().any():
            df['total_engagement'] = df['like_count'] + df['comment_count']
            df['engagement_rate']  = np.where(df['view_count'] > 0,
                                              (df['total_engagement'] / df['view_count']) * 100, 0)

        df['engagement_velocity']    = (df.get('likes_per_day', 0) + df.get('comments_per_day', 0))
        df['log_views']              = np.log1p(df['view_count'])
        df['log_likes']              = np.log1p(df['like_count'])
        df['log_comments']           = np.log1p(df['comment_count'])
        df['engagement_consistency'] = (
            (df['like_rate']    / (df['like_rate'].max()    + 1e-10)) * 0.5 +
            (df['comment_rate'] / (df['comment_rate'].max() + 1e-10)) * 0.5
        )
        return df

    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating sentiment features...")
        for col in ['avg_sentiment', 'sentiment_polarity', 'sentiment_subjectivity',
                    'positive_ratio', 'negative_ratio', 'neutral_ratio', 'sentiment_variance']:
            if col not in df.columns or df[col].isna().all():
                df[col] = 0.0

        df['sentiment_strength']  = np.abs(df['sentiment_polarity'])
        df['approval_score'] = (df['positive_ratio'] / (df['positive_ratio'] + df['negative_ratio'] + 1e-10)) * 100
        df['comments_disabled']   = (df['comment_count'] == 0).astype(int)
        _has_volume               = df['comment_rate'] >= 0.05
        df['controversy_score']   = np.where(
            _has_volume,
            df['comment_rate'] * (1 - np.abs(df['positive_ratio'] - df['negative_ratio'])),
            0.0
        )
        df['sentiment_polarization'] = np.where(
            _has_volume,
            df['sentiment_variance'] * df['comment_rate'],
            0.0
        )
        return df

    def _create_reach_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating reach features...")
        df['reach_score']    = np.log1p(df['view_count'])
        avg_views            = df['view_count'].mean()
        df['relative_reach'] = df['view_count'] / (avg_views + 1e-10)
        df['growth_momentum'] = np.where(
            df.get('video_age_days', 1) > 0,
            df.get('views_per_day', 0) / np.sqrt(df.get('video_age_days', 1)),
            df.get('views_per_day', 0)
        )
        df['viewership_sustainability'] = np.where(
            df.get('video_age_days', 1) > 7, df.get('views_per_day', 0), 0
        )
        if 'category_name' in df.columns:
            category_avg = df.groupby('category_name')['view_count'].transform('mean')
            df['category_performance'] = df['view_count'] / (category_avg + 1e-10)
        else:
            df['category_performance'] = 1.0
        return df

    def _create_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating baseline quality features...")
        if 'duration_minutes' not in df.columns:
            df['duration_minutes'] = df.get('duration_seconds', 0) / 60

        df['optimal_duration'] = (
            (df['duration_minutes'] >= 8) & (df['duration_minutes'] <= 15)
        ).astype(int)

        df['title_quality_score'] = (
            (df.get('title_length',         0) >= 30).astype(int) * 0.3 +
            (df.get('title_word_count',     0) >= 5 ).astype(int) * 0.3 +
            df.get('title_has_question',    0) * 0.2 +
            df.get('title_has_exclamation', 0) * 0.2
        )
        df['description_completeness'] = np.where(
            df.get('description_length', 0) > 100, 1,
            df.get('description_length', 0) / 100
        )
        df['metadata_quality'] = (
            df.get('has_description', 0) * 0.25 +
            df.get('has_transcript',  0).astype(int) * 0.25 +
            (df.get('tag_count',              0) > 0 ).astype(int) * 0.25 +
            (df.get('description_word_count', 0) > 10).astype(int) * 0.25
        )
        df['transcript_richness_score'] = 0.0
        df['transcript_quality_score']  = 0.0

        df['content_richness'] = (
            df['title_quality_score']      * 0.45 +
            df['description_completeness'] * 0.25 +
            df['metadata_quality']         * 0.30
        )
        return df

    def _create_virality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating virality features...")
        if 'viral_score' not in df.columns or df['viral_score'].isna().any():
            df['viral_score'] = (
                np.log1p(df.get('views_per_day', 0)) *
                df.get('engagement_rate', 0) *
                (1 + df.get('quality_engagement', 0))
            )
        df['exponential_growth'] = np.where(
            df.get('video_age_days', 1) > 0,
            df['view_count'] / (df.get('video_age_days', 1) ** 0.5), 0
        )
        df['shareability'] = (
            df.get('like_rate', 0)         * 0.4 +
            df.get('comment_rate', 0)      * 0.3 +
            df.get('discussion_ratio', 0) * 100 * 0.3
        )
        df['viral_velocity'] = np.where(
            df.get('video_age_days', 1) <= 7,
            df.get('views_per_day', 0) * 2, df.get('views_per_day', 0)
        )
        df['engagement_momentum'] = (
            df.get('likes_per_day', 0) + df.get('comments_per_day', 0) * 2
        )
        return df

    def _enhance_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Enhancing temporal features...")
        df['freshness_score'] = np.exp(-df.get('video_age_days', 0) / 365)
        season_boost = {'Winter': 1.0, 'Spring': 1.1, 'Summer': 1.2, 'Fall': 1.05}
        df['seasonal_boost'] = df.get('publish_season', 'Winter').map(season_boost).fillna(1.0)
        df['timing_score']   = 1.0 + df.get('is_weekend', 0) * 0.1
        return df
