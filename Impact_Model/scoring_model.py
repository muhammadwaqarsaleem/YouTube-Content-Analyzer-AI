"""
YouTube Impact Score Model (v9) — pure scoring, no data fetching.
Expects a fully engineered pd.Series row as input to predict_impact().
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from Impact_Model.config import FEATURE_WEIGHTS, IMPACT_THRESHOLDS

logger = logging.getLogger(__name__)


class YouTubeImpactModel:
    """
    Scores a single engineered video row across five dimensions:
      quality (0.30) | engagement (0.25) | sentiment (0.20)
      | reach (0.15) | virality (0.10)

    Can be used in:
      - dataset-free mode  → call initialize_population_stats()
      - dataset mode       → call fit(df)
    """

    def __init__(self, weights: Dict = None):
        self.weights               = weights or FEATURE_WEIGHTS
        self.thresholds            = IMPACT_THRESHOLDS
        self.feature_stats         = {}
        self._median_comment_count = 50.0

        self.feature_groups = {
            'engagement': [
                'engagement_rate', 'like_rate', 'comment_rate',
                'quality_engagement',
                'engagement_velocity', 'engagement_consistency', 'engagement_momentum',
            ],
            'sentiment': [
                'approval_score', 'positive_ratio',
                'sentiment_strength', 'avg_sentiment',
                'controversy_score', 'sentiment_polarization',
            ],
            'reach': [
                'log_views',
                'views_per_day', 'views_per_day_log', 'growth_momentum',
                'relative_reach', 'category_performance',
            ],
            'quality': [
                'transcript_quality_score', 'transcript_richness_score',
                'transcript_has_content',
                'title_quality_score', 'description_completeness', 'metadata_quality',
                'optimal_duration',
            ],
            'virality': [
                'log_views',
                'viral_velocity', 'viral_score',
                'shareability',
            ],
        }

    # ── Dataset-free population stats ────────────────────────────────────────
    def initialize_population_stats(self):
        """
        Set pre-computed YouTube-wide percentile benchmarks so the model
        can score individual videos without a training dataset.
        """
        logger.info("Initialising model with population-level benchmarks (dataset-free mode)...")

        def stats(mn, q25, q50, q75, q90, q95, mx, mean=None, std=None):
            return {
                'min':  mn,  'q25': q25, 'q50': q50,
                'q75':  q75, 'q90': q90, 'q95': q95,
                'max':  mx,
                'mean': mean or q50,
                'std':  std  or (q75 - q25),
            }

        self.feature_stats = {
            'engagement': {
                'engagement_rate':        stats(0,    0.3,   1.2,   3.5,   7.0,   12.0,  100),
                'like_rate':              stats(0,    0.8,   2.2,   5.5,   10.0,  16.0,  100),
                'comment_rate':           stats(0,    0.01,  0.05,  0.18,  0.45,  0.80,  10),
                'quality_engagement':     stats(0,    0.4,   1.5,   4.0,   8.0,   14.0,  100),
                'engagement_velocity':    stats(0,    0.5,   5.0,   30.0,  150.0, 500.0, 50000),
                'engagement_consistency': stats(0,    0.10,  0.25,  0.50,  0.75,  0.90,  1),
                'engagement_momentum':    stats(0,    1.0,   10.0,  60.0,  300.0, 1000.0,100000),
            },
            'sentiment': {
                'approval_score':         stats(0,    30.0,  45.0,  60.0,  75.0,  85.0,  100),
                'positive_ratio':         stats(0,    0.05,  0.12,  0.22,  0.35,  0.50,  1.0),
                'sentiment_strength':     stats(0,    0.10,  0.30,  0.55,  0.75,  0.85,  1),
                'avg_sentiment':          stats(-1,   0.00,  0.20,  0.45,  0.65,  0.80,  1),
                'controversy_score':      stats(0,    0.00,  0.02,  0.08,  0.20,  0.40,  10),
                'sentiment_polarization': stats(0,    0.000, 0.001, 0.005, 0.020, 0.050, 1),
            },
            'reach': {
                'log_views':              stats(0,    7.0,   9.2,   11.5,  13.1,  14.2,  20),
                'views_per_day':          stats(0,    5.0,   50.0,  500.0, 3000.0,10000.0,5000000),
                'views_per_day_log':      stats(0,    1.8,   3.9,   6.2,   8.0,   9.2,   15),
                'growth_momentum':        stats(0,    1.0,   10.0,  80.0,  400.0, 1000.0,200000),
                'relative_reach':         stats(0,    0.10,  0.50,  2.00,  8.00,  20.00, 5000),
                'category_performance':   stats(0,    0.30,  0.80,  2.00,  5.00,  10.00, 500),
            },
            'quality': {
                'transcript_quality_score':  stats(0, 0.08,  0.22,  0.42,  0.58,  0.70,  1),
                'transcript_richness_score': stats(0, 0.05,  0.18,  0.38,  0.55,  0.68,  1),
                'transcript_has_content':    stats(0, 0.0,   1.0,   1.0,   1.0,   1.0,   1),
                'title_quality_score':       stats(0, 0.20,  0.40,  0.60,  0.80,  0.90,  1),
                'description_completeness':  stats(0, 0.30,  0.70,  1.00,  1.00,  1.00,  1),
                'metadata_quality':          stats(0, 0.25,  0.50,  0.75,  1.00,  1.00,  1),
                'optimal_duration':          stats(0, 0.0,   0.0,   1.0,   1.0,   1.0,   1),
            },
            'virality': {
                'log_views':     stats(0, 7.0,   9.2,   11.5,  13.1,  14.2,  20),
                'viral_velocity':stats(0, 5.0,   50.0,  500.0, 3000.0,10000.0,5000000),
                'viral_score':   stats(0, 0.01,  0.50,  5.00,  30.0,  100.0, 10000),
                'shareability':  stats(0, 0.50,  2.00,  6.00,  15.0,  30.0,  500),
            },
        }

        self._median_comment_count = 50.0
        logger.info("Population stats initialised — model ready for single-video analysis.")
        return self

    # ── Fitting from a DataFrame (optional, for dataset mode) ────────────────
    def fit(self, df: pd.DataFrame):
        """Fit percentile stats from a training DataFrame (optional)."""
        logger.info("Fitting Impact Score Model from dataset...")
        if 'comment_count' in df.columns:
            non_zero = df['comment_count'][df['comment_count'] > 0]
            self._median_comment_count = float(non_zero.median()) if len(non_zero) > 0 else 50.0

        for group, features in self.feature_groups.items():
            self.feature_stats[group] = {}
            available = [f for f in features if f in df.columns]
            for feat in available:
                data = df[feat].fillna(0)
                self.feature_stats[group][feat] = {
                    'mean': float(data.mean()),   'std':  float(data.std()),
                    'min':  float(data.min()),    'max':  float(data.max()),
                    'q25':  float(data.quantile(0.25)),
                    'q50':  float(data.quantile(0.50)),
                    'q75':  float(data.quantile(0.75)),
                    'q90':  float(data.quantile(0.90)),
                    'q95':  float(data.quantile(0.95)),
                }
        logger.info("Model fitting complete.")
        return self

    # ── Core percentile scorer ────────────────────────────────────────────────
    def calculate_component_score(self, value: float, stats: Dict,
                                  handle_missing: bool = True) -> float:
        if pd.isna(value) and handle_missing:
            return 50.0
        value = float(value)
        if value <= stats['q25']:
            return 25.0 * (value - stats['min']) / (stats['q25'] - stats['min'] + 1e-10)
        elif value <= stats['q50']:
            return 25.0 + 25.0 * (value - stats['q25']) / (stats['q50'] - stats['q25'] + 1e-10)
        elif value <= stats['q75']:
            return 50.0 + 25.0 * (value - stats['q50']) / (stats['q75'] - stats['q50'] + 1e-10)
        elif value <= stats['q95']:
            return 75.0 + 20.0 * (value - stats['q75']) / (stats['q95'] - stats['q75'] + 1e-10)
        else:
            return min(100.0, 95.0 + 5.0 * (value - stats['q95']) /
                       (stats['max'] - stats['q95'] + 1e-10))

    def _sf(self, row: pd.Series, group: str, feat: str,
            fallback: float = 50.0) -> float:
        gs  = self.feature_stats.get(group, {})
        if feat not in gs:
            return fallback
        val = row.get(feat, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return fallback
        return self.calculate_component_score(float(val), gs[feat])

    # ── Dimension scorers ─────────────────────────────────────────────────────

    def _score_engagement(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Rate Quality 45% | Signal Strength 35% | Momentum 20%"""
        g = 'engagement'
        rate_score    = self._sf(row, g, 'like_rate') * 0.55 + self._sf(row, g, 'comment_rate') * 0.45
        signal_score  = (self._sf(row, g, 'engagement_rate')        * 0.65 +
                         self._sf(row, g, 'engagement_consistency')  * 0.20 +
                         self._sf(row, g, 'quality_engagement')      * 0.15)
        momentum_score = (self._sf(row, g, 'engagement_velocity') * 0.50 +
                          self._sf(row, g, 'engagement_momentum')  * 0.50)
        final   = rate_score * 0.45 + signal_score * 0.35 + momentum_score * 0.20
        factors = []
        if rate_score    >= 75: factors.append("High like/comment rate")
        elif rate_score  <= 25: factors.append("Low like/comment rate")
        if signal_score  >= 75: factors.append("High engagement rate")
        elif signal_score <= 25: factors.append("Low engagement rate")
        if momentum_score >= 75: factors.append("High engagement velocity")
        return final, factors

    def _score_sentiment(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Valence 45% (confidence-weighted) | Intensity 25% | Polarization 30%"""
        g = 'sentiment'
        comment_count     = float(row.get('comment_count', 0))
        comments_disabled = float(row.get('comments_disabled', 0))
        if comments_disabled or comment_count == 0:
            confidence = 0.0
        else:
            confidence = min(comment_count / (self._median_comment_count + 1e-10), 1.0)
            confidence = max(confidence, 0.10)

        valence_raw   = self._sf(row, g, 'approval_score') * 0.60 + self._sf(row, g, 'positive_ratio') * 0.40
        valence_score = valence_raw * confidence + 50.0 * (1.0 - confidence)
        intensity_score = (self._sf(row, g, 'sentiment_strength') * 0.55 +
                           self._sf(row, g, 'avg_sentiment')       * 0.45)
        comment_rate   = float(row.get('comment_rate', 0))
        _polar_trusted = comment_rate >= 0.05
        if _polar_trusted:
            polar_score = (self._sf(row, g, 'controversy_score')      * 0.65 +
                           self._sf(row, g, 'sentiment_polarization') * 0.35)
        else:
            polar_score = 50.0

        final   = valence_score * 0.45 + intensity_score * 0.25 + polar_score * 0.30
        factors = []
        conf_label = f" (confidence {confidence:.0%})"
        if valence_score >= 70:  factors.append(f"High audience approval{conf_label}")
        elif valence_score <= 30: factors.append(f"Low/mixed audience approval{conf_label}")
        if not _polar_trusted:   factors.append("Low comment rate — polarization unavailable")
        elif polar_score >= 70:  factors.append("Highly polarizing/debate content")
        if intensity_score >= 75: factors.append("Strong audience sentiment intensity")
        if confidence < 0.20:    factors.append("Low comment volume — sentiment signal limited")
        return final, factors

    def _score_reach(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Absolute Scale 30% | Velocity 30% | Relative Performance 40%"""
        g = 'reach'
        scale_score    = self._sf(row, g, 'log_views')
        velocity_score = (self._sf(row, g, 'views_per_day')   * 0.60 +
                          self._sf(row, g, 'growth_momentum') * 0.40)
        relative_score = (self._sf(row, g, 'relative_reach')       * 0.50 +
                          self._sf(row, g, 'category_performance') * 0.50)
        final   = scale_score * 0.30 + velocity_score * 0.30 + relative_score * 0.40
        factors = []
        if scale_score    >= 75: factors.append("High lifetime view count")
        elif scale_score  <= 25: factors.append("Low total views")
        if relative_score >= 75: factors.append("High category performance")
        elif relative_score <= 25: factors.append("Low relative reach")
        if velocity_score >= 75: factors.append("High view velocity")
        return final, factors

    def _score_quality(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Transcript Depth 80% | Metadata & Title 15% | Structural 5%"""
        g = 'quality'
        has_transcript = float(row.get('transcript_has_content', 0))
        if has_transcript:
            transcript_score = (self._sf(row, g, 'transcript_quality_score')  * 0.65 +
                                self._sf(row, g, 'transcript_richness_score') * 0.35)
        else:
            transcript_score = 20.0

        title_score    = self._sf(row, g, 'title_quality_score')
        desc_score     = self._sf(row, g, 'description_completeness')
        meta_score     = self._sf(row, g, 'metadata_quality')
        metadata_score = title_score * 0.45 + desc_score * 0.25 + meta_score * 0.30

        structural_score = (self._sf(row, g, 'optimal_duration') * 0.40 +
                            self._sf(row, g, 'metadata_quality') * 0.60)

        final   = transcript_score * 0.80 + metadata_score * 0.15 + structural_score * 0.05
        factors = []
        if not has_transcript:
            factors.append("No transcript — quality sub-score at baseline")
        elif transcript_score >= 70:
            factors.append("High transcript quality")
        elif transcript_score <= 30:
            factors.append("Low transcript depth")
        if metadata_score   >= 75: factors.append("Strong title/metadata quality")
        if structural_score >= 75: factors.append("Strong structural quality")
        return final, factors

    def _score_virality(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Historical Lift 40% | Current Velocity 35% | Shareability 25%"""
        reach_gs = self.feature_stats.get('reach', {})
        if 'log_views' in reach_gs:
            historical_score = self.calculate_component_score(
                float(row.get('log_views', 0)), reach_gs['log_views']
            )
        else:
            historical_score = 50.0
        g = 'virality'
        velocity_score     = (self._sf(row, g, 'viral_velocity') * 0.60 +
                              self._sf(row, g, 'viral_score')    * 0.40)
        shareability_score = self._sf(row, g, 'shareability')
        final   = historical_score * 0.40 + velocity_score * 0.35 + shareability_score * 0.25
        factors = []
        if historical_score >= 75:   factors.append("High lifetime viral reach")
        elif historical_score <= 25: factors.append("Low historical reach")
        if velocity_score >= 75:     factors.append("High viral velocity")
        elif velocity_score <= 25:   factors.append("Low viral velocity")
        if shareability_score >= 75: factors.append("High shareability")
        return final, factors

    def calculate_group_score(self, row: pd.Series, group: str) -> Tuple[float, List[str]]:
        dispatch = {
            'engagement': self._score_engagement,
            'sentiment':  self._score_sentiment,
            'reach':      self._score_reach,
            'quality':    self._score_quality,
            'virality':   self._score_virality,
        }
        return dispatch[group](row) if group in dispatch else (50.0, [])

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_impact(self, row: pd.Series) -> Dict:
        scores, all_factors = {}, []
        for group in self.feature_groups:
            score, factors = self.calculate_group_score(row, group)
            scores[group]  = float(np.clip(score, 0, 100))
            all_factors.extend(factors)

        linear_score     = sum(scores.get(g, 50) * w for g, w in self.weights.items())
        high_dim_count   = sum(1 for s in scores.values() if s >= 70)
        excellence_bonus = high_dim_count * 2.0
        final_score      = float(np.clip(linear_score + excellence_bonus, 0, 100))

        return {
            'impact_score':     round(final_score, 2),
            'impact_level':     self._get_impact_level(final_score),
            'dimension_scores': {k: round(v, 2) for k, v in scores.items()},
            'key_factors':      list(dict.fromkeys(all_factors))[:6],
            'reasoning':        self._generate_reasoning(scores, all_factors, row),
            'excellence_bonus': round(excellence_bonus, 2),
        }

    def _get_impact_level(self, score: float) -> str:
        for level, (lo, hi) in self.thresholds.items():
            if lo <= score < hi:
                return level
        return 'viral' if score >= 80 else 'minimal'

    def _generate_reasoning(self, scores: Dict, factors: List[str], row: pd.Series) -> str:
        reasons = []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_dim,  top_score  = sorted_scores[0]
        weak_dim, weak_score = sorted_scores[-1]
        if top_score  >= 70: reasons.append(f"Excels in {top_dim} ({top_score:.1f}/100)")
        if weak_score < 40:  reasons.append(f"Limited {weak_dim} ({weak_score:.1f}/100)")

        views = row.get('view_count', 0)
        if   views > 1_000_000: reasons.append(f"Exceptional reach: {views:,.0f} views")
        elif views > 100_000:   reasons.append(f"Good reach: {views:,.0f} views")
        elif views > 10_000:    reasons.append(f"Moderate reach: {views:,.0f} views")

        er = row.get('engagement_rate', 0)
        if er > 5:   reasons.append(f"Strong engagement rate: {er:.2f}%")
        elif er > 2: reasons.append(f"Average engagement rate: {er:.2f}%")

        ap = row.get('approval_score', 0)
        cc = row.get('comment_count', 0)
        if cc > 0:
            if ap > 80:  reasons.append("Highly positive audience sentiment")
            elif ap < 20: reasons.append("Mixed or negative audience response")

        vpd = row.get('views_per_day', 0)
        if vpd > 10_000:  reasons.append(f"High growth velocity ({vpd:,.0f} views/day)")
        elif vpd > 1_000: reasons.append(f"Steady growth ({vpd:,.0f} views/day)")

        tqs = row.get('transcript_quality_score', 0)
        wc  = row.get('transcript_word_count',    0)
        vr  = row.get('transcript_vocab_richness', 0)
        if row.get('transcript_has_content', 0) == 1:
            if tqs >= 0.60:
                reasons.append(f"High-quality transcript (score {tqs:.2f}): "
                                f"{wc:,.0f} words, vocab richness {vr:.2f}")
            elif tqs >= 0.35:
                reasons.append(f"Moderate transcript quality (score {tqs:.2f}): {wc:,.0f} words")
            else:
                reasons.append(f"Low transcript depth (score {tqs:.2f}): {wc:,.0f} words")
        else:
            reasons.append("No transcript — quality signal limited")

        if row.get('comments_disabled', 0) == 1:
            reasons.append("Comments disabled — sentiment confidence = 0")

        return " | ".join(reasons) if reasons else "Balanced performance across all dimensions"
