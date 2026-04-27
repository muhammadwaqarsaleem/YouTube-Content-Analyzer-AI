# Suppress TF / oneDNN / Keras noise before any heavy imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import re
import sys
import warnings
import concurrent.futures
warnings.filterwarnings('ignore')

# Force UTF-8 encoding for standard output to prevent emoji crashes on Windows
sys.stdout.reconfigure(encoding='utf-8')

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

# Silence absl/tf loggers
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Adjust sys.path to allow imports from subdirectories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import yt_dlp
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import models
from Age_Classification_Model.age_classification_main import AgeBracketModel
from Clickbait_Model.harm_detector_Main import run_harm_detector
from Impact_Model.impact_model_main import run_impact_model
from Category_Violence_Model.violence_analyzer_main import ViolenceSpecsManager
from Category_Violence_Model.category_classification_main import ModelSpecsManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeVideoFetcher:
    """
    Data fetcher for YouTube URLs.
    Extracts metadata, comments, and transcript to be passed to various ML models.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_video_id(self, url: str) -> str:
        patterns = [
            r'(?:youtube\.com/watch\?v=)([^&]+)',
            r'(?:youtu\.be/)([^?]+)',
            r'(?:youtube\.com/embed/)([^?]+)',
            r'(?:youtube\.com/v/)([^?]+)'
        ]
        for p in patterns:
            m = re.search(p, url)
            if m: return m.group(1)
        return url

    def _parse_duration(self, d: str) -> int:
        m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d)
        if not m: return 0
        return int(m.group(1) or 0)*3600 + int(m.group(2) or 0)*60 + int(m.group(3) or 0)

    def fetch_video_metadata(self, video_id: str) -> Dict:
        logger.info(f"Fetching metadata for video: {video_id}")
        resp = self.youtube.videos().list(
            part='snippet,statistics,contentDetails', id=video_id
        ).execute()
        if not resp['items']:
            raise ValueError(f"Video not found: {video_id}")

        v  = resp['items'][0]
        sn = v['snippet']; st = v['statistics']; cd = v['contentDetails']
        dur   = self._parse_duration(cd['duration'])
        pub   = pd.to_datetime(sn['publishedAt'], utc=True)
        age   = (pd.Timestamp.now(tz='UTC') - pub).days
        dur_m = dur / 60

        if   dur_m < 1:  dcat = 'short'
        elif dur_m < 5:  dcat = 'medium'
        elif dur_m < 15: dcat = 'standard'
        elif dur_m < 30: dcat = 'long'
        else:            dcat = 'extended'

        md = {
            'title':                  sn['title'],
            'description':            sn.get('description', ''),
            'published_at':           pub.timestamp(),
            'published_date':         pub.strftime('%Y-%m-%d'),
            'channel_title':          sn['channelTitle'],
            'channel_id':             sn['channelId'],
            'category_name':          sn.get('categoryId', 'Unknown'),
            'category_id':            float(sn.get('categoryId', 0)),
            'view_count':             int(st.get('viewCount',    0)),
            'like_count':             float(st.get('likeCount',  0)),
            'comment_count':          float(st.get('commentCount', 0)),
            'duration_seconds':       float(dur),
            'duration_minutes':       dur_m,
            'duration_category':      dcat,
            'tags':                   ','.join(sn.get('tags', [])),
            'tags_list':              sn.get('tags', []),
            'tag_count':              len(sn.get('tags', [])),
            'video_age_days':         age,
            'publish_year':           pub.year,
            'publish_month':          pub.month,
            'publish_day_of_week':    pub.dayofweek,
            'publish_quarter':        (pub.month - 1) // 3 + 1,
            'is_weekend':             int(pub.dayofweek >= 5),
            'default_language':       sn.get('defaultLanguage', 'en'),
            'title_length':           len(sn['title']),
            'title_word_count':       len(sn['title'].split()),
            'title_has_question':     int('?' in sn['title']),
            'title_has_exclamation':  int('!' in sn['title']),
            'description_length':     len(sn.get('description', '')),
            'description_word_count': len(sn.get('description', '').split()),
            'has_description':        int(len(sn.get('description', '')) > 0),
        }
        d = max(age, 1)
        md['views_per_day']        = md['view_count']    / d
        md['likes_per_day']        = md['like_count']    / d
        md['comments_per_day']     = md['comment_count'] / d
        md['views_per_day_log']    = np.log1p(md['views_per_day'])
        md['likes_per_day_log']    = np.log1p(md['likes_per_day'])
        md['comments_per_day_log'] = np.log1p(md['comments_per_day'])
        mo = pub.month
        md['publish_season'] = ('Winter' if mo in [12,1,2] else
                                'Spring' if mo in [3,4,5]  else
                                'Summer' if mo in [6,7,8]  else 'Fall')
        logger.info("Metadata fetched successfully")
        return md

    def fetch_comments(self, video_id: str, max_comments: int = 100) -> List[str]:
        logger.info(f"Fetching comments for video: {video_id}")
        comments = []
        try:
            resp = self.youtube.commentThreads().list(
                part='snippet', videoId=video_id,
                maxResults=min(max_comments, 100), textFormat='plainText'
            ).execute()
            for item in resp['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            logger.info(f"Fetched {len(comments)} comments")
        except Exception as e:
            logger.warning(f"Could not fetch comments: {e}")
        return comments

    def fetch_transcript(self, video_id: str) -> str:
        """Multi-strategy transcript fetcher."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Fetching transcript for: {video_id}")

        try:
            lang_priority = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            entries = YouTubeTranscriptApi.get_transcript(video_id, languages=lang_priority)
            text = ' '.join(
                e.get('text', '') if isinstance(e, dict) else getattr(e, 'text', str(e))
                for e in entries
            ).strip()
            if text:
                logger.info(f"Transcript fetched (strategy 1): {len(text):,} chars")
                return text
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            def _priority(t):
                if not t.is_generated and t.language_code.startswith('en'): return 0
                if t.is_generated  and t.language_code.startswith('en'):    return 1
                if not t.is_generated:                                       return 2
                return 3
            for t_obj in sorted(list(transcript_list), key=_priority):
                try:
                    entries = t_obj.fetch()
                    text = ' '.join(
                        e.get('text', '') if isinstance(e, dict) else getattr(e, 'text', str(e))
                        for e in entries
                    ).strip()
                    if text:
                        logger.info(f"Transcript fetched (strategy 2): {len(text):,} chars")
                        return text
                except: continue
        except Exception as e:
            logger.debug(f"Strategy 2 failed: {e}")

        try:
            import tempfile, glob as _glob
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    'skip_download': True, 'writeautomaticsub': True,
                    'writesubtitles': True, 'subtitleslangs': ['en', 'en-orig'],
                    'subtitlesformat': 'vtt',
                    'outtmpl': os.path.join(tmpdir, '%(id)s.%(ext)s'),
                    'quiet': True, 'no_warnings': True,
                    'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                vtt_files = _glob.glob(os.path.join(tmpdir, '*.vtt'))
                if vtt_files:
                    with open(vtt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read()
                    text = re.sub(r'<[^>]+>', '', raw).strip() 
                    if text:
                        logger.info(f"Transcript fetched (strategy 3): {len(text):,} chars")
                        return text
        except Exception as e:
            logger.debug(f"Strategy 3 failed: {e}")

        logger.warning(f"All transcript strategies failed for: {video_id}")
        return ''


class GlobalOrchestrator:
    def __init__(self, api_key: str):
        self.fetcher = YouTubeVideoFetcher(api_key=api_key)
        logger.info("Initializing models...")
        self.age_model = AgeBracketModel()
        self.violence_manager = ViolenceSpecsManager()
        self.category_manager = ModelSpecsManager()
        logger.info("Initialization complete.")

    def _run_pipeline(self, url: str):
        """Shared core pipeline: fetches data and runs all models. Returns raw results tuple."""
        video_id = self.fetcher.extract_video_id(url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Phase 1: Start models that do their own fetching immediately
            logger.info("Background: Starting Violence and Category models...")
            future_violence = executor.submit(self.violence_manager.analyze_violence, url, True)
            future_category = executor.submit(self.category_manager.analyze_video, url, True)

            # Phase 2: Fetch data required for the other models
            logger.info("--- Data Fetching (Sequential for Text Models) ---")
            metadata = self.fetcher.fetch_video_metadata(video_id)
            comments = self.fetcher.fetch_comments(video_id)
            transcript = self.fetcher.fetch_transcript(video_id)

            title = metadata.get('title', '')
            description = metadata.get('description', '')

            # Phase 3: Submit text-based models now that data is ready
            logger.info("--- Model Inference (Text Models) ---")
            future_age = executor.submit(self.age_model.predict_age, transcript_text=transcript)
            future_harm = executor.submit(run_harm_detector, title=title, description=description, transcript=transcript)
            future_impact = executor.submit(run_impact_model, video_metadata=metadata, comments=comments, transcript_text=transcript)

            # Phase 4: Wait for everything to complete
            logger.info("Awaiting all results...")
            age_result = future_age.result()
            harm_result = future_harm.result()
            impact_result = future_impact.result()
            violence_out = future_violence.result()
            category_out = future_category.result()

        return metadata, title, age_result, harm_result, impact_result, violence_out, category_out

    def analyze(self, url: str):
        logger.info(f"=== Starting Global Analysis for {url} ===")
        try:
            metadata, title, age_result, harm_result, impact_result, violence_out, category_out = self._run_pipeline(url)
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return
        # Present Results in CLI
        self.present_results(url, title, age_result, harm_result, impact_result, violence_out, category_out)

    def get_results_as_dict(self, url: str) -> dict:
        """
        Runs the full analysis pipeline and returns a JSON-serializable dict.
        Used by the Flask web server. The CLI path (analyze/present_results) is unchanged.
        """
        logger.info(f"=== Starting Web Analysis for {url} ===")
        metadata, title, age_result, harm_result, impact_result, violence_out, category_out = self._run_pipeline(url)

        # ── Video summary ──────────────────────────────────────────────────
        dur_s = int(metadata.get('duration_seconds', 0))
        dur_str = f"{dur_s // 3600:02d}:{(dur_s % 3600) // 60:02d}:{dur_s % 60:02d}" if dur_s >= 3600 \
                  else f"{dur_s // 60:02d}:{dur_s % 60:02d}"

        video_id = self.fetcher.extract_video_id(url)
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        video_section = {
            "title":          title,
            "channel":        metadata.get('channel_title', ''),
            "published_date": metadata.get('published_date', ''),
            "view_count":     int(metadata.get('view_count', 0)),
            "like_count":     int(metadata.get('like_count', 0)),
            "comment_count":  int(metadata.get('comment_count', 0)),
            "duration":       dur_str,
            "category_id":    metadata.get('category_name', ''),
            "thumbnail_url":  thumbnail_url,
            "url":            url,
        }

        # ── Age classification ─────────────────────────────────────────────
        if age_result:
            age_section = {
                "label":       getattr(age_result, 'predicted_label', 'Unknown'),
                "confidence":  round(float(getattr(age_result, 'confidence', 0.0)), 4),
                "probabilities": {
                    k: round(float(v), 4)
                    for k, v in (getattr(age_result, 'probabilities', {}) or {}).items()
                },
                "num_chunks":  getattr(age_result, 'num_chunks_used', 0),
                "top3_chunks": [
                    {
                        "rank":      c.get('rank', 0),
                        "chunk_idx": c.get('chunk_idx', 0),
                        "weight":    round(float(c.get('weight', 0.0)), 4),
                        "text":      c.get('text', '')[:200],
                    }
                    for c in (getattr(age_result, 'top3_chunks', []) or [])
                ],
            }
        else:
            age_section = None

        # ── Harm / clickbait detection ─────────────────────────────────────
        if harm_result:
            harm_section = {
                "label":         harm_result.get('label_name', 'Unknown'),
                "confidence":    round(float(harm_result.get('confidence', 0.0)), 4),
                "probabilities": {
                    k: round(float(v), 4)
                    for k, v in (harm_result.get('probabilities', {}) or {}).items()
                },
            }
        else:
            harm_section = None

        # ── Impact score ───────────────────────────────────────────────────
        if impact_result:
            ta = impact_result.get('transcript_analysis', {}) or {}
            impact_section = {
                "score":       round(float(impact_result.get('impact_score', 0)), 2),
                "level":       impact_result.get('impact_level', 'unknown').upper(),
                "dimension_scores": {
                    k: round(float(v), 2)
                    for k, v in (impact_result.get('dimension_scores', {}) or {}).items()
                },
                "key_factors":      impact_result.get('key_factors', []),
                "excellence_bonus": round(float(impact_result.get('excellence_bonus', 0.0)), 2),
                "reasoning":        impact_result.get('reasoning', ''),
                "transcript_analysis": {
                    "has_content":        bool(ta.get('transcript_has_content', 0)),
                    "word_count":         int(ta.get('transcript_word_count', 0)),
                    "vocab_richness":      round(float(ta.get('transcript_vocab_richness', 0)), 3),
                    "avg_sentence_len":    round(float(ta.get('transcript_avg_sentence_len', 0)), 1),
                    "sentiment_score":     round(float(ta.get('transcript_sentiment_score', 0)), 3),
                    "semantic_similarity": round(float(ta.get('semantic_similarity', 0)), 3),
                    "fk_grade_norm":       round(float(ta.get('fk_grade_norm', 0)), 3),
                    "mtld_norm":           round(float(ta.get('mtld_norm', 0)), 3),
                    "filler_penalty":      round(float(ta.get('filler_penalty', 0)), 3),
                    "linguistic_score":    round(float(ta.get('linguistic_score', 0)), 3),
                    "quality_score":       round(float(ta.get('transcript_quality_score', 0)), 3),
                },
            }
        else:
            impact_section = None

        # ── Category classification ────────────────────────────────────────
        if category_out and category_out[0]:
            cat_res, _ = category_out
            category_section = {
                "primary":             cat_res.get('primary_category', 'Unknown'),
                "primary_probability": round(float(cat_res.get('primary_probability', 0.0)), 4),
                "all_categories": [
                    {"category": c['category'], "probability": round(float(c['probability']), 4)}
                    for c in (cat_res.get('all_categories', []) or [])[:10]
                ],
            }
        else:
            category_section = None

        # ── Violence classification ────────────────────────────────────────
        if violence_out and violence_out[0]:
            v_res, _ = violence_out
            category_section_viol = {
                "is_violent":          bool(v_res.get('is_violent', False)),
                "severity":            v_res.get('severity', 'NONE'),
                "violence_percentage": round(float(v_res.get('violence_percentage', 0.0)), 2),
                "violent_frame_count": int(v_res.get('violent_frame_count', 0)),
                "total_frames":        int(v_res.get('total_frames', 0)),
                "peak_confidence":     round(float(v_res.get('max_confidence', 0.0)), 4),
                "tier_used":           int(v_res.get('tier_used', 0)),
                "violent_timestamps":  [
                    round(float(ts), 1)
                    for ts in sorted(set(v_res.get('violent_frame_timestamps', []) or []))
                ][:20],
                "recommendation":      v_res.get('recommendation', ''),
            }
        else:
            category_section_viol = None

        return {
            "video":              video_section,
            "age_classification": age_section,
            "harm_detection":     harm_section,
            "impact":             impact_section,
            "category":           category_section,
            "violence":           category_section_viol,
        }

    def present_results(self, url, title, age_result, harm_result, impact_result, violence_out, category_out):
        W = 80

        def bar(value, width=20):
            n = int(round(min(value, 100) / 100 * width))
            return '█' * n + '░' * (width - n)

        def pbar(prob, width=28):
            n = int(round(min(prob, 1.0) * width))
            return '█' * n + '░' * (width - n)

        def score_emoji(s):
            if s >= 80: return '🟣 EXCELLENT'
            if s >= 70: return '🟢 GOOD'
            if s >= 55: return '🟡 AVERAGE'
            return '🔴 LOW'

        # ══ HEADER ══════════════════════════════════════════════════════════
        print('\n' + '═'*W)
        print(f'  🌐  GLOBAL ANALYSIS REPORT FOR: {title}')
        print('═'*W)

        # ══ 1. AGE CLASSIFICATION ══════════════════════════════════════════
        print('\n' + '─'*W)
        print('  🔞  AGE CLASSIFICATION  (Hierarchical RoBERTa)')
        print('─'*W)
        if age_result:
            label    = getattr(age_result, 'predicted_label', 'Unknown')
            conf     = getattr(age_result, 'confidence', 0.0)
            probs    = getattr(age_result, 'probabilities', {})
            n_chunks = getattr(age_result, 'num_chunks_used', '?')
            top3     = getattr(age_result, 'top3_chunks', [])

            icons = {'General': '🟢', 'Teen': '🔵', 'Mature': '🔴'}
            print(f'  Verdict          : {icons.get(label, "⚪")}  {label}  ({conf:.1%} confidence)')
            print(f'  Chunks Analyzed  : {n_chunks}  (sliding-window 512-token chunks)')

            if probs:
                print(f'\n  Class Probabilities:')
                for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                    marker = '◀ predicted' if cls == label else ''
                    print(f'    {cls:<10}  {pbar(p)}  {p:5.1%}  {marker}')

            if top3:
                print(f'\n  Top Contributing Transcript Segments (XAI Evidence):')
                medals = ['🥇', '🥈', '🥉']
                for chunk in top3:
                    rank = chunk.get('rank', 0)
                    cidx = chunk.get('chunk_idx', 0)
                    wt   = chunk.get('weight', 0.0)
                    text = chunk.get('text', '')[:160].replace('\n', ' ')
                    medal = medals[rank-1] if 1 <= rank <= 3 else '  '
                    print(f'    {medal} Chunk {cidx:02d}  [weight={wt:.4f}]')
                    print(f'       "{text}…"')
        else:
            print('  ⚠️  Not enough data or empty transcript — no output.')

        # ══ 2. CLICKBAIT / HARM DETECTION ═════════════════════════════════
        print('\n' + '─'*W)
        print('  🎣  CLICKBAIT & HARM DETECTION  (RoBERTa 7-class)')
        print('─'*W)
        if harm_result:
            h_label = harm_result.get('label_name', 'Unknown')
            h_conf  = harm_result.get('confidence', 0.0)
            h_probs = harm_result.get('probabilities', {})

            danger_icons = {
                'Harmless':      '✅', 'Clickbait':     '🎣',
                'Info Harm':     '⚠️', 'Physical Harm': '🩹',
                'Addiction':     '💊', 'Sexual':        '🔞',
                'Hate/Harass':   '🚫',
            }
            icon = danger_icons.get(h_label, '❓')
            print(f'  Verdict          : {icon}  {h_label}  ({h_conf:.1%} confidence)')

            if h_probs:
                print(f'\n  Risk Breakdown Across All Categories:')
                for cls, p in sorted(h_probs.items(), key=lambda x: -x[1]):
                    marker = '◀ FLAGGED' if cls == h_label else ''
                    flag   = danger_icons.get(cls, '  ')
                    print(f'    {flag} {cls:<16}  {pbar(p, 20)}  {p:5.1%}  {marker}')
        else:
            print('  ⚠️  No output from harm detector.')

        # ══ 3. IMPACT SCORE ══════════════════════════════════════════════
        print('\n' + '─'*W)
        print('  📊  YOUTUBE VIDEO IMPACT ANALYSIS  (v9)')
        print('─'*W)
        if impact_result:
            score     = impact_result.get('impact_score', 0)
            level     = impact_result.get('impact_level', 'unknown').upper()
            dims      = impact_result.get('dimension_scores', {})
            factors   = impact_result.get('key_factors', [])
            bonus     = impact_result.get('excellence_bonus', 0.0)
            reasoning = impact_result.get('reasoning', '')
            ta        = impact_result.get('transcript_analysis', {})

            level_desc = {
                'MINIMAL':  '0–39   — Limited traction.',
                'LOW':      '40–54  — Early traction.',
                'MODERATE': '55–69  — Decent performance.',
                'HIGH':     '70–79  — Strong multi-dimension performance.',
                'VIRAL':    '80–100 — Exceptional — viral-level signals.',
            }

            print(f'\n  🎯  OVERALL IMPACT SCORE')
            print(f'  ' + '─'*38)
            print(f'  Score  : {score:.2f} / 100   {bar(score)}  {level}')
            if bonus > 0:
                high_count = int(round(bonus / 2))
                print(f'  ✨ Multi-dimension excellence bonus: +{bonus:.1f} pts ({high_count} dimensions scored ≥ 70)')
            print(f'  Range  : {level_desc.get(level, "")}')

            if dims:
                dim_weights = {'quality': 30, 'engagement': 25, 'sentiment': 20, 'reach': 15, 'virality': 10}
                dim_sub = {
                    'quality':    'Transcript depth (80%) + Title/Metadata (15%) + Structural (5%)',
                    'engagement': 'Rate quality (45%) + Signal (35%) + Momentum (20%)',
                    'sentiment':  'Valence confidence-weighted (45%) + Intensity (25%) + Polarization (30%)',
                    'reach':      'Scale (30%) + Velocity (30%) + Relative performance (40%)',
                    'virality':   'Historical lift (40%) + Current velocity (35%) + Shareability (25%)',
                }
                print(f'\n  📈  DIMENSION SCORES')
                print(f'  ' + '─'*38)
                for dim, val in dims.items():
                    wt  = dim_weights.get(dim, 0)
                    sub = dim_sub.get(dim, '')
                    print(f'  {dim.capitalize():<12} [wt={wt}%]: {bar(val)} {val:>5.1f}  {score_emoji(val)}')
                    print(f'    {sub}')

            if factors:
                print(f'\n  🔑  KEY FACTORS')
                print(f'  ' + '─'*38)
                for f in factors:
                    print(f'  ▲ {f}')

            if ta:
                wc    = ta.get('transcript_word_count', 0)
                vocab = ta.get('transcript_vocab_richness', 0)
                asl   = ta.get('transcript_avg_sentence_len', 0)
                sent  = ta.get('transcript_sentiment_score', 0)
                sem   = ta.get('semantic_similarity', 0)
                fk    = ta.get('fk_grade_norm', 0)
                mtld  = ta.get('mtld_norm', 0)
                fill  = ta.get('filler_penalty', 0)
                ling  = ta.get('linguistic_score', 0)
                tqs   = ta.get('transcript_quality_score', 0)
                has_t = bool(ta.get('transcript_has_content', 0))

                wc_tag   = '[Low]'  if wc < 200  else ('[Good]' if wc < 1000 else '[High]')
                asl_tag  = '[Good]' if 12 <= asl <= 25 else ('[Short]' if asl < 12 else '[Long]')
                sem_tag  = '🔴 Yapping/filler content' if sem < 0.2 else ('🟡 Moderate depth' if sem < 0.4 else '🟢 Informative')
                fill_tag = '🔴 Heavy filler' if fill > 0.15 else ('🟡 Some filler' if fill > 0.08 else '🟢 Clean')
                sent_lbl = '😊 Positive' if sent > 0.05 else ('😞 Negative' if sent < -0.05 else '😐 Neutral')

                tqs_stars   = '★' * max(1, round(tqs * 5)) + '☆' * (5 - max(1, round(tqs * 5)))
                tqs_verdict = ('🟢 High depth — well-structured, educational content.' if tqs >= 0.75
                               else ('🟡 Moderate depth — more structured content would help.' if tqs >= 0.45
                               else '🔴 Low depth — mostly filler or very short transcript.'))

                print(f'\n  📝  TRANSCRIPT & CONTENT QUALITY')
                print(f'  ' + '─'*38)
                print(f'  Has Transcript       : {"✅ Yes" if has_t else "❌ No"}')
                print(f'  Word Count           : {wc:>8,}  {wc_tag}')
                print(f'  Vocab Richness       : {vocab:>8.3f}  {"[Excellent]" if vocab > 0.7 else ("[Good]" if vocab > 0.5 else "[Low]")}')
                print(f'  Avg Sentence Length  : {asl:>8.1f}  words  {asl_tag}')
                print(f'  Transcript Sentiment : {sent:>+8.3f}  [{sent_lbl}]')
                print(f'\n  ── Quality Signals ──────────────────────────────────')
                print(f'  Semantic Similarity  : {sem:>8.3f}  [{sem_tag}]')
                print(f'  Flesch-Kincaid (norm): {fk:>8.3f}  [reading complexity]')
                print(f'  MTLD Lexical Div.    : {mtld:>8.3f}  [vocab diversity]')
                print(f'  Filler Word Density  : {fill:>8.3f}  [{fill_tag}]')
                print(f'  Linguistic Score     : {ling:>8.3f}  [combined linguistic]')
                print(f'\n  Transcript Quality   : {tqs:>8.3f} / 1.00  [{tqs_stars}]')
                print(f'  Verdict              : {tqs_verdict}')

            if reasoning:
                parts = [p.strip() for p in reasoning.split('|') if p.strip()]
                print(f'\n  💡  ANALYSIS REASONING')
                print(f'  ' + '─'*38)
                for part in parts:
                    print(f'  • {part}')
        else:
            print('  ⚠️  No output from impact model.')

        # ══ 4. CATEGORY CLASSIFICATION ═══════════════════════════════════
        print('\n' + '─'*W)
        print('  🏷️  CATEGORY CLASSIFICATION  (XGBoost / Metadata)')
        print('─'*W)
        if category_out and category_out[0]:
            cat_res, _ = category_out
            primary_cat  = cat_res.get('primary_category', 'Unknown')
            primary_prob = cat_res.get('primary_probability', 0.0)
            
            icon = "🟢" if primary_prob > 0.6 else "🟡"
            print(f'  Verdict          : {icon}  {primary_cat}  ({primary_prob:.1%} confidence)')
            
            all_cats = cat_res.get('all_categories', [])
            if all_cats:
                print(f'\n  Top Probability Distribution:')
                for cat in all_cats[:5]:
                     p = cat['probability']
                     marker = '◀ predicted' if cat['category'] == primary_cat else ''
                     print(f'    {cat["category"]:<20}  {pbar(p, 20)}  {p:5.1%}  {marker}')
        else:
             print('  ⚠️  No output from category model.')

        # ══ 5. VIOLENCE CLASSIFICATION ═══════════════════════════════════
        print('\n' + '─'*W)
        print('  🚨  VIOLENCE CLASSIFICATION  (ResNet-50 / Computer Vision)')
        print('─'*W)
        if violence_out and violence_out[0]:
            v_res, _ = violence_out
            is_violent = v_res.get('is_violent', False)
            severity = v_res.get('severity', 'NONE')
            v_frames = v_res.get('violent_frame_count', 0)
            tot_frames = v_res.get('total_frames', 0)
            v_perc = v_res.get('violence_percentage', 0.0)
            peak_conf = v_res.get('max_confidence', 0.0)
            tier_used = v_res.get('tier_used', 0)
            
            tier_labels = {
                1: "Tier 1 — yt-dlp android/tv_embedded stream",
                2: "Tier 2 — yt-dlp ios/web_embedded stream",
                3: "Tier 3 — YouTube storyboard scraping",
                4: "Tier 4 — Static thumbnail fallback"
            }
            
            v_icon = '🟥' if is_violent else '🟩'
            v_stat = 'VIOLENT CONTENT DETECTED' if is_violent else 'NO SIGNIFICANT VIOLENCE'
            print(f'  Verdict          : {v_icon}  {v_stat}  (Severity: {severity})')
            print(f'  Frame Source     : {tier_labels.get(tier_used, "Unknown")}')
            print(f'  Violence Score   : {v_perc:.2f}% of frames ({v_frames}/{tot_frames} frames)')
            print(f'  Peak Raw Confidence : {peak_conf:.2%}')
            
            timestamps = v_res.get('violent_frame_timestamps', [])
            if is_violent and timestamps:
                print(f'\n  🕒 Detected Violence Timeline:')
                shown = sorted(set(round(ts, 1) for ts in timestamps))
                for ts in shown[:5]:
                    print(f'    {int(ts // 60):02d}:{int(ts % 60):02d}s   !!! VIOLENT ENTRY')
                if len(shown) > 5:
                    print(f'    ... + {len(shown)-5} more moments')
                    
            print(f'\n  📋 Recommendation:\n   {v_res.get("recommendation", "None")}')
            
            if tier_used == 4:
                print('  ⚠️  Warning: Analysis ran on static thumbnail frames only (streams failed).')
        else:
            print('  ⚠️  No output from violence model.')

        print('\n' + '═'*W)

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed. Relying on system environment variables.")

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY environment variable not set. Please define it in a .env file.")
        sys.exit(1)
        
    # Initialize orchestrator once
    orchestrator = GlobalOrchestrator(api_key=YOUTUBE_API_KEY)
    W = 80
    
    while True:
        print("\n" + "="*W)
        print(" 🚀 YOUTUBE CONTENT ANALYZER (GLOBAL)")
        print("="*W)
        url = input("\nEnter YouTube URL (or 'exit' to quit): ").strip()
        
        if url.lower() in ['exit', 'quit', 'e', 'q']:
            print("\nExiting. Goodbye!")
            break
            
        if not url:
            continue
            
        try:
            orchestrator.analyze(url)
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
