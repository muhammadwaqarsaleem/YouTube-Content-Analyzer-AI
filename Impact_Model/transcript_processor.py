"""
Transcript quality scoring for the YouTube Impact Score Model (v9).
No data fetching — expects raw transcript text as input.
"""

import logging
import re
import string
from typing import Dict, List

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


class TranscriptQualityScorer:
    """
    Scores transcript quality using two complementary signals:

    1. Semantic Similarity (45%)
       Uses sentence-transformers (all-MiniLM-L6-v2).
       Yapping/filler content → low score.
       Educational/analytical content → high score.

    2. Linguistic Quality (55%)
       a. Flesch-Kincaid Grade Level  (20%)
       b. MTLD Lexical Diversity      (20%)
       c. Filler Word Penalty         (15%)
       d. Avg Sentence Length         (10%)
       e. Has Content bonus           (10%)

    v9 short-transcript fix:
      When word_count < 150: linguistic_score scaled by (word_count/150)
      and semantic weight raised to 0.65.
    """

    QUALITY_ANCHORS = [
        "This study demonstrates that neural networks achieve 94% accuracy on benchmark "
        "tasks. The methodology involves three phases of cross-validation to ensure "
        "statistical reliability. The results indicate a significant correlation between "
        "model depth and generalization performance.",

        "In this video, we are going to break down how this system works step-by-step. "
        "We will start with the basic components and then look at how they interact "
        "within the larger framework. Understanding these core principles is essential "
        "before we dive into more advanced technical implementations.",

        "The French Revolution fundamentally restructured European political thought. "
        "Historians attribute this transformation to three core economic factors: "
        "fiscal crisis, social inequality, and the spread of Enlightenment ideas. "
        "The subsequent Napoleonic era consolidated many revolutionary reforms.",

        "To implement binary search, you first identify the midpoint of the sorted array. "
        "If the target value is less than the midpoint element, you recursively search "
        "the left half. This divide-and-conquer approach achieves O(log n) time complexity.",

        "Compound interest operates on the principle of earning returns on previously "
        "accumulated interest. Over a 30-year horizon, a 7% annual return doubles "
        "the initial investment approximately every decade, demonstrating the "
        "exponential nature of long-term wealth accumulation.",

        "Mitochondria generate ATP through oxidative phosphorylation, a process occurring "
        "across the inner mitochondrial membrane. Electron transport chain complexes "
        "create a proton gradient that drives ATP synthase. This mechanism underlies "
        "cellular energy metabolism in all aerobic organisms.",

        "The primary cause of this phenomenon can be attributed to three interrelated "
        "factors. First, the structural changes in market conditions created downward "
        "pressure on valuations. Second, regulatory uncertainty led institutional "
        "investors to reduce exposure. Third, macroeconomic headwinds compounded "
        "the existing vulnerabilities in the sector.",
    ]

    FILLER_WORDS = {
        'um', 'uh', 'umm', 'uhh', 'like', 'you know', 'basically', 'literally',
        'actually', 'honestly', 'obviously', 'clearly', 'totally', 'absolutely',
        'kinda', 'sorta', 'gonna', 'wanna', 'gotta', 'yknow', 'anyways',
        'whatever', 'stuff', 'things', 'thing', 'right', 'okay so', 'so yeah',
        'i mean', 'you see', 'you guys', 'hey guys', "what's up", 'subscribe',
        'smash that', 'hit that', 'drop a like', 'blah', 'bro', 'dude',
        'awesome', 'amazing', 'incredible', 'insane', 'crazy', 'super',
    }

    def __init__(self):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading sentence-transformer model (all-MiniLM-L6-v2) on {device}...")
        self.encoder      = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.stop_words   = set(stopwords.words('english'))
        self._anchor_vecs = None
        self._encode_anchors()
        logger.info("TranscriptQualityScorer ready.")

    def _encode_anchors(self):
        self._anchor_vecs = self.encoder.encode(
            self.QUALITY_ANCHORS, convert_to_numpy=True, show_progress_bar=False
        )

    @staticmethod
    def _mtld(tokens: List[str], ttr_threshold: float = 0.72) -> float:
        if len(tokens) < 10:
            return 0.0

        def _forward(toks):
            segments, start, seen = 0.0, 0, set()
            for i, t in enumerate(toks):
                seen.add(t)
                ttr = len(seen) / (i - start + 1)
                if ttr < ttr_threshold:
                    segments += 1; start = i + 1; seen = set()
            remainder = len(toks) - start
            if remainder > 0 and start < len(toks):
                seen_rem = set(toks[start:])
                ttr_rem  = len(seen_rem) / remainder
                segments += (1 - ttr_rem) / (1 - ttr_threshold + 1e-10)
            return len(toks) / (segments + 1e-10)

        return (_forward(tokens) + _forward(list(reversed(tokens)))) / 2

    def _filler_density(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        filler_count = sum(1 for w in words if w.strip(string.punctuation) in self.FILLER_WORDS)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        filler_count += sum(1 for b in bigrams if b in self.FILLER_WORDS)
        return min(filler_count / len(words), 1.0)

    def _semantic_score(self, text: str) -> float:
        words  = text.split()
        chunks = []
        step   = 200
        for i in range(0, max(len(words), step), step):
            chunk = ' '.join(words[i:i + step])
            if chunk.strip():
                chunks.append(chunk)
        if not chunks:
            return 0.0
        chunk_vecs = self.encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        from numpy.linalg import norm
        sims = []
        for cv in chunk_vecs:
            for av in self._anchor_vecs:
                cos = np.dot(cv, av) / (norm(cv) * norm(av) + 1e-10)
                sims.append(cos)
        return float(np.max(sims)) if sims else 0.0

    def _linguistic_score(self, text: str, tokens: List[str]) -> Dict:
        try:
            fk_grade = textstat.flesch_kincaid_grade(text)
            if len(tokens) < 30:
                fk_norm = 0.0
            else:
                fk_norm = float(np.clip(fk_grade / 16.0, 0, 1))
        except Exception:
            fk_norm = 0.0

        mtld_raw  = self._mtld(tokens)
        mtld_norm = float(np.clip(mtld_raw / 100.0, 0, 1))

        filler_d     = self._filler_density(text)
        filler_score = 1.0 - filler_d

        try:
            sentences = sent_tokenize(text)
            asl       = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        except Exception:
            asl = len(tokens) / max(text.count('.') + 1, 1)
        asl_norm    = float(np.clip((asl - 5) / 25.0, 0, 1))
        has_content = 1.0 if len(tokens) > 20 else 0.0

        active_weight = 0.20 + 0.20 + 0.15 + 0.10 + (0.10 if has_content else 0.0)
        combined = (
            fk_norm      * 0.20 +
            mtld_norm    * 0.20 +
            filler_score * 0.15 +
            asl_norm     * 0.10 +
            has_content  * 0.10
        ) / (active_weight + 1e-10)

        return {
            'fk_grade_norm':    round(fk_norm,      4),
            'mtld_norm':        round(mtld_norm,     4),
            'filler_penalty':   round(filler_d,      4),
            'filler_score':     round(filler_score,  4),
            'asl_norm':         round(asl_norm,      4),
            'has_content':      has_content,
            'linguistic_score': round(float(np.clip(combined, 0, 1)), 4),
        }

    def score(self, raw_text: str) -> Dict:
        """Main entry point. Returns all signals + final transcript_quality_score."""
        if not isinstance(raw_text, str) or raw_text.strip() == '':
            return {
                'transcript_word_count':       0,
                'transcript_avg_sentence_len': 0.0,
                'transcript_vocab_richness':   0.0,
                'transcript_sentiment_score':  0.0,
                'transcript_has_content':      0,
                'transcript_quality_score':    0.0,
                'semantic_similarity':         0.0,
                'fk_grade_norm':               0.0,
                'mtld_norm':                   0.0,
                'filler_penalty':              0.0,
                'linguistic_score':            0.0,
            }

        clean  = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
        clean  = re.sub(r'\s+', ' ', clean).strip()
        try:    tokens = word_tokenize(clean)
        except: tokens = clean.split()
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]

        word_count     = len(tokens)
        vocab_richness = len(set(tokens)) / max(word_count, 1)

        temp_text = raw_text
        if raw_text.count('.') < (len(raw_text.split()) / 50):
            words     = raw_text.split()
            temp_text = '. '.join([' '.join(words[i:i + 20]) for i in range(0, len(words), 20)])

        try:
            sentences = sent_tokenize(temp_text)
            avg_sent  = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0
        except:
            avg_sent  = word_count / max(raw_text.count('.') + 1, 1)

        try:    sentiment_score = SentimentIntensityAnalyzer().polarity_scores(raw_text[:5000])['compound']
        except: sentiment_score = 0.0

        semantic = self._semantic_score(raw_text)
        ling     = self._linguistic_score(raw_text, tokens)

        # Short-transcript saturation fix (v9)
        MIN_WORDS_FOR_SCORING = 150
        if word_count > 0 and word_count < MIN_WORDS_FOR_SCORING:
            word_count_penalty = word_count / MIN_WORDS_FOR_SCORING
            penalised_ling     = ling['linguistic_score'] * word_count_penalty
            sem_weight         = 0.65
            ling_weight        = 0.35
        else:
            penalised_ling     = ling['linguistic_score']
            sem_weight         = 0.45
            ling_weight        = 0.55

        tqs = float(np.clip(semantic * sem_weight + penalised_ling * ling_weight, 0, 1))

        return {
            'transcript_word_count':       word_count,
            'transcript_avg_sentence_len': round(avg_sent,        4),
            'transcript_vocab_richness':   round(vocab_richness,   4),
            'transcript_sentiment_score':  round(sentiment_score,  4),
            'transcript_has_content':      1 if word_count > 0 else 0,
            'transcript_quality_score':    round(tqs,              4),
            'semantic_similarity':         round(semantic,          4),
            'fk_grade_norm':               ling['fk_grade_norm'],
            'mtld_norm':                   ling['mtld_norm'],
            'filler_penalty':              ling['filler_penalty'],
            'linguistic_score':            ling['linguistic_score'],
        }


class TranscriptProcessor:
    """Thin wrapper for API compatibility. Delegates to TranscriptQualityScorer."""

    def __init__(self):
        self.quality_scorer = TranscriptQualityScorer()
        self.is_fitted      = True

    def process_single(self, transcript_text: str) -> Dict:
        result = self.quality_scorer.score(transcript_text)
        result['tfidf_mean_score']    = 0.0
        result['tfidf_max_score']     = 0.0
        result['tfidf_nonzero_count'] = 0
        return result
