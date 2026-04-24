"""
Violence Context Classifier Module
Classifies detected violence into categories to improve interpretation

This module adds semantic understanding on top of raw violence detection
"""


class ViolenceContextClassifier:
    """
    Classifies violence into contextual categories:
    - Regulated Sport (UFC, boxing, wrestling)
    - Entertainment (movie, TV show, staged)
    - Real Violence (crime, assault, news footage)
    - Educational/Documentary (historical, training)
    """
    
    def __init__(self):
        # Context-specific keyword sets
        self.context_keywords = {
            'sport': {
                # Combat sports
                'ufc', 'wwe', 'wrestling', 'mma', 'boxing', 'kickboxing',
                'bellator', 'one championship', 'pfl',
                'grappling', 'jiu jitsu', 'bjj', 'muay thai',
                'fight night', 'main event', 'ppv', 'pay-per-view',
                'octagon', 'ring', 'referee', 'rounds', 'knockout', 'ko',
                'submission', 'takedown', 'ground and pound',
                # General sports violence
                'tackle', 'collision', 'impact', 'hit', 'check',
                'football', 'hockey', 'rugby', 'lacrosse'
            },
            'entertainment': {
                'movie', 'film', 'cinema', 'trailer', 'scene',
                'tv show', 'television', 'series', 'episode',
                'action', 'thriller', 'horror', 'drama',
                'staged', 'choreographed', 'stunt', 'special effects',
                'behind the scenes', 'making of', 'bloopers'
            },
            'real_violence': {
                'crime', 'assault', 'attack', 'victim', 'police',
                'arrest', 'charges', 'court', 'trial',
                'news', 'breaking', 'report', 'incident',
                'surveillance', 'cctv', 'dashcam', 'bodycam',
                'fight', 'brawl', 'shooting', 'stabbing'
            },
            'educational': {
                'documentary', 'history', 'historical', 'archive',
                'training', 'tutorial', 'instructional', 'how to',
                'analysis', 'breakdown', 'study', 'research',
                'educational', 'learning', 'lecture', 'course'
            }
        }
        
        # Severity calibration by context
        self.context_severity_multipliers = {
            'sport': 0.6,      # Reduce perceived severity (regulated)
            'entertainment': 0.4,  # Further reduce (staged/fake)
            'real_violence': 1.2,  # Increase severity (actual harm)
            'educational': 0.3   # Minimal severity (informational)
        }
    
    def classify(self, metadata, transcript=None, violence_analysis=None):
        """
        Classify the context of detected violence
        
        Args:
            metadata: Video metadata (title, description, tags, channel)
            transcript: Optional video transcript text
            violence_analysis: Results from violence detection
            
        Returns:
            Dict with classification results
        """
        # Gather all available text
        text_context = self._build_text_context(metadata, transcript)
        
        # Score each context type
        context_scores = self._score_contexts(text_context)
        
        # Determine primary context
        primary_context = max(context_scores, key=context_scores.get)
        confidence = context_scores[primary_context] / sum(context_scores.values())
        
        # Get severity multiplier
        severity_multiplier = self.context_severity_multipliers[primary_context]
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            primary_context, 
            confidence,
            violence_analysis,
            severity_multiplier
        )
        
        return {
            'primary_context': primary_context,
            'confidence': confidence,
            'context_scores': context_scores,
            'severity_multiplier': severity_multiplier,
            'interpretation': interpretation,
            'recommendation': self._generate_recommendation(primary_context, violence_analysis)
        }
    
    def _build_text_context(self, metadata, transcript):
        """Combine all text sources for analysis"""
        title = metadata.get('title', '').lower()
        description = metadata.get('description', '').lower()
        tags = ' '.join(metadata.get('tags', [])).lower()
        channel = metadata.get('channel', '').lower()
        
        combined = f"{title} {description} {tags} {channel}"
        
        # Add transcript if available
        if transcript:
            combined += f" {transcript.lower()}"
        
        return combined
    
    def _score_contexts(self, text):
        """Score each context based on keyword matches"""
        scores = {
            'sport': 0,
            'entertainment': 0,
            'real_violence': 0,
            'educational': 0
        }
        
        for context, keywords in self.context_keywords.items():
            match_count = sum(1 for kw in keywords if kw in text)
            scores[context] = match_count
        
        return scores
    
    def _generate_interpretation(self, context, confidence, violence_analysis, severity_mult):
        """Generate human-readable interpretation"""
        interpretations = {
            'sport': {
                'high_conf': "Regulated athletic competition with sanctioned violence",
                'med_conf': "Likely sports content with competitive physical contact",
                'low_conf': "Possible sports-related physical activity"
            },
            'entertainment': {
                'high_conf': "Staged/performance violence for entertainment purposes",
                'med_conf': "Likely fictional or choreographed action sequences",
                'low_conf': "Possible entertainment content with action elements"
            },
            'real_violence': {
                'high_conf': "Actual violence with potential for real harm",
                'med_conf': "Likely real-world violent incident or footage",
                'low_conf': "Possible documentation of real violence"
            },
            'educational': {
                'high_conf': "Educational/informational content about violence",
                'med_conf': "Documentary or analytical content",
                'low_conf': "Possible educational context"
            }
        }
        
        if confidence >= 0.6:
            conf_level = 'high_conf'
        elif confidence >= 0.3:
            conf_level = 'med_conf'
        else:
            conf_level = 'low_conf'
        
        base_interpretation = interpretations[context][conf_level]
        
        # Add violence metrics if available
        if violence_analysis:
            violence_pct = violence_analysis.get('violence_percentage', 0)
            severity = violence_analysis.get('severity', 'NONE')
            
            interpretation = f"{base_interpretation}. "
            interpretation += f"Violence level: {violence_pct:.0f}% ({severity})"
        else:
            interpretation = base_interpretation
        
        return interpretation
    
    def _generate_recommendation(self, context, violence_analysis):
        """Generate content recommendation based on context"""
        if not violence_analysis:
            return "Insufficient data for recommendation"
        
        is_violent = violence_analysis.get('is_violent', False)
        severity = violence_analysis.get('severity', 'NONE')
        
        if not is_violent:
            return "Content appears safe for general audiences"
        
        recommendations = {
            'sport': {
                'EXTREME': "HIGH sports violence. Sanctioned competition. Viewer discretion advised (16+)",
                'HIGH': "MODERATE-HIGH sports violence. Competitive contact (13+)",
                'MODERATE': "MODERATE sports contact. Athletic competition (10+)",
                'LOW': "Mild sports action. Generally suitable",
                'NONE': "No significant violence detected"
            },
            'entertainment': {
                'EXTREME': "INTENSE fictional violence. Mature themes (18+)",
                'HIGH': "HIGH action violence. Staged combat (16+)",
                'MODERATE': "MODERATE action sequences. Choreographed (13+)",
                'LOW': "Mild action content. Generally suitable",
                'NONE': "No significant violence detected"
            },
            'real_violence': {
                'EXTREME': "GRAPHIC real violence. Disturbing content (18+ only)",
                'HIGH': "REAL violent incident. May be disturbing (16+)",
                'MODERATE': "Real-world violence. Viewer discretion advised",
                'LOW': "Minor real violence. Context needed",
                'NONE': "No significant violence detected"
            },
            'educational': {
                'EXTREME': "GRAPHIC educational content. Mature audience learning (16+)",
                'HIGH': "Detailed violence analysis. Educational purpose (13+)",
                'MODERATE': "Moderate educational violence. Informational",
                'LOW': "Mild educational references. Suitable for learning",
                'NONE': "No significant violence detected"
            }
        }
        
        context_recs = recommendations.get(context, recommendations['sport'])
        return context_recs.get(severity, context_recs['NONE'])


def main():
    """Test the violence context classifier"""
    print("="*80)
    print("VIOLENCE CONTEXT CLASSIFIER TEST")
    print("="*80)
    
    classifier = ViolenceContextClassifier()
    
    # Test case 1: UFC Fight
    print("\n\nTEST 1: UFC Fight Video")
    print("-"*80)
    ufc_metadata = {
        'title': 'Khabib vs McGregor | FULL FIGHT | UFC 229',
        'channel': 'UFC',
        'description': 'Full fight between Khabib Nurmagomedov and Conor McGregor',
        'tags': ['ufc', 'mma', 'fighting', 'khabib', 'mcgregor', 'ufc 229']
    }
    
    ufc_violence = {
        'is_violent': True,
        'violence_percentage': 87.0,
        'severity': 'HIGH'
    }
    
    result = classifier.classify(ufc_metadata, violence_analysis=ufc_violence)
    
    print(f"Primary Context: {result['primary_context']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Severity Multiplier: {result['severity_multiplier']}")
    print(f"\nInterpretation: {result['interpretation']}")
    print(f"\nRecommendation: {result['recommendation']}")
    
    # Test case 2: Movie Scene
    print("\n\nTEST 2: Action Movie Scene")
    print("-"*80)
    movie_metadata = {
        'title': 'John Wick 4 - Best Action Scenes Compilation',
        'channel': 'MovieClips',
        'description': 'Best fight scenes from John Wick Chapter 4',
        'tags': ['movie', 'action', 'john wick', 'film', 'trailer']
    }
    
    movie_violence = {
        'is_violent': True,
        'violence_percentage': 65.0,
        'severity': 'MODERATE'
    }
    
    result = classifier.classify(movie_metadata, violence_analysis=movie_violence)
    
    print(f"Primary Context: {result['primary_context']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Severity Multiplier: {result['severity_multiplier']}")
    print(f"\nInterpretation: {result['interpretation']}")
    print(f"\nRecommendation: {result['recommendation']}")
    
    # Test case 3: News Footage
    print("\n\nTEST 3: News Violence Footage")
    print("-"*80)
    news_metadata = {
        'title': 'Breaking News: Police Confrontation Caught on Camera',
        'channel': 'CNN',
        'description': 'Surveillance footage shows violent incident',
        'tags': ['news', 'crime', 'breaking', 'police', 'incident']
    }
    
    news_violence = {
        'is_violent': True,
        'violence_percentage': 45.0,
        'severity': 'MODERATE'
    }
    
    result = classifier.classify(news_metadata, violence_analysis=news_violence)
    
    print(f"Primary Context: {result['primary_context']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Severity Multiplier: {result['severity_multiplier']}")
    print(f"\nInterpretation: {result['interpretation']}")
    print(f"\nRecommendation: {result['recommendation']}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
