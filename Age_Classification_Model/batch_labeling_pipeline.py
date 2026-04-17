"""
Production-Grade Batch Labeling Pipeline for YouTube Age Classification
=========================================================================
Implements Programmatic Weak Supervision using keyword-density analysis
to create a Silver Training Dataset for NLP model training.

Research Project: YouTube Video Age Bracket Classification
Authors: University AI Research Team
Date: 2026
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# COMPREHENSIVE KEYWORD DICTIONARIES - Content Moderation Expert Definitions
# ============================================================================

# General (G-Rated) Content Indicators - Family-Friendly, Educational
GENERAL_KEYWORDS = {
    # Educational & Learning
    'tutorial', 'learn', 'educational', 'teaching', 'lesson', 'study', 'guide',
    'how to', 'explanation', 'science', 'mathematics', 'history', 'geography',
    'biology', 'chemistry', 'physics', 'astronomy', 'knowledge', 'discover',
    'explore', 'experiment', 'research', 'analysis', 'understanding',
    
    # Family & Children
    'family', 'kids', 'children', 'toddler', 'baby', 'parent', 'mom', 'dad',
    'grandmother', 'grandfather', 'sibling', 'wholesome', 'innocent', 'safe',
    'friendly', 'kind', 'caring', 'gentle', 'sweet', 'adorable', 'cute',
    
    # Positive Values & Affirmations
    'kindness', 'compassion', 'empathy', 'respect', 'honesty', 'integrity',
    'gratitude', 'appreciation', 'love', 'happiness', 'joy', 'peace',
    'harmony', 'cooperation', 'teamwork', 'friendship', 'help', 'support',
    'encourage', 'inspire', 'motivate', 'positive', 'uplifting', 'hopeful',
    
    # Creative & Arts
    'art', 'craft', 'drawing', 'painting', 'music', 'singing', 'dancing',
    'creative', 'imagination', 'storytelling', 'poetry', 'design', 'handmade',
    'DIY', 'cooking', 'baking', 'recipe', 'gardening', 'nature', 'animals',
    
    # Entertainment (Clean)
    'cartoon', 'animation', 'disney', 'pixar', 'toy', 'game show', 'comedy',
    'funny', 'laugh', 'smile', 'entertainment', 'show', 'performance',
    
    # Technology & Innovation (Educational)
    'innovation', 'invention', 'technology', 'engineering', 'coding',
    'programming', 'robotics', 'space', 'NASA', 'discovery', 'future',
    
    # Health & Wellness (Positive)
    'health', 'wellness', 'fitness', 'exercise', 'yoga', 'meditation',
    'nutrition', 'healthy eating', 'mental health', 'wellbeing', 'self-care',
    
    # Community & Social Good
    'community', 'volunteer', 'charity', 'donation', 'helping others',
    'social good', 'sustainability', 'environment', 'conservation', 'recycling',
    'climate', 'awareness', 'advocacy', 'equality', 'diversity', 'inclusion'
}

# Teen (T-Rated) Content Indicators - Gaming, Mild Language, Action
TEEN_KEYWORDS = {
    # Gaming & Esports
    'gaming', 'gamer', 'gameplay', 'playthrough', 'walkthrough', 'speedrun',
    'pvp', 'multiplayer', 'battle royale', 'esports', 'competitive', 'rank',
    'noob', 'pwned', 'rekt', 'gg', 'clutch', 'headshot', 'respawn', 'lag',
    'camping', 'grinding', 'loot', 'nerf', 'buff', 'meta', 'op', 'broken',
    
    # Gaming Platforms & Titles
    'fortnite', 'minecraft', 'roblox', 'league of legends', 'valorant',
    'apex legends', 'call of duty', 'overwatch', 'destiny', 'warzone',
    'pubg', 'fifa', 'madden', 'NBA 2K', 'rocket league', 'among us',
    
    # Mild Slang & Teen Language
    'dude', 'bro', 'bruh', 'lit', 'fire', 'cringe', 'yeet', 'sus', 'cap',
    'no cap', 'lowkey', 'highkey', 'stan', 'simp', 'toxic', 'salty',
    'rage', 'tilted', 'throwing', 'griefing', 'trolling', 'meme', 'based',
    
    # Action & Combat (Non-Graphic)
    'fight', 'battle', 'combat', 'attack', 'defend', 'weapon', 'sword',
    'gun', 'shoot', 'explosion', 'destroy', 'defeat', 'victory', 'defeat',
    'warrior', 'soldier', 'mission', 'quest', 'raid', 'boss fight', 'enemy',
    
    # Mild Intensity Language
    'damn', 'hell', 'crap', 'sucks', 'pissed', 'stupid', 'idiot', 'dumb',
    'jerk', 'loser', 'lame', 'annoying', 'hate', 'disgusting', 'gross',
    
    # Teen Drama & Relationships
    'drama', 'breakup', 'ex', 'crush', 'dating', 'relationship', 'toxic',
    'ghosting', 'friendzone', 'heartbreak', 'betrayal', 'rumors', 'gossip',
    
    # Social Media & Internet Culture
    'tiktok', 'instagram', 'snapchat', 'twitter', 'viral', 'trending',
    'influencer', 'streamer', 'youtuber', 'subscriber', 'follower', 'likes',
    'views', 'algorithm', 'clickbait', 'thumbnail', 'drama alert',
    
    # Music & Entertainment (Teen-oriented)
    'rap', 'hip hop', 'trap', 'diss track', 'beef', 'roast', 'savage',
    'shade', 'clout', 'flex', 'drip', 'vibe', 'mood', 'aesthetic',
    
    # Competitive Language
    'dominate', 'crush', 'destroy', 'annihilate', 'obliterate', 'wreck',
    'own', 'domination', 'massacre', 'slaughter', 'rampage', 'killing spree',
    
    # Edgy Humor
    'dark humor', 'edgy', 'offensive', 'controversial', 'inappropriate',
    'nsfw', 'trigger warning', 'cancelled', 'problematic', 'yikes'
}

# Mature (M-Rated) Content Indicators - Strong Language, Violence, Adult Themes
MATURE_KEYWORDS = {
    # Strong Profanity (Academic examples for content moderation)
    'fuck', 'fucking', 'fucked', 'shit', 'shitty', 'bullshit', 'bitch',
    'bastard', 'ass', 'asshole', 'dick', 'pussy', 'cock', 'prick',
    
    # Violence & Gore
    'gore', 'bloody', 'blood', 'murder', 'kill', 'killing', 'death',
    'torture', 'brutal', 'savage', 'violent', 'violence', 'beating',
    'stabbing', 'shooting', 'execution', 'massacre', 'slaughter',
    'dismember', 'mutilation', 'corpse', 'dead body', 'suicide', 'self-harm',
    
    # Weapons & Combat (Graphic)
    'gun violence', 'mass shooting', 'terrorist', 'terrorism', 'bomb',
    'explosive', 'grenade', 'assault rifle', 'ammunition', 'sniper',
    
    # Substance Use
    'drugs', 'cocaine', 'heroin', 'meth', 'weed', 'marijuana', 'cannabis',
    'high', 'stoned', 'drunk', 'alcohol', 'beer', 'liquor', 'vodka',
    'whiskey', 'smoking', 'cigarette', 'vaping', 'addiction', 'overdose',
    'dealer', 'dealing', 'trafficking', 'junkie', 'addict',
    
    # Sexual Content & References
    'sex', 'sexual', 'nude', 'naked', 'porn', 'pornography', 'explicit',
    'erotic', 'seduction', 'affair', 'cheating', 'orgasm', 'intercourse',
    'prostitution', 'escort', 'strip club', 'stripper', 'brothel',
    
    # Crime & Illegal Activity
    'crime', 'criminal', 'theft', 'robbery', 'burglary', 'stealing',
    'fraud', 'scam', 'illegal', 'contraband', 'trafficking', 'smuggling',
    'gang', 'cartel', 'mafia', 'organized crime', 'hitman', 'assassin',
    
    # Disturbing Content
    'disturbing', 'graphic', 'explicit', 'horrific', 'traumatic', 'abuse',
    'domestic violence', 'assault', 'rape', 'molestation', 'predator',
    'victim', 'trauma', 'PTSD', 'depression', 'anxiety', 'mental illness',
    
    # Mature Themes
    'adultery', 'infidelity', 'divorce', 'abortion', 'racism', 'sexism',
    'discrimination', 'hate crime', 'extremism', 'radicalization',
    'conspiracy', 'cult', 'brainwashing', 'manipulation',
    
    # Dark/Horror Content
    'horror', 'scary', 'terrifying', 'nightmare', 'demon', 'satanic',
    'occult', 'ritual', 'sacrifice', 'haunted', 'paranormal', 'ghost',
    'zombie', 'vampire', 'monster', 'creature', 'beast'
}

# Zero Tolerance Keywords - Immediate Mature Classification
ZERO_TOLERANCE_KEYWORDS = {
    # Extreme Slurs (Academic examples - minimal set for research purposes)
    'n***a', 'f****t', 'r****d', 'c**t',
    
    # Graphic Violence
    'beheading', 'decapitation', 'genocide', 'holocaust', 'war crime',
    'child abuse', 'pedophile', 'child porn', 'snuff film',
    
    # Extreme Sexual Content
    'rape', 'sexual assault', 'molestation', 'incest', 'bestiality',
    
    # Terrorism & Extremism
    'terrorist attack', 'suicide bomber', 'mass shooting', 'school shooting',
    'isis', 'al qaeda', 'extremist', 'white supremacy', 'nazi',
    
    # Self-Harm & Suicide
    'suicide', 'kill myself', 'end my life', 'self-harm', 'cutting'
}


@dataclass
class LabelingMetrics:
    """Data class to store labeling statistics and metrics."""
    total_processed: int
    general_count: int
    teen_count: int
    mature_count: int
    avg_general_length: float
    avg_teen_length: float
    avg_mature_length: float
    zero_tolerance_triggers: int


class WeakSupervisionLabeler:
    """
    Advanced weak supervision labeling engine using keyword-density analysis.
    
    Implements a content moderation framework to classify YouTube transcripts
    into age-appropriate categories (General, Teen, Mature) based on keyword
    density rather than absolute counts.
    """
    
    def __init__(self, csv_path: str, output_path: str = "training_ready_dataset.csv"):
        """
        Initialize the labeling pipeline.
        
        Args:
            csv_path: Path to the input CSV file
            output_path: Path for the output labeled dataset
        """
        self.csv_path = Path(csv_path)
        self.output_path = Path(output_path)
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Optional[LabelingMetrics] = None
        
        # Compile regex patterns for efficiency (critical for large transcripts)
        print("🔧 Compiling keyword patterns for efficient processing...")
        self.general_pattern = self._compile_pattern(GENERAL_KEYWORDS)
        self.teen_pattern = self._compile_pattern(TEEN_KEYWORDS)
        self.mature_pattern = self._compile_pattern(MATURE_KEYWORDS)
        self.zero_tolerance_pattern = self._compile_pattern(ZERO_TOLERANCE_KEYWORDS)
        
        # Words per minute assumption for timestamp estimation
        self.WPM = 150
        
    def _compile_pattern(self, keywords: Set[str]) -> re.Pattern:
        """
        Compile keywords into an efficient regex pattern with word boundaries.
        
        Args:
            keywords: Set of keywords to compile
            
        Returns:
            Compiled regex pattern
        """
        # Escape special regex characters and add word boundaries
        escaped_keywords = [re.escape(kw) for kw in keywords]
        # Use word boundaries to avoid partial matches
        pattern_str = r'\b(' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern_str, re.IGNORECASE)
    
    def load_and_clean_data(self) -> bool:
        """
        Load CSV and clean data by removing rows with missing transcripts.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 80)
        print("📂 LOADING AND CLEANING DATA")
        print("=" * 80)
        
        if not self.csv_path.exists():
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        
        try:
            # Load data
            print(f"⏳ Loading {self.csv_path.name}...")
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            initial_rows = len(self.df)
            print(f"✅ Loaded {initial_rows:,} rows\n")
            
            # Check for transcript column
            if 'transcript' not in self.df.columns:
                print("❌ Error: 'transcript' column not found in dataset")
                print(f"Available columns: {', '.join(self.df.columns)}")
                return False
            
            # Data cleaning
            print("🧹 Cleaning Data:")
            
            # Count missing transcripts
            missing_count = self.df['transcript'].isna().sum()
            print(f"   • Missing transcripts: {missing_count:,} ({(missing_count/initial_rows)*100:.2f}%)")
            
            # Drop rows with missing transcripts
            self.df = self.df.dropna(subset=['transcript'])
            
            # Reset index to maintain clean indexing
            self.df = self.df.reset_index(drop=True)
            
            final_rows = len(self.df)
            removed_rows = initial_rows - final_rows
            
            print(f"   • Removed: {removed_rows:,} rows")
            print(f"   • Remaining: {final_rows:,} rows")
            print(f"   ✅ {len(self.df.columns)} original columns retained\n")
            
            if final_rows == 0:
                print("❌ Error: No valid transcripts remaining after cleaning")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_keyword_matches(self, text: str, pattern: re.Pattern) -> List[Tuple[str, int]]:
        """
        Calculate keyword matches and their positions in text.
        
        Args:
            text: Text to search
            pattern: Compiled regex pattern
            
        Returns:
            List of (matched_word, character_position) tuples
        """
        matches = []
        for match in pattern.finditer(text):
            matches.append((match.group().lower(), match.start()))
        return matches
    
    def _estimate_timestamps(self, positions: List[int], total_words: int) -> List[int]:
        """
        Estimate minute timestamps based on word positions.
        
        Args:
            positions: Character positions of matches
            total_words: Total word count in transcript
            
        Returns:
            List of estimated minute marks
        """
        if total_words == 0:
            return []
        
        # Convert character positions to approximate word positions
        # Assumption: average word length + space = 6 characters
        word_positions = [pos // 6 for pos in positions]
        
        # Convert word positions to minutes
        minutes = [word_pos // self.WPM for word_pos in word_positions]
        
        # Return unique, sorted minutes (cap at 10 occurrences)
        return sorted(list(set(minutes)))[:10]
    
    def _create_content_flags(self, 
                             mature_matches: List[Tuple[str, int]], 
                             teen_matches: List[Tuple[str, int]],
                             total_words: int) -> str:
        """
        Create content flags with categories, counts, and estimated timestamps.
        
        Args:
            mature_matches: List of (word, position) for mature content
            teen_matches: List of (word, position) for teen content
            total_words: Total word count
            
        Returns:
            Formatted content flags string
        """
        flags = []
        
        # Categorize mature content
        if mature_matches:
            # Group by content type
            violence_words = {'gore', 'bloody', 'murder', 'kill', 'death', 'torture', 'violent'}
            profanity_words = {'fuck', 'shit', 'bitch', 'ass', 'damn'}
            substance_words = {'drug', 'alcohol', 'weed', 'drunk', 'high', 'smoking'}
            sexual_words = {'sex', 'sexual', 'nude', 'porn', 'explicit'}
            
            violence_matches = [(w, p) for w, p in mature_matches if any(v in w for v in violence_words)]
            profanity_matches = [(w, p) for w, p in mature_matches if any(v in w for v in profanity_words)]
            substance_matches = [(w, p) for w, p in mature_matches if any(v in w for v in substance_words)]
            sexual_matches = [(w, p) for w, p in mature_matches if any(v in w for v in sexual_words)]
            
            if violence_matches:
                positions = [p for _, p in violence_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Violence_Reference: {len(violence_matches)} (approx {timestamp_str})")
            
            if profanity_matches:
                positions = [p for _, p in profanity_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Strong_Language: {len(profanity_matches)} (approx {timestamp_str})")
            
            if substance_matches:
                positions = [p for _, p in substance_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Substance_Reference: {len(substance_matches)} (approx {timestamp_str})")
            
            if sexual_matches:
                positions = [p for _, p in sexual_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Sexual_Content: {len(sexual_matches)} (approx {timestamp_str})")
        
        # Categorize teen content
        if teen_matches:
            gaming_words = {'gaming', 'gamer', 'game', 'play', 'fortnite', 'minecraft'}
            mild_lang_words = {'damn', 'hell', 'crap', 'stupid', 'idiot'}
            
            gaming_matches = [(w, p) for w, p in teen_matches if any(v in w for v in gaming_words)]
            mild_matches = [(w, p) for w, p in teen_matches if any(v in w for v in mild_lang_words)]
            
            if gaming_matches:
                positions = [p for _, p in gaming_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Gaming_Content: {len(gaming_matches)} (approx {timestamp_str})")
            
            if mild_matches:
                positions = [p for _, p in mild_matches]
                timestamps = self._estimate_timestamps(positions, total_words)
                timestamp_str = ', '.join([f"min {t}" for t in timestamps])
                flags.append(f"Mild_Language: {len(mild_matches)} (approx {timestamp_str})")
        
        # Return formatted flags or "None" if no flags
        return ' | '.join(flags) if flags else 'None'
    
    def label_transcript(self, transcript: str) -> Tuple[str, int, float, float, str]:
        """
        Apply weak supervision labeling logic to a single transcript.
        
        Args:
            transcript: Text transcript to label
            
        Returns:
            Tuple of (label, severity_score, mature_density, teen_density, content_flags)
        """
        # Convert to lowercase for processing
        text_lower = transcript.lower()
        
        # Calculate word count
        words = text_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return 'General', 0, 0.0, 0.0, 'None'
        
        # Find all matches with positions
        zero_tolerance_matches = self._calculate_keyword_matches(text_lower, self.zero_tolerance_pattern)
        mature_matches = self._calculate_keyword_matches(text_lower, self.mature_pattern)
        teen_matches = self._calculate_keyword_matches(text_lower, self.teen_pattern)
        
        # Calculate counts
        zero_tolerance_count = len(zero_tolerance_matches)
        mature_count = len(mature_matches)
        teen_count = len(teen_matches)
        
        # Calculate keyword density (per 1,000 words)
        mature_density = (mature_count / word_count) * 1000
        teen_density = (teen_count / word_count) * 1000
        
        # Calculate severity score (total mature + teen matches)
        severity_score = mature_count + teen_count
        
        # Create content flags
        content_flags = self._create_content_flags(mature_matches, teen_matches, word_count)
        
        # Apply labeling logic with thresholds
        if zero_tolerance_count >= 1:
            label = 'Mature'
        elif mature_density >= 1.5:
            label = 'Mature'
        elif teen_density >= 2.5:
            label = 'Teen'
        else:
            label = 'General'
        
        return label, severity_score, mature_density, teen_density, content_flags
    
    def process_batch_labeling(self) -> bool:
        """
        Process the entire dataset with batch labeling.
        
        Returns:
            bool: True if successful
        """
        print("=" * 80)
        print("🏷️  BATCH LABELING PIPELINE - WEAK SUPERVISION")
        print("=" * 80)
        print(f"Processing {len(self.df):,} transcripts with keyword-density analysis...\n")
        
        # Initialize new columns
        labels = []
        severity_scores = []
        mature_densities = []
        teen_densities = []
        word_counts = []
        content_flags_list = []
        
        # Track metrics
        zero_tolerance_count = 0
        
        # Process each transcript with progress bar
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), 
                            desc="Labeling transcripts", unit="transcript"):
            transcript = str(row['transcript'])
            
            # Calculate word count
            word_count = len(transcript.split())
            word_counts.append(word_count)
            
            # Apply labeling
            label, severity, mature_dens, teen_dens, flags = self.label_transcript(transcript)
            
            # Track zero tolerance triggers
            if 'mature' in label.lower() and severity > 0:
                zero_tolerance_check = len(self._calculate_keyword_matches(
                    transcript.lower(), self.zero_tolerance_pattern))
                if zero_tolerance_check > 0:
                    zero_tolerance_count += 1
            
            # Store results
            labels.append(label)
            severity_scores.append(severity)
            mature_densities.append(round(mature_dens, 3))
            teen_densities.append(round(teen_dens, 3))
            content_flags_list.append(flags)
        
        # Add new columns to DataFrame (appending to existing columns)
        print("\n✅ Labeling complete! Adding new columns to dataset...")
        self.df['Age_Label'] = labels
        self.df['Severity_Score'] = severity_scores
        self.df['Mature_Density'] = mature_densities
        self.df['Teen_Density'] = teen_densities
        self.df['Transcript_Word_Count'] = word_counts
        self.df['Content_Flags'] = content_flags_list
        
        # Calculate metrics
        general_count = labels.count('General')
        teen_count = labels.count('Teen')
        mature_count = labels.count('Mature')
        
        # Calculate average lengths per category
        general_mask = self.df['Age_Label'] == 'General'
        teen_mask = self.df['Age_Label'] == 'Teen'
        mature_mask = self.df['Age_Label'] == 'Mature'
        
        avg_general = self.df[general_mask]['Transcript_Word_Count'].mean() if general_count > 0 else 0
        avg_teen = self.df[teen_mask]['Transcript_Word_Count'].mean() if teen_count > 0 else 0
        avg_mature = self.df[mature_mask]['Transcript_Word_Count'].mean() if mature_count > 0 else 0
        
        self.metrics = LabelingMetrics(
            total_processed=len(self.df),
            general_count=general_count,
            teen_count=teen_count,
            mature_count=mature_count,
            avg_general_length=avg_general,
            avg_teen_length=avg_teen,
            avg_mature_length=avg_mature,
            zero_tolerance_triggers=zero_tolerance_count
        )
        
        print(f"   • Age_Label: Classification result")
        print(f"   • Severity_Score: Total keyword matches")
        print(f"   • Mature_Density: Mature keywords per 1,000 words")
        print(f"   • Teen_Density: Teen keywords per 1,000 words")
        print(f"   • Transcript_Word_Count: Word count for analysis")
        print(f"   • Content_Flags: Timestamped content categories")
        print()
        
        return True
    
    def save_labeled_dataset(self) -> bool:
        """
        Save the labeled dataset to a new CSV file.
        
        Returns:
            bool: True if successful
        """
        print("=" * 80)
        print("💾 SAVING LABELED DATASET")
        print("=" * 80)
        
        try:
            # Save to new file (do not overwrite raw data)
            print(f"📁 Saving to: {self.output_path}")
            self.df.to_csv(self.output_path, index=False)
            
            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully saved {len(self.df):,} rows")
            print(f"📊 File size: {file_size_mb:.2f} MB")
            print(f"📋 Columns: {len(self.df.columns)}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving dataset: {str(e)}")
            return False
    
    def print_summary_statistics(self) -> None:
        """Print comprehensive summary statistics to terminal."""
        if not self.metrics:
            print("⚠️  No metrics available")
            return
        
        print("=" * 80)
        print("📊 LABELING SUMMARY STATISTICS")
        print("=" * 80)
        
        m = self.metrics
        total = m.total_processed
        
        print(f"\n🏷️  LABEL DISTRIBUTION:")
        print("-" * 80)
        print(f"{'Category':<15} {'Count':<12} {'Percentage':<15} {'Avg Length (words)':<20}")
        print("-" * 80)
        print(f"{'General (G)':<15} {m.general_count:<12,} {(m.general_count/total)*100:<14.2f}% {m.avg_general_length:<19.1f}")
        print(f"{'Teen (T)':<15} {m.teen_count:<12,} {(m.teen_count/total)*100:<14.2f}% {m.avg_teen_length:<19.1f}")
        print(f"{'Mature (M)':<15} {m.mature_count:<12,} {(m.mature_count/total)*100:<14.2f}% {m.avg_mature_length:<19.1f}")
        print("-" * 80)
        print(f"{'TOTAL':<15} {total:<12,} {'100.00%':<15}")
        
        print(f"\n⚠️  SAFETY METRICS:")
        print("-" * 80)
        print(f"Zero Tolerance Triggers: {m.zero_tolerance_triggers:,} transcripts")
        print(f"   (Extreme content requiring immediate Mature classification)")
        
        print(f"\n📈 DATASET QUALITY INDICATORS:")
        print("-" * 80)
        print(f"Total Labeled Transcripts: {total:,}")
        print(f"Average Transcript Length: {self.df['Transcript_Word_Count'].mean():.1f} words")
        print(f"Median Transcript Length: {self.df['Transcript_Word_Count'].median():.1f} words")
        print(f"Longest Transcript: {self.df['Transcript_Word_Count'].max():,} words")
        print(f"Shortest Transcript: {self.df['Transcript_Word_Count'].min():,} words")
        
        # Severity score distribution
        print(f"\n🔢 SEVERITY SCORE STATISTICS:")
        print("-" * 80)
        print(f"Mean Severity Score: {self.df['Severity_Score'].mean():.2f}")
        print(f"Median Severity Score: {self.df['Severity_Score'].median():.1f}")
        print(f"Max Severity Score: {self.df['Severity_Score'].max()}")
        
        # Density analysis
        print(f"\n📊 KEYWORD DENSITY ANALYSIS (per 1,000 words):")
        print("-" * 80)
        print(f"Mature Content Density:")
        print(f"   • Mean: {self.df['Mature_Density'].mean():.3f}")
        print(f"   • Median: {self.df['Mature_Density'].median():.3f}")
        print(f"   • 95th Percentile: {self.df['Mature_Density'].quantile(0.95):.3f}")
        print(f"Teen Content Density:")
        print(f"   • Mean: {self.df['Teen_Density'].mean():.3f}")
        print(f"   • Median: {self.df['Teen_Density'].median():.3f}")
        print(f"   • 95th Percentile: {self.df['Teen_Density'].quantile(0.95):.3f}")
        
        # Content flags statistics
        flagged_count = (self.df['Content_Flags'] != 'None').sum()
        print(f"\n🚩 CONTENT FLAGS:")
        print("-" * 80)
        print(f"Transcripts with Flags: {flagged_count:,} ({(flagged_count/total)*100:.2f}%)")
        print(f"Clean Transcripts: {total - flagged_count:,} ({((total-flagged_count)/total)*100:.2f}%)")
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - Silver Training Dataset Ready!")
        print("=" * 80)
        print(f"\n📁 Output File: {self.output_path}")
        print(f"📊 Total Samples: {total:,}")
        print(f"🏷️  Labels: General ({m.general_count:,}), Teen ({m.teen_count:,}), Mature ({m.mature_count:,})")
        print(f"\n💡 Next Steps:")
        print(f"   1. Review the Content_Flags column for quality assurance")
        print(f"   2. Analyze the Severity_Score distribution")
        print(f"   3. Consider filtering high-severity samples if needed")
        print(f"   4. Proceed with model training using Age_Label as target")
        print()
    
    def run_pipeline(self) -> bool:
        """
        Execute the complete weak supervision labeling pipeline.
        
        Returns:
            bool: True if successful
        """
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "   WEAK SUPERVISION BATCH LABELING PIPELINE".center(78) + "║")
        print("║" + "   YouTube Age Classification Research Project".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        
        # Step 1: Load and clean data
        if not self.load_and_clean_data():
            return False
        
        # Step 2: Process batch labeling
        if not self.process_batch_labeling():
            return False
        
        # Step 3: Save labeled dataset
        if not self.save_labeled_dataset():
            return False
        
        # Step 4: Print summary statistics
        self.print_summary_statistics()
        
        return True


def main():
    """Main entry point for the batch labeling pipeline."""
    # ============================================================================
    # CONFIGURATION - Update these paths for your dataset
    # ============================================================================
    INPUT_CSV_PATH = "youtube_data.csv"
    OUTPUT_CSV_PATH = "training_ready_dataset.csv"
    
    # ============================================================================
    # Run Pipeline
    # ============================================================================
    try:
        labeler = WeakSupervisionLabeler(
            csv_path=INPUT_CSV_PATH,
            output_path=OUTPUT_CSV_PATH
        )
        
        success = labeler.run_pipeline()
        
        if success:
            print("=" * 80)
            print("✅ SUCCESS - Labeling pipeline completed successfully!")
            print("=" * 80)
            return 0
        else:
            print("=" * 80)
            print("❌ FAILED - Pipeline encountered errors.")
            print("=" * 80)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
