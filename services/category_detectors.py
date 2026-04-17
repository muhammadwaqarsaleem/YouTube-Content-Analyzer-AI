"""
Category-Specific Override Detectors
Implements detection logic for all 16 YouTube categories
"""

from .category_override_detector import CategoryOverrideDetector
from .keyword_loader_mixin import KeywordLoaderMixin
from typing import List, Tuple, Dict


# ============== TIER 1: HIGH-PRIORITY CATEGORIES ==============

class FilmAnimationDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Film & Animation content"""
    
    def __init__(self):
        self.keyword_file = 'film_animation_keywords.txt'
        super().__init__('Film & Animation', priority=4)
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'trailer', 'official trailer', 'clip', 'scene', 'movie', 'film',
                'animation', 'animated', 'cartoon', 'teaser', 'preview',
                'full movie', 'streaming now', 'in theaters', 'box office',
                'marvel', 'dc comics', 'pixar', 'disney', 'warner bros',
                'universal pictures', 'paramount', 'sony pictures'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            'marvel entertainment', 'dc', 'pixar', 'waltdisneyanimationstudios',
            'warner bros.', 'universal pictures', 'paramount pictures',
            'sony pictures entertainment', 'studio ghibli', 'dreamworks animation'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Trailers: 2-3 min, Episodes: 20-45 min, Movies: 90-180 min
        if not duration:
            return 0, []
        elif 120 <= duration <= 180 or 1200 <= duration <= 2700 or 5400 <= duration <= 10800:
            return 2, ['duration_match']
        elif 60 <= duration <= 300 or 900 <= duration <= 3600:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = [' - ', '(', ')', ':', '|']  # Movie name formatting
        return self._check_title_pattern(title, patterns)


class GamingDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Gaming content"""
    
    def __init__(self):
        self.keyword_file = 'gaming_keywords.txt'
        super().__init__('Gaming', priority=5)
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'gameplay', 'walkthrough', 'let\'s play', 'gaming', 'esports',
                'streamer', 'twitch', 'youtube gaming', 'live stream',
                'minecraft', 'fortnite', 'gta v', 'call of duty', 'league of legends',
                'valorant', 'apex legends', 'roblox', 'among us', 'overwatch',
                'boss fight', 'ending', 'secret', 'easter egg', 'review gameplay'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            'pewdiepie', 'ninja', 'markiplier', 'jacksepticeye',
            'game theory', 'film theory', 'ign', 'gamespot',
            'polygon', 'kotaku', 'eurogamer'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Gaming videos typically 10-60 minutes
        if not duration:
            return 0, []
        elif 600 <= duration <= 3600:
            return 2, ['duration_match']
        elif 300 <= duration <= 5400:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['ep.', 'episode', 'part ', 'day ', ' Gameplay', ' Review']
        return self._check_title_pattern(title, patterns)


class HowtoStyleDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Howto & Style content"""
    
    def __init__(self):
        self.keyword_file = 'howto_style_keywords.txt'
        super().__init__('Howto & Style', priority=10)
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'tutorial', 'how to', 'diy', 'guide', 'makeup', 'cooking',
                'recipe', 'hack', 'tips', 'tricks', 'fashion', 'style',
                'beauty', 'skincare', 'haircare', 'workout', 'fitness',
                'home decor', 'craft', 'knitting', 'sewing', 'painting',
                'life hack', '5-minute crafts', 'buzzfeed'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            '5-minute crafts', 'buzzfeed', 'tasty', 'bon appétit',
            'jeffree star', 'james charles', 'nikkie tutorials',
            'home depot', 'lowe\'s'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Tutorials typically 5-20 minutes
        if not duration:
            return 0, []
        elif 300 <= duration <= 1200:
            return 2, ['duration_match']
        elif 180 <= duration <= 1800:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['how to', 'tutorial', 'diy', 'ways to', 'tips for']
        return self._check_title_pattern(title, patterns)


class EducationDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Education content"""
    
    def __init__(self):
        self.keyword_file = 'education_keywords.txt'
        super().__init__('Education', priority=6)
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'lecture', 'course', 'learn', 'education', 'documentary',
                'explained', 'history', 'science', 'mathematics', 'physics',
                'chemistry', 'biology', 'literature', 'philosophy',
                'ted talk', 'crash course', 'khan academy', 'university',
                'college', 'school', 'academic', 'research', 'study'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            'ted', 'tedx', 'khan academy', 'crashcourse',
            'veritasium', 'vsauce', 'numberphile', 'periodic videos',
            'mit opencourseware', 'yale courses', 'stanford', 'harvard'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Educational content 10-60 minutes
        if not duration:
            return 0, []
        elif 600 <= duration <= 3600:
            return 2, ['duration_match']
        elif 300 <= duration <= 5400:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['introduction to', 'lecture', 'course', 'explained', 'documentary']
        return self._check_title_pattern(title, patterns)


# ============== TIER 2: MEDIUM-PRIORITY CATEGORIES ==============

class ComedyDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Comedy content"""
    
    def __init__(self):
        self.keyword_file = 'comedy_keywords.txt'
        super().__init__('Comedy', priority=7)
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'comedy', 'funny', 'humor', 'joke', 'parody', 'skit',
                'stand-up', 'prank', 'compilation', 'lol', 'hilarious',
                'laugh', 'comedic', 'satire', 'mockumentary', 'improv',
                'snl', 'saturday night live', 'conan', 'fallon'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            'snl', 'saturday night live', 'team coco', 'the tonight show',
            'comedy central', 'funny or die', 'collegehumor'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Comedy typically 3-15 minutes
        if not duration:
            return 0, []
        elif 180 <= duration <= 900:
            return 2, ['duration_match']
        elif 60 <= duration <= 1200:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['funny moments', 'best of', 'compilation', 'stand-up']
        return self._check_title_pattern(title, patterns)


class PeopleBlogsDetector(CategoryOverrideDetector):
    """Detects People & Blogs content"""
    
    def __init__(self):
        super().__init__('People & Blogs', priority=14)
    
    def _get_keywords(self) -> List[str]:
        return [
            'vlog', 'daily', 'life', 'story', 'my experience', 'personal',
            'day in my life', 'routine', 'morning routine', 'night routine',
            'q&a', 'ask me anything', 'storytime', 'update', 'chat'
        ]
    
    def _get_channels(self) -> List[str]:
        return []  # Individual creators, hard to predefine
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Vlogs typically 5-20 minutes
        if not duration:
            return 0, []
        elif 300 <= duration <= 1200:
            return 2, ['duration_match']
        elif 180 <= duration <= 1800:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['vlog', 'day in', 'my ', 'i ', 'we ']  # Personal pronouns
        return self._check_title_pattern(title, patterns)


class ScienceTechDetector(CategoryOverrideDetector):
    """Detects Science & Technology content"""
    
    def __init__(self):
        super().__init__('Science & Technology', priority=9)
    
    def _get_keywords(self) -> List[str]:
        return [
            'tech', 'review', 'unboxing', 'gadget', 'smartphone', 'ai',
            'robot', 'space', 'physics', 'technology', 'innovation',
            'tesla', 'apple', 'google', 'microsoft', 'amazon',
            'cpu', 'gpu', 'benchmark', 'specifications', 'features'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'mkbhd', 'marques brownlee', 'veritasium', 'vsauce',
            'mrwhosetheboss', 'unbox therapy', 'linus tech tips',
            'dave2d', 'austin evans', 'i just wanna make things'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Tech reviews 8-20 minutes
        if not duration:
            return 0, []
        elif 480 <= duration <= 1200:
            return 2, ['duration_match']
        elif 300 <= duration <= 1800:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['review', 'unboxing', 'vs ', 'comparison', 'hands-on']
        return self._check_title_pattern(title, patterns)


class AutosVehiclesDetector(CategoryOverrideDetector):
    """Detects Autos & Vehicles content"""
    
    def __init__(self):
        super().__init__('Autos & Vehicles', priority=11)
    
    def _get_keywords(self) -> List[str]:
        return [
            'car', 'auto', 'vehicle', 'review', 'test drive', 'supercar',
            'electric vehicle', 'ev', 'hybrid', 'automotive',
            'tesla', 'bmw', 'mercedes', 'ferrari', 'porsche', 'lamborghini',
            'ford', 'chevrolet', 'toyota', 'honda', 'nissan',
            'top gear', 'motor trend', 'car and driver'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'top gear', 'motor trend', 'car and driver', 'roadshow',
            'throttle house', 'savagegeese', 'engineering explained'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Car reviews 10-25 minutes
        if not duration:
            return 0, []
        elif 600 <= duration <= 1500:
            return 2, ['duration_match']
        elif 300 <= duration <= 2400:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['review', 'test drive', 'vs ', 'comparison', 'buying guide']
        return self._check_title_pattern(title, patterns)


# ============== TIER 3: SPECIALIZED CATEGORIES ==============

class TravelEventsDetector(CategoryOverrideDetector):
    """Detects Travel & Events content"""
    
    def __init__(self):
        super().__init__('Travel & Events', priority=12)
    
    def _get_keywords(self) -> List[str]:
        return [
            'travel', 'destination', 'vacation', 'tourism', 'guide',
            'adventure', 'explore', 'trip', 'journey', 'backpacking',
            'hotel', 'resort', 'airbnb', 'flight', 'airport',
            'paris', 'london', 'tokyo', 'new york', 'dubai', 'bali'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'lonely planet', 'national geographic travel', 'rick steves',
            'drew binsky', 'migrationology', 'expert vagabond'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Travel videos 10-30 minutes
        if not duration:
            return 0, []
        elif 600 <= duration <= 1800:
            return 2, ['duration_match']
        elif 300 <= duration <= 2700:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['travel guide', 'things to do', 'visit ', 'exploring ']
        return self._check_title_pattern(title, patterns)


class PetsAnimalsDetector(CategoryOverrideDetector):
    """Detects Pets & Animals content"""
    
    def __init__(self):
        super().__init__('Pets & Animals', priority=13)
    
    def _get_keywords(self) -> List[str]:
        return [
            'pet', 'animal', 'dog', 'cat', 'wildlife', 'cute',
            'funny animals', 'puppy', 'kitten', 'bird', 'fish',
            'hamster', 'rabbit', 'horse', 'elephant', 'lion', 'tiger',
            'zoo', 'aquarium', 'nature', 'animal compilation'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'the dodo', 'animal planet', 'discovery channel',
            'nat geo wild', 'bbcearth', 'the ellen show'  # animal segments
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Animal videos vary widely (1-15 minutes typical)
        if not duration:
            return 0, []
        elif 60 <= duration <= 900:
            return 2, ['duration_match']
        elif 30 <= duration <= 1200:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['cute ', 'funny ', 'compilation', 'tries ']
        return self._check_title_pattern(title, patterns)


class ShowsDetector(CategoryOverrideDetector):
    """Detects Shows content"""
    
    def __init__(self):
        super().__init__('Shows', priority=8)
    
    def _get_keywords(self) -> List[str]:
        return [
            'episode', 'season', 'series', 'full episode', 'talk show',
            'tv show', 'web series', 'sitcom', 'drama', 'comedy series',
            'netflix', 'hulu', 'amazon prime', 'hbo', 'disney+',
            'the office', 'friends', 'stranger things', 'game of thrones'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'netflix', 'hulu', 'amazon prime video', 'hbo',
            'disney plus', 'bbc', 'nbc', 'abc', 'cbs', 'fox'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # TV episodes 20-60 minutes
        if not duration:
            return 0, []
        elif 1200 <= duration <= 3600:
            return 2, ['duration_match']
        elif 900 <= duration <= 4500:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['episode', 'season', 's', 'e', 'full episode']
        return self._check_title_pattern(title, patterns)


class NonprofitsActivismDetector(CategoryOverrideDetector):
    """Detects Nonprofits & Activism content"""
    
    def __init__(self):
        super().__init__('Nonprofits & Activism', priority=15)
    
    def _get_keywords(self) -> List[str]:
        return [
            'nonprofit', 'charity', 'cause', 'donate', 'fundraiser',
            'awareness', 'activism', 'volunteer', 'social cause',
            'climate change', 'human rights', 'equality', 'justice',
            'red cross', 'unesco', 'unicef', 'greenpeace', 'wwf',
            'save the children', 'doctors without borders'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'unicef', 'red cross', 'greenpeace', 'wwf', 'unesco',
            'doctors without borders', 'oxfam', 'save the children',
            'world wildlife fund', 'the nature conservancy'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Nonprofit videos 3-15 minutes
        if not duration:
            return 0, []
        elif 180 <= duration <= 900:
            return 2, ['duration_match']
        elif 60 <= duration <= 1200:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        patterns = ['support', 'donate', 'help', 'save', 'protect']
        return self._check_title_pattern(title, patterns)


# ============== PRIORITY TIER 0: CRITICAL OVERRIDES ==============
# These have highest priority due to time-sensitivity or strong verification

class MusicDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Music videos - official releases and artist content"""
    
    def __init__(self):
        self.keyword_file = 'music_keywords.txt'
        super().__init__('Music', priority=1)  # Highest priority
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                # Music video formats
                'official video', 'official music video', 'lyric video', 'audio',
                'feat.', 'ft.', 'featuring', 'prod. by', 'directed by',
                'album', 'single', 'ep', 'lp', 'remix', 'cover',
                'music video', 'mv', 'teaser', 'trailer', 'visualizer',
                
                # Top global artists (2024-2025)
                'taylor swift', 'ariana grande', 'justin bieber', 'ed sheeran',
                'beyoncé', 'rihanna', 'drake', 'the weeknd', 'dua lipa',
                'billie eilish', 'harry styles', 'adele', 'bruno mars',
                'lady gaga', 'katy perry', 'selena gomez', 'shawn mendes',
                'olivia rodrigo', 'doja cat', 'post malone', 'bad bunny',
                'j balvin', 'karol g', 'rosalía', 'anuel aa', 'daddy yankee',
                'bts', 'blackpink', 'stray kids', 'twice', 'newjeans',
                'jung kook', 'jimin', 'v', 'suga', 'rm', 'j-hope', 'jin',
                'jisoo', 'jennie', 'rosé', 'lisa', 'coldplay', 'imagine dragons',
                'maroon 5', 'one republic', 'the chainsmokers', 'kygo', 'avicii',
                
                # Music genres
                'pop', 'rock', 'hip hop', 'rap', 'r&b', 'country', 'jazz',
                'electronic', 'edm', 'house', 'techno', 'dubstep', 'trap',
                'reggaeton', 'latin', 'k-pop', 'j-pop', 'indie', 'alternative',
                'metal', 'punk', 'folk', 'blues', 'soul', 'funk', 'disco',
                
                # Music industry terms
                'grammy', 'billboard', 'hot 100', 'spotify', 'apple music',
                'youtube music', 'soundcloud', 'tidal', 'vevo',
                'record label', 'universal music', 'sony music', 'warner music',
                'atlantic records', 'columbia records', 'republic records',
                
                # Music production
                'producer', 'songwriter', 'composer', 'arranger',
                'mixing', 'mastering', 'recording', 'studio session',
                'live performance', 'concert', 'tour', 'world tour',
                'acoustic', 'unplugged', 'live session', 'studio version'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        # Major artist channels and record labels
        return [
            'taylor swift', 'beyoncé', 'drake', 'the weeknd', 'ariana grande',
            'justin bieber', 'billie eilish', 'dua lipa', 'post malone',
            'vevo', 'universal music group', 'sony music', 'warner music group',
            'republic records', 'atlantic records', 'columbia records',
            'capitol records', 'def jam recordings', 'epic records'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Music videos typically 2-6 minutes
        if not duration:
            return 0, []
        elif 120 <= duration <= 360:
            return 2, ['typical_music_video_duration']
        elif 90 <= duration <= 480:
            return 1, ['acceptable_duration']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        # Artist - Song Title format
        patterns = [' - ', '(', ')', '[', ']', 'official', 'video', 'audio']
        score, indicators = self._check_title_pattern(title, patterns)
        
        # Bonus for common music video patterns
        if 'official' in title.lower() and 'video' in title.lower():
            score += 1
            indicators.append('official_video_tag')
        
        return score, indicators


class NewsDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects News & Politics content - breaking news and political coverage"""
    
    def __init__(self):
        self.keyword_file = 'news_politics_keywords.txt'
        super().__init__('News & Politics', priority=2)  # Very high priority
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                # Breaking news & urgency
                'breaking', 'news', 'report', 'update', 'live',
                'just in', 'developing', 'exclusive', 'investigation',
                
                # Conflict & military terminology
                'war', 'attack', 'drone', 'strike', 'military',
                'missile', 'bombing', 'artillery', 'combat', 'frontline',
                'ceasefire', 'invasion', 'occupation', 'liberation',
                'wounded', 'killed', 'victims', 'casualties', 'fatalities',
                
                # Government & politics
                'president', 'government', 'minister', 'parliament',
                'congress', 'senate', 'prime minister', 'secretary',
                'ambassador', 'diplomat', 'legislation', 'bill', 'law',
                'election', 'vote', 'political', 'policy', 'campaign',
                'referendum', 'ballot', 'candidate', 'nominee',
                
                # Crisis & international relations
                'crisis', 'conflict', 'tension', 'sanctions',
                'embargo', 'negotiation', 'summit', 'treaty',
                'alliance', 'coalition', 'un resolution', 'nato',
                'geopolitical', 'foreign policy', 'diplomatic ties',
                
                # News reporting phrases
                'says', 'reported', 'according to', 'sources',
                'statement', 'announcement', 'press conference',
                'briefing', 'correspondent', 'on the ground',
                'exclusive interview', 'special report', 'documentary',
                
                # News organizations & programs
                'bbc', 'cnn', 'reuters', 'ap news', 'associated press',
                'evening news', 'nightly news', 'morning show',
                'news hour', 'news tonight', 'world news',
                
                # Specific news topics
                'economy', 'inflation', 'interest rates', 'gdp',
                'climate change', 'natural disaster', 'earthquake',
                'pandemic', 'outbreak', 'vaccine', 'public health',
                'supreme court', 'lawsuit', 'verdict', 'trial'
            ]
        
        return keywords
    
    def _get_channels(self) -> List[str]:
        return [
            'bbc news', 'cnn', 'al jazeera english', 'sky news',
            'fox news', 'msnbc', 'reuters', 'associated press',
            'abc news', 'cbs news', 'nbc news', 'pbs newshour',
            'euronews', 'dw news', 'france 24', 'cgtn', 'ndtv'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # News clips typically 2-15 minutes
        if not duration:
            return 0, []
        elif 120 <= duration <= 900:
            return 2, ['typical_news_duration']
        elif 60 <= duration <= 1800:
            return 1, ['acceptable_duration']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        # News title patterns (attribution, urgency)
        patterns = [':', ' - ', 'says', 'reports', 'breaking', 'update']
        score, indicators = self._check_title_pattern(title, patterns)
        
        # Check for attribution pattern (X says Y)
        if 'says' in title.lower() or 'according to' in title.lower():
            score += 1
            indicators.append('attribution_pattern')
        
        return score, indicators


class SportsDetector(CategoryOverrideDetector):
    """Detects Sports content - especially combat sports"""
    
    def __init__(self):
        super().__init__('Sports', priority=3)  # High priority
    
    def _get_keywords(self) -> List[str]:
        return [
            # Combat sports (MMA, Wrestling, Boxing)
            'ufc', 'wwe', 'wrestling', 'mma', 'boxing', 'kickboxing',
            'fight night', 'bellator', 'one championship', 'pfl',
            'grappling', 'jiu jitsu', 'bjj', 'muay thai', 'karate',
            'taekwondo', 'judo', 'sambo', 'bare knuckle',
            'khabib', 'mcgregor', 'conor', 'nurmagomedov',
            'ufc fight', 'wwe raw', 'smackdown', 'ppv', 'main event',
            'knockout', 'ko', 'tko', 'submission', 'guillotine', 'rear naked',
            'octagon', 'ring', 'corner', 'referee', 'stoppage',
            
            # Team sports
            'football', 'basketball', 'soccer', 'baseball', 'hockey',
            'rugby', 'cricket', 'volleyball', 'handball', 'water polo',
            'nfl', 'nba', 'mlb', 'nhl', 'premier league', 'la liga',
            'champions league', 'euroleague', 'super bowl', 'world cup',
            'touchdown', 'goal', 'home run', 'slam dunk', 'hat trick',
            
            # Individual sports
            'tennis', 'golf', 'swimming', 'athletics', 'olympics',
            'formula 1', 'f1', 'motogp', 'nascar', 'rally',
            'boxing', 'cycling', 'marathon', 'triathlon', 'ironman',
            'skiing', 'snowboarding', 'skateboarding', 'surfing',
            'gymnastics', 'weightlifting', 'wrestling freestyle',
            
            # Sports events & competitions
            'championship', 'playoffs', 'finals', 'tournament', 'league',
            'grand prix', 'masters', 'open', 'cup', 'trophy',
            'medal', 'gold', 'silver', 'bronze', 'podium',
            'world record', 'personal best', 'season opener',
            
            # Sports terminology
            'team', 'coach', 'player', 'athlete', 'stadium', 'arena',
            'manager', 'captain', 'rookie', 'mvp', 'all-star',
            'transfer', 'trade', 'contract', 'free agent', 'draft',
            'overtime', 'extra time', 'penalty', 'foul', 'injury',
            'highlights', 'recap', 'analysis', 'post-game', 'interview'
        ]
    
    def _get_channels(self) -> List[str]:
        return [
            'ufc', 'wwe', 'espn', 'nfl', 'nba', 'mlb', 'nhl',
            'tennis channel', 'pga tour', 'fifa', 'formula 1',
            'sky sports', 'bt sport', 'dazn', 'bein sports',
            'fox sports', 'cbssports', 'nbc sports'
        ]
    
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        # Combat sports: 15-60 min, Team sports: 1-3 hours
        if not duration:
            return 0, []
        elif 900 <= duration <= 3600 or 5400 <= duration <= 10800:
            return 2, ['typical_sports_duration']
        elif 600 <= duration <= 14400:
            return 1, ['sports_event_length']
        else:
            return 0, []
    
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        # Sports title patterns (vs, fight card, game number)
        patterns = [' vs ', ' - ', 'round', 'knockout', 'ko', 'submission',
                   'game', 'match', 'set', 'inning', 'period', 'quarter']
        return self._check_title_pattern(title, patterns)


# ============== FACTORY FUNCTION ==============

def get_all_detectors() -> List[CategoryOverrideDetector]:
    """Return list of all category detectors in priority order"""
    detectors = [
        # Tier 0 - Critical Overrides (highest priority)
        MusicDetector(),      # Priority 1 - Artist verification
        NewsDetector(),       # Priority 2 - Breaking news time-sensitive
        SportsDetector(),     # Priority 3 - Live sports events
        
        # Tier 1 - High Priority
        FilmAnimationDetector(),
        GamingDetector(),
        HowtoStyleDetector(),
        EducationDetector(),
        
        # Tier 2 - Medium Priority  
        ComedyDetector(),
        PeopleBlogsDetector(),
        ScienceTechDetector(),
        AutosVehiclesDetector(),
        
        # Tier 3 - Specialized
        TravelEventsDetector(),
        PetsAnimalsDetector(),
        ShowsDetector(),
        NonprofitsActivismDetector()
    ]
    
    # Sort by priority (lower number = higher priority)
    detectors.sort(key=lambda x: x.priority)
    
    return detectors
