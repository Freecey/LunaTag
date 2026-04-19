"""Tag generator combining all features for comprehensive tagging."""

from typing import Dict, List, Any, Optional

from .genre_classifier import GenreClassifier
from .mood_detector import MoodDetector


class TagGenerator:
    """Generate comprehensive tags from audio analysis results.
    
    Combines genre classification, mood detection, and audio features
    to produce a rich set of tags for music distribution.
    """
    
    # Production style tags
    PRODUCTION_TAGS = {
        "high_energy_production": lambda f: f.get('energy', 0) > 0.7,
        "clean_production": lambda f: f.get('spectral_contrast', 0) > 0.6,
        "raw_production": lambda f: f.get('spectral_contrast', 0) < 0.4,
        "warm_analog": lambda f: f.get('spectral_centroid', 3000) < 2500,
        "digital": lambda f: f.get('spectral_centroid', 3000) > 5000,
        "compressed": lambda f: f.get('spectral_rolloff', 5000) > 8000,
        "acoustic": lambda f: f.get('energy', 0.5) < 0.4 and f.get('spectral_centroid', 3000) < 3000,
    }
    
    # Instrumentation tags based on spectral features
    INSTRUMENTATION_TAGS = {
        "bass_heavy": lambda f: f.get('spectral_bandwidth', 3000) < 2500,
        "bright": lambda f: f.get('spectral_centroid', 3000) > 5000,
        "dark": lambda f: f.get('spectral_centroid', 3000) < 2500,
        "mid_range_focused": lambda f: 2500 <= f.get('spectral_centroid', 3000) <= 4500,
        "full_spectrum": lambda f: f.get('spectral_bandwidth', 3000) > 4000,
    }
    
    # Tempo-based tags
    TEMPO_TAGS = {
        "slow": lambda f: f.get('bpm', 120) < 80,
        "mid_tempo": lambda f: 80 <= f.get('bpm', 120) <= 110,
        "upbeat": lambda f: 110 < f.get('bpm', 120) <= 130,
        "fast": lambda f: f.get('bpm', 120) > 130,
        "club_tempo": lambda f: 120 <= f.get('bpm', 120) <= 130,
    }
    
    def __init__(self):
        """Initialize tag generator with sub-detectors."""
        self.genre_classifier = GenreClassifier()
        self.mood_detector = MoodDetector()
    
    def generate_tags(self, features: Dict[str, Any], 
                      include_genre: bool = True,
                      include_mood: bool = True,
                      include_production: bool = True,
                      include_instrumentation: bool = True,
                      include_tempo: bool = True,
                      limit: Optional[int] = None) -> List[str]:
        """Generate all tags from audio features.
        
        Args:
            features: Audio features dictionary
            include_genre: Include genre tags
            include_mood: Include mood tags
            include_production: Include production style tags
            include_instrumentation: Include instrumentation tags
            include_tempo: Include tempo-based tags
            limit: Maximum number of tags to return
            
        Returns:
            List of generated tags
        """
        tags = []
        
        # Genre tags
        if include_genre:
            genre_tags = self.genre_classifier.get_genre_tags(features)
            tags.extend(genre_tags)
        
        # Mood tags
        if include_mood:
            mood_tags = self.mood_detector.detect(features)
            tags.extend(mood_tags)
        
        # Production style tags
        if include_production:
            production_tags = self._generate_production_tags(features)
            tags.extend(production_tags)
        
        # Instrumentation tags
        if include_instrumentation:
            instr_tags = self._generate_instrumentation_tags(features)
            tags.extend(instr_tags)
        
        # Tempo tags
        if include_tempo:
            tempo_tags = self._generate_tempo_tags(features)
            tags.extend(tempo_tags)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        if limit:
            return unique_tags[:limit]
        
        return unique_tags
    
    def _generate_production_tags(self, features: Dict[str, Any]) -> List[str]:
        """Generate production style tags."""
        tags = []
        for tag, condition in self.PRODUCTION_TAGS.items():
            try:
                if condition(features):
                    tags.append(tag)
            except Exception:
                pass
        return tags
    
    def _generate_instrumentation_tags(self, features: Dict[str, Any]) -> List[str]:
        """Generate instrumentation tags."""
        tags = []
        for tag, condition in self.INSTRUMENTATION_TAGS.items():
            try:
                if condition(features):
                    tags.append(tag)
            except Exception:
                pass
        return tags
    
    def _generate_tempo_tags(self, features: Dict[str, Any]) -> List[str]:
        """Generate tempo-based tags."""
        tags = []
        for tag, condition in self.TEMPO_TAGS.items():
            try:
                if condition(features):
                    tags.append(tag)
            except Exception:
                pass
        return tags
    
    def suggest_tags(self, features: Dict[str, Any], limit: int = 10) -> List[str]:
        """Suggest top tags based on audio features.
        
        Args:
            features: Audio features dictionary
            limit: Maximum number of tags to suggest
            
        Returns:
            List of suggested tags
        """
        return self.generate_tags(features, limit=limit)
    
    def get_tag_summary(self, features: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get a summary of tags by category.
        
        Returns:
            Dictionary with categories as keys and tag lists as values
        """
        return {
            "genres": self.genre_classifier.get_genre_tags(features),
            "moods": self.mood_detector.detect(features),
            "production": self._generate_production_tags(features),
            "instrumentation": self._generate_instrumentation_tags(features),
            "tempo": self._generate_tempo_tags(features),
        }
