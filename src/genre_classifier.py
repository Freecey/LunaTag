"""Genre classifier using rule-based profiles."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class GenreClassifierError(Exception):
    """Exception for genre classification errors."""
    pass


class GenreClassifier:
    """Classify music genre using audio feature profiles.
    
    Uses rule-based matching against predefined genre profiles
    to determine the most likely genre(s) for a track.
    """
    
    # Default genre profiles
    DEFAULT_PROFILES = {
        "electronic": {
            "bpm_range": [120, 150],
            "energy_range": [0.6, 1.0],
            "spectral_range": [3000, 8000],
            "moods": ["energetic", "intense", "chill", "euphoric"],
            "keywords": ["electronic", "edm", "dance", "club"]
        },
        "house": {
            "bpm_range": [115, 130],
            "energy_range": [0.5, 0.9],
            "spectral_range": [4000, 10000],
            "moods": ["chill", "groovy", "euphoric"],
            "keywords": ["house", "deep house", "progressive"]
        },
        "techno": {
            "bpm_range": [125, 150],
            "energy_range": [0.7, 1.0],
            "spectral_range": [2000, 6000],
            "moods": ["intense", "dark", "hypnotic"],
            "keywords": ["techno", "industrial", "minimal"]
        },
        "hip_hop": {
            "bpm_range": [80, 120],
            "energy_range": [0.4, 0.8],
            "spectral_range": [2000, 6000],
            "moods": ["cool", "chill", "confident"],
            "keywords": ["hip-hop", "rap", "trap", "boom bap"]
        },
        "trap": {
            "bpm_range": [130, 160],
            "energy_range": [0.6, 1.0],
            "spectral_range": [3000, 8000],
            "moods": ["aggressive", "intense", "dark"],
            "keywords": ["trap", "drill", "808"]
        },
        "rock": {
            "bpm_range": [100, 140],
            "energy_range": [0.5, 0.9],
            "spectral_range": [3000, 7000],
            "moods": ["energetic", "intense", "raw"],
            "keywords": ["rock", "alternative", "indie"]
        },
        "metal": {
            "bpm_range": [100, 180],
            "energy_range": [0.7, 1.0],
            "spectral_range": [4000, 10000],
            "moods": ["aggressive", "intense", "dark"],
            "keywords": ["metal", "death", "black", "doom"]
        },
        "jazz": {
            "bpm_range": [60, 140],
            "energy_range": [0.2, 0.6],
            "spectral_range": [2000, 5000],
            "moods": ["chill", "sophisticated", "smooth"],
            "keywords": ["jazz", "bebop", "swing", "fusion"]
        },
        "soul": {
            "bpm_range": [70, 120],
            "energy_range": [0.3, 0.7],
            "spectral_range": [2000, 6000],
            "moods": ["smooth", "soulful", "uplifting"],
            "keywords": ["soul", "funk", "r&b"]
        },
        "pop": {
            "bpm_range": [100, 130],
            "energy_range": [0.4, 0.8],
            "spectral_range": [3000, 8000],
            "moods": ["happy", "catchy", "uplifting"],
            "keywords": ["pop", " mainstream", "chart"]
        },
        "dance": {
            "bpm_range": [120, 140],
            "energy_range": [0.6, 1.0],
            "spectral_range": [4000, 10000],
            "moods": ["energetic", "euphoric", "happy"],
            "keywords": ["dance", "disco", "electro"]
        },
        "ambient": {
            "bpm_range": [60, 100],
            "energy_range": [0.1, 0.4],
            "spectral_range": [1000, 4000],
            "moods": ["chill", "dreamy", "melancholic", "peaceful"],
            "keywords": ["ambient", "atmospheric", "experimental"]
        },
        "classical": {
            "bpm_range": [40, 180],
            "energy_range": [0.2, 0.7],
            "spectral_range": [1000, 6000],
            "moods": ["sophisticated", "emotional", "peaceful"],
            "keywords": ["classical", "orchestral", "chamber"]
        },
        "lofi": {
            "bpm_range": [70, 100],
            "energy_range": [0.2, 0.5],
            "spectral_range": [2000, 5000],
            "moods": ["chill", "relaxing", "melancholic", "nostalgic"],
            "keywords": ["lofi", "chillhop", "beats"]
        },
        "rnb": {
            "bpm_range": [60, 110],
            "energy_range": [0.3, 0.6],
            "spectral_range": [2000, 6000],
            "moods": ["smooth", "soulful", "romantic"],
            "keywords": ["r&b", "quiet storm", "contemporary"]
        }
    }
    
    def __init__(self, profiles_path: Optional[str] = None):
        """Initialize the genre classifier.
        
        Args:
            profiles_path: Path to custom genre profiles JSON file
        """
        self.profiles = self.DEFAULT_PROFILES.copy()
        
        if profiles_path:
            self._load_profiles(profiles_path)
    
    def _load_profiles(self, path: str) -> None:
        """Load custom genre profiles from JSON."""
        try:
            with open(path, 'r') as f:
                custom = json.load(f)
                self.profiles.update(custom)
        except Exception as e:
            raise GenreClassifierError(f"Failed to load profiles: {e}")
    
    def classify(self, features: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Classify genre from audio features.
        
        Args:
            features: Dictionary containing audio features
            
        Returns:
            List of (genre, confidence) tuples, sorted by confidence
        """
        if not NUMPY_AVAILABLE:
            return [("electronic", 0.5)]  # Fallback
        
        scores = {}
        
        for genre, profile in self.profiles.items():
            score = self._calculate_match_score(features, profile)
            scores[genre] = score
        
        # Sort by score descending
        sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores to probabilities
        total = sum(s[1] for s in sorted_genres) + 1e-10
        normalized = [(g, s/total) for g, s in sorted_genres]
        
        return normalized[:5]  # Top 5 genres
    
    def _calculate_match_score(self, features: Dict[str, Any], profile: Dict) -> float:
        """Calculate how well features match a genre profile."""
        score = 0.0
        weights = {
            'bpm': 0.3,
            'energy': 0.4,
            'spectral': 0.3
        }
        
        # BPM match
        bpm = features.get('bpm', 120)
        bpm_range = profile.get('bpm_range', [80, 160])
        if bpm_range[0] <= bpm <= bpm_range[1]:
            score += weights['bpm'] * 1.0
        else:
            # Penalty for out of range
            dist = min(abs(bpm - bpm_range[0]), abs(bpm - bpm_range[1]))
            score += weights['bpm'] * max(0, 1 - dist/50)
        
        # Energy match
        energy = features.get('energy', 0.5)
        energy_range = profile.get('energy_range', [0.2, 0.8])
        if energy_range[0] <= energy <= energy_range[1]:
            score += weights['energy'] * 1.0
        else:
            dist = min(abs(energy - energy_range[0]), abs(energy - energy_range[1]))
            score += weights['energy'] * max(0, 1 - dist*2)
        
        # Spectral centroid match
        centroid = features.get('spectral_centroid', 3000)
        spectral_range = profile.get('spectral_range', [2000, 6000])
        if spectral_range[0] <= centroid <= spectral_range[1]:
            score += weights['spectral'] * 1.0
        else:
            dist = min(abs(centroid - spectral_range[0]), abs(centroid - spectral_range[1]))
            score += weights['spectral'] * max(0, 1 - dist/2000)
        
        return score
    
    def get_primary_genre(self, features: Dict[str, Any]) -> str:
        """Get the primary genre for a track."""
        results = self.classify(features)
        return results[0][0] if results else "unknown"
    
    def get_genre_tags(self, features: Dict[str, Any], limit: int = 5) -> List[str]:
        """Get genre tags as a simple list of strings."""
        results = self.classify(features)
        return [g for g, _ in results[:limit]]
