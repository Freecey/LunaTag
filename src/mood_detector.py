"""Mood detector using audio characteristics."""

from typing import Dict, List, Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class MoodDetector:
    """Detect mood/emotion from audio features.
    
    Analyzes tempo, energy, spectral characteristics, and valence
    to determine the emotional quality of a track.
    """
    
    # Mood definitions with thresholds
    MOOD_PROFILES = {
        "energetic": {
            "energy_min": 0.6,
            "tempo_min": 120,
            "valence_min": 0.4,
            "spectral_min": 4000
        },
        "chill": {
            "energy_max": 0.4,
            "tempo_max": 100,
            "valence_range": [0.3, 0.7],
            "spectral_max": 4000
        },
        "intense": {
            "energy_min": 0.7,
            "tempo_min": 130,
            "valence_max": 0.6,
            "spectral_min": 5000
        },
        "happy": {
            "energy_min": 0.4,
            "tempo_min": 100,
            "valence_min": 0.6,
            "spectral_min": 3500
        },
        "dark": {
            "energy_max": 0.5,
            "valence_max": 0.4,
            "spectral_max": 4000
        },
        "dreamy": {
            "energy_max": 0.4,
            "tempo_max": 110,
            "valence_min": 0.4,
            "spectral_max": 3500
        },
        "melancholic": {
            "energy_max": 0.4,
            "valence_max": 0.35,
            "spectral_max": 4000
        },
        "aggressive": {
            "energy_min": 0.75,
            "tempo_min": 140,
            "valence_max": 0.35
        },
        "uplifting": {
            "energy_min": 0.5,
            "valence_min": 0.65,
            "tempo_min": 100
        },
        "peaceful": {
            "energy_max": 0.3,
            "tempo_max": 90,
            "valence_min": 0.4
        },
        "euphoric": {
            "energy_min": 0.7,
            "valence_min": 0.7,
            "tempo_min": 120
        },
        "groovy": {
            "energy_min": 0.4,
            "tempo_range": [100, 130],
            "valence_min": 0.45
        },
        "moody": {
            "valence_max": 0.4,
            "spectral_max": 4500
        },
        "sophisticated": {
            "energy_range": [0.3, 0.6],
            "valence_range": [0.4, 0.65],
            "spectral_range": [2500, 5500]
        },
        "romantic": {
            "energy_max": 0.5,
            "tempo_max": 100,
            "valence_min": 0.5
        }
    }
    
    def __init__(self):
        """Initialize mood detector."""
        pass
    
    def detect(self, features: Dict[str, Any]) -> List[str]:
        """Detect moods from audio features.
        
        Args:
            features: Dictionary containing audio features
            
        Returns:
            List of detected moods
        """
        if not NUMPY_AVAILABLE:
            return ["chill"]
        
        bpm = features.get('bpm', 120)
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        centroid = features.get('spectral_centroid', 3000)
        
        detected_moods = []
        
        # Check each mood profile
        for mood, profile in self.MOOD_PROFILES.items():
            if self._matches_profile(bpm, energy, valence, centroid, profile):
                detected_moods.append(mood)
        
        # Always ensure at least one mood
        if not detected_moods:
            detected_moods.append(self._infer_default_mood(energy, valence, bpm))
        
        return detected_moods[:4]  # Limit to top 4 moods
    
    def _matches_profile(self, bpm: float, energy: float, 
                         valence: float, centroid: float,
                         profile: Dict) -> bool:
        """Check if audio features match a mood profile."""
        try:
            # Energy constraints
            if 'energy_min' in profile and energy < profile['energy_min']:
                return False
            if 'energy_max' in profile and energy > profile['energy_max']:
                return False
            if 'energy_range' in profile:
                er = profile['energy_range']
                if not (er[0] <= energy <= er[1]):
                    return False
            
            # Tempo constraints
            if 'tempo_min' in profile and bpm < profile['tempo_min']:
                return False
            if 'tempo_max' in profile and bpm > profile['tempo_max']:
                return False
            if 'tempo_range' in profile:
                tr = profile['tempo_range']
                if not (tr[0] <= bpm <= tr[1]):
                    return False
            
            # Valence constraints
            if 'valence_min' in profile and valence < profile['valence_min']:
                return False
            if 'valence_max' in profile and valence > profile['valence_max']:
                return False
            if 'valence_range' in profile:
                vr = profile['valence_range']
                if not (vr[0] <= valence <= vr[1]):
                    return False
            
            # Spectral constraints
            if 'spectral_min' in profile and centroid < profile['spectral_min']:
                return False
            if 'spectral_max' in profile and centroid > profile['spectral_max']:
                return False
            if 'spectral_range' in profile:
                sr = profile['spectral_range']
                if not (sr[0] <= centroid <= sr[1]):
                    return False
            
            return True
        except Exception:
            return False
    
    def _infer_default_mood(self, energy: float, valence: float, bpm: float) -> str:
        """Infer a default mood when no profile matches."""
        if energy > 0.7:
            return "energetic" if bpm > 110 else "intense"
        elif energy < 0.3:
            return "chill" if valence > 0.4 else "melancholic"
        elif valence > 0.6:
            return "happy"
        else:
            return "moody"
    
    def get_mood_vector(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get a mood vector with intensity scores.
        
        Returns:
            Dictionary of mood -> intensity (0.0 to 1.0)
        """
        moods = self.detect(features)
        vector = {}
        
        for mood in self.MOOD_PROFILES.keys():
            vector[mood] = 1.0 if mood in moods else 0.0
        
        return vector
