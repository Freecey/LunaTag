"""Audio analyzer using librosa for feature extraction."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioAnalyzerError(Exception):
    """Custom exception for audio analysis errors."""
    pass


class AudioAnalyzer:
    """Analyze audio files and extract musical features.
    
    Extracts BPM, key, energy, spectral features, and more using librosa.
    """
    
    # Musical keys for key detection
    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MODES = ['major', 'minor']
    
    def __init__(self, sr: int = 22050):
        """Initialize the audio analyzer.
        
        Args:
            sr: Sample rate for audio loading (default: 22050)
        """
        self.sr = sr
        if not LIBROSA_AVAILABLE:
            raise AudioAnalyzerError(
                "librosa is required for audio analysis. "
                "Install it with: pip install librosa"
            )
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """Analyze an audio file and extract all features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing all extracted features
            
        Raises:
            AudioAnalyzerError: If file cannot be loaded
        """
        path = Path(audio_path)
        if not path.exists():
            raise AudioAnalyzerError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            y, sr = librosa.load(str(path), sr=self.sr)
        except Exception as e:
            raise AudioAnalyzerError(f"Failed to load audio: {e}")
        
        if len(y) == 0:
            raise AudioAnalyzerError(f"Audio file is empty: {audio_path}")
        
        # Extract all features
        features = {}
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        features['bpm'] = self._detect_bpm(y, sr)
        features['key'], features['mode'] = self._detect_key(y, sr)
        features['energy'] = self._compute_energy(y)
        features['valence'] = self._compute_valence(y, sr)
        features['tempo'] = features['bpm']
        
        # Spectral features
        spectral = self._compute_spectral_features(y, sr)
        features.update(spectral)
        
        # MFCC and chroma
        features['mfcc'] = self._compute_mfcc(y, sr).tolist()
        features['chroma'] = self._compute_chroma(y, sr).tolist()
        
        return features
    
    def _detect_bpm(self, y: np.ndarray, sr: int) -> float:
        """Detect BPM using onset detection and tempo estimation."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            bpm = float(tempo)
            # Normalize to reasonable range
            while bpm > 200:
                bpm /= 2
            while bpm < 60:
                bpm *= 2
            return round(bpm, 1)
        except Exception:
            return 120.0  # Default fallback
    
    def _detect_key(self, y: np.ndarray, sr: int) -> tuple:
        """Detect musical key using chromagram analysis.
        
        Returns:
            Tuple of (key_note, mode) e.g., ("C", "major")
        """
        try:
            # Compute chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            
            # Map chroma to keys (major and minor profiles)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 3.98, 4.22, 2.42, 5.19, 2.74, 3.52, 2.25, 2.63])
            
            # Find best matching key
            key_idx = 0
            mode = "major"
            
            # Normalize chroma
            chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-10)
            major_profile_norm = major_profile / major_profile.sum()
            minor_profile_norm = minor_profile / minor_profile.sum()
            
            # Calculate correlation for major
            major_corr = np.corrcoef(chroma_norm, major_profile_norm)[0, 1]
            minor_corr = np.corrcoef(chroma_norm, minor_profile_norm)[0, 1]
            
            if major_corr > minor_corr:
                key_idx = np.argmax(chroma_mean)
                mode = "major"
            else:
                # Minor mode - shift indices
                key_idx = np.argmax(chroma_mean)
                mode = "minor"
            
            return self.KEYS[key_idx], mode
        except Exception:
            return "C", "major"
    
    def _compute_energy(self, y: np.ndarray) -> float:
        """Compute RMS energy normalized to 0-1 range."""
        try:
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))
            # Normalize to 0-1 (rough approximation)
            return min(1.0, max(0.0, energy * 10))
        except Exception:
            return 0.5
    
    def _compute_valence(self, y: np.ndarray, sr: int) -> float:
        """Compute valence (mood positivity) from audio features.
        
        Higher valence = more positive/happy mood.
        """
        try:
            # Simple valence estimation based on spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            # Brightness correlates with valence
            brightness = np.mean(spectral_centroid) / 10000
            return min(1.0, max(0.0, brightness))
        except Exception:
            return 0.5
    
    def _compute_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute spectral features for timbre analysis."""
        features = {}
        try:
            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(cent))
            
            # Spectral bandwidth
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth'] = float(np.mean(bw))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff'] = float(np.mean(rolloff))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = float(np.mean(contrast))
            
        except Exception:
            features['spectral_centroid'] = 2000.0
            features['spectral_bandwidth'] = 3000.0
            features['spectral_rolloff'] = 5000.0
            features['spectral_contrast'] = 0.5
        
        return features
    
    def _compute_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCCs for timbre analysis."""
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return np.mean(mfccs, axis=1)
        except Exception:
            return np.zeros(n_mfcc)
    
    def _compute_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute chroma features."""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            return np.mean(chroma, axis=1)
        except Exception:
            return np.zeros(12)
