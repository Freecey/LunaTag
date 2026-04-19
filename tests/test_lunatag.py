"""Tests for LunaTag package."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAudioAnalyzer:
    """Tests for AudioAnalyzer class."""
    
    @patch('src.audio_analyzer.LIBROSA_AVAILABLE', False)
    def test_init_requires_librosa(self):
        """Test that initialization fails without librosa."""
        with pytest.raises(Exception):
            from src.audio_analyzer import AudioAnalyzer
            # Would need to reinstall librosa to test properly
    
    @patch('librosa.load')
    def test_analyze_returns_dict(self, mock_load):
        """Test that analyze returns a dictionary."""
        import numpy as np
        mock_load.return_value = (np.random.randn(22050), 22050)
        
        from src.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        # Mock the actual analysis
        analyzer._detect_bpm = Mock(return_value=120.0)
        analyzer._detect_key = Mock(return_value=('C', 'major'))
        analyzer._compute_energy = Mock(return_value=0.5)
        analyzer._compute_valence = Mock(return_value=0.5)
        analyzer._compute_spectral_features = Mock(return_value={})
        analyzer._compute_mfcc = Mock(return_value=np.zeros(13))
        analyzer._compute_chroma = Mock(return_value=np.zeros(12))
        
        result = analyzer.analyze('test.mp3')
        assert isinstance(result, dict)
    
    @patch('librosa.load')
    def test_detect_bpm_returns_float(self, mock_load):
        """Test that BPM detection returns a float."""
        import numpy as np
        mock_load.return_value = (np.random.randn(22050), 22050)
        
        from src.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        bpm = analyzer._detect_bpm(np.random.randn(10000), 22050)
        assert isinstance(bpm, float)
    
    def test_detect_key_returns_tuple(self):
        """Test that key detection returns tuple."""
        from src.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        result = analyzer._detect_key([1, 2, 3], 22050)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGenreClassifier:
    """Tests for GenreClassifier class."""
    
    def test_init_default_profiles(self):
        """Test initialization with default profiles."""
        from src.genre_classifier import GenreClassifier
        classifier = GenreClassifier()
        assert len(classifier.profiles) > 0
        assert 'electronic' in classifier.profiles
    
    def test_classify_returns_list(self):
        """Test that classify returns a list."""
        from src.genre_classifier import GenreClassifier
        classifier = GenreClassifier()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'spectral_centroid': 5000,
        }
        
        result = classifier.classify(features)
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_classify_returns_tuple_elements(self):
        """Test that classify returns tuples with genre and confidence."""
        from src.genre_classifier import GenreClassifier
        classifier = GenreClassifier()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'spectral_centroid': 5000,
        }
        
        result = classifier.classify(features)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)
    
    def test_get_primary_genre(self):
        """Test getting primary genre."""
        from src.genre_classifier import GenreClassifier
        classifier = GenreClassifier()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'spectral_centroid': 5000,
        }
        
        primary = classifier.get_primary_genre(features)
        assert isinstance(primary, str)
        assert primary != ""
    
    def test_get_genre_tags(self):
        """Test getting genre tags list."""
        from src.genre_classifier import GenreClassifier
        classifier = GenreClassifier()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'spectral_centroid': 5000,
        }
        
        tags = classifier.get_genre_tags(features, limit=3)
        assert isinstance(tags, list)
        assert len(tags) <= 3


class TestMoodDetector:
    """Tests for MoodDetector class."""
    
    def test_init(self):
        """Test mood detector initialization."""
        from src.mood_detector import MoodDetector
        detector = MoodDetector()
        assert len(detector.MOOD_PROFILES) > 0
    
    def test_detect_returns_list(self):
        """Test that detect returns a list."""
        from src.mood_detector import MoodDetector
        detector = MoodDetector()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
        }
        
        result = detector.detect(features)
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_detect_includes_mood_strings(self):
        """Test that detected moods are valid mood strings."""
        from src.mood_detector import MoodDetector
        detector = MoodDetector()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
        }
        
        moods = detector.detect(features)
        for mood in moods:
            assert mood in detector.MOOD_PROFILES
    
    def test_get_mood_vector(self):
        """Test mood vector generation."""
        from src.mood_detector import MoodDetector
        detector = MoodDetector()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
        }
        
        vector = detector.get_mood_vector(features)
        assert isinstance(vector, dict)
        assert len(vector) == len(detector.MOOD_PROFILES)


class TestTagGenerator:
    """Tests for TagGenerator class."""
    
    def test_init(self):
        """Test tag generator initialization."""
        from src.tag_generator import TagGenerator
        generator = TagGenerator()
        assert generator.genre_classifier is not None
        assert generator.mood_detector is not None
    
    def test_generate_tags(self):
        """Test tag generation."""
        from src.tag_generator import TagGenerator
        generator = TagGenerator()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
            'spectral_bandwidth': 3000,
            'spectral_rolloff': 6000,
            'spectral_contrast': 0.5,
        }
        
        tags = generator.generate_tags(features)
        assert isinstance(tags, list)
        assert len(tags) > 0
    
    def test_suggest_tags(self):
        """Test tag suggestions."""
        from src.tag_generator import TagGenerator
        generator = TagGenerator()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
        }
        
        suggestions = generator.suggest_tags(features, limit=5)
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
    
    def test_get_tag_summary(self):
        """Test tag summary by category."""
        from src.tag_generator import TagGenerator
        generator = TagGenerator()
        
        features = {
            'bpm': 128,
            'energy': 0.7,
            'valence': 0.6,
            'spectral_centroid': 5000,
            'spectral_bandwidth': 3000,
            'spectral_contrast': 0.5,
        }
        
        summary = generator.get_tag_summary(features)
        assert isinstance(summary, dict)
        assert 'genres' in summary
        assert 'moods' in summary


class TestMetadataWriter:
    """Tests for MetadataWriter class."""
    
    @patch('src.metadata_writer.MUTAGEN_AVAILABLE', False)
    def test_init_requires_mutagen(self):
        """Test that initialization fails without mutagen."""
        with pytest.raises(Exception):
            from src.metadata_writer import MetadataWriter
            # Would need mutagen installed
    
    def test_supported_formats(self):
        """Test that supported formats are defined."""
        from src.metadata_writer import MetadataWriter
        writer = MetadataWriter()
        assert '.mp3' in writer.SUPPORTED_FORMATS
        assert '.flac' in writer.SUPPORTED_FORMATS


class TestCLICommands:
    """Tests for CLI commands."""
    
    def test_cli_imports(self):
        """Test that CLI module imports correctly."""
        from src import cli
        assert cli.cli is not None
    
    def test_cli_has_commands(self):
        """Test that CLI has required commands."""
        from src import cli
        commands = ['analyze', 'tag', 'batch', 'export', 'suggest']
        for cmd in commands:
            assert hasattr(cli.cli, cmd) or cmd in [c.name for c in cli.cli.commands.values()]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
