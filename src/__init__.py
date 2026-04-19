"""LunaTag — AI-powered music metadata tagger"""

__version__ = "0.1.0"
__author__ = "Luna"
__app_name__ = "LunaTag"

from src.audio_analyzer import AudioAnalyzer
from src.genre_classifier import GenreClassifier
from src.mood_detector import MoodDetector
from src.tag_generator import TagGenerator
from src.metadata_writer import MetadataWriter

__all__ = [
    "AudioAnalyzer",
    "GenreClassifier", 
    "MoodDetector",
    "TagGenerator",
    "MetadataWriter",
]
