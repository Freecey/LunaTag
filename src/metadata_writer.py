"""Metadata writer using mutagen for audio file tagging."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    from mutagen import File
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.oggvorbis import OggVorbis
    from mutagen.mp4 import MP4
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


class MetadataWriterError(Exception):
    """Exception for metadata writing errors."""
    pass


class MetadataWriter:
    """Write metadata tags to audio files using mutagen.
    
    Supports MP3 (ID3v2), FLAC, OGG, and M4A formats.
    """
    
    SUPPORTED_FORMATS = ['.mp3', '.flac', '.ogg', '.m4a', '.wav', '.aiff']
    
    def __init__(self):
        """Initialize metadata writer."""
        if not MUTAGEN_AVAILABLE:
            raise MetadataWriterError(
                "mutagen is required for metadata writing. "
                "Install it with: pip install mutagen"
            )
    
    def write_tags(self, audio_path: str, 
                   genre: Optional[str] = None,
                   bpm: Optional[float] = None,
                   key: Optional[str] = None,
                   mood: Optional[List[str]] = None,
                   tags: Optional[List[str]] = None,
                   title: Optional[str] = None,
                   artist: Optional[str] = None,
                   album: Optional[str] = None,
                   comment: Optional[str] = None) -> bool:
        """Write metadata tags to an audio file.
        
        Args:
            audio_path: Path to the audio file
            genre: Primary genre tag
            bpm: BPM value
            key: Musical key
            mood: List of mood tags
            tags: List of custom tags
            title: Track title
            artist: Artist name
            album: Album name
            comment: Comment text
            
        Returns:
            True if successful
            
        Raises:
            MetadataWriterError: If writing fails
        """
        path = Path(audio_path)
        if not path.exists():
            raise MetadataWriterError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file with mutagen
            audio = File(str(path))
            if audio is None:
                raise MetadataWriterError(f"Unsupported file format: {audio_path}")
            
            # Write common tags
            if genre:
                self._set_genre(audio, genre)
            
            if bpm:
                self._set_bpm(audio, bpm)
            
            if key:
                self._set_key(audio, key)
            
            if mood:
                self._set_mood(audio, mood)
            
            if tags:
                self._set_tags(audio, tags)
            
            if title:
                audio['title'] = title
            
            if artist:
                audio['artist'] = artist
            
            if album:
                audio['album'] = album
            
            if comment:
                audio['comment'] = comment
            
            # Save the file
            audio.save()
            return True
            
        except Exception as e:
            raise MetadataWriterError(f"Failed to write metadata: {e}")
    
    def _set_genre(self, audio, genre: str) -> None:
        """Set genre tag based on file format."""
        if isinstance(audio, MP3):
            audio['genre'] = genre
        elif isinstance(audio, (FLAC, OggVorbis)):
            audio['genre'] = genre
        elif isinstance(audio, MP4):
            audio['genre'] = genre
        else:
            audio['genre'] = genre
    
    def _set_bpm(self, audio, bpm: float) -> None:
        """Set BPM tag."""
        # BPM is stored as a comment or custom tag depending on format
        if isinstance(audio, MP3):
            audio['bpm'] = str(int(bpm))
        elif isinstance(audio, MP4):
            audio['bpm'] = str(int(bpm))
        else:
            # Store BPM in comments for other formats
            if audio.get('comment'):
                comment = audio['comment']
                if isinstance(comment, list):
                    comment.append(f"BPM: {int(bpm)}")
                else:
                    audio['comment'] = [comment, f"BPM: {int(bpm)}"]
            else:
                audio['comment'] = f"BPM: {int(bpm)}"
    
    def _set_key(self, audio, key: str) -> None:
        """Set musical key tag."""
        if isinstance(audio, MP3):
            audio['key'] = key
        elif isinstance(audio, MP4):
            audio['key'] = key
        else:
            # Store in comment for other formats
            if audio.get('comment'):
                comment = audio['comment']
                if isinstance(comment, list):
                    comment.append(f"Key: {key}")
                else:
                    audio['comment'] = [comment, f"Key: {key}"]
            else:
                audio['comment'] = f"Key: {key}"
    
    def _set_mood(self, audio, moods: List[str]) -> None:
        """Set mood tags."""
        mood_str = ', '.join(moods)
        if isinstance(audio, MP3):
            audio['mood'] = mood_str
        elif isinstance(audio, MP4):
            audio['mood'] = mood_str
        else:
            if audio.get('comment'):
                comment = audio['comment']
                if isinstance(comment, list):
                    comment.append(f"Mood: {mood_str}")
                else:
                    audio['comment'] = [comment, f"Mood: {mood_str}"]
            else:
                audio['comment'] = f"Mood: {mood_str}"
    
    def _set_tags(self, audio, tags: List[str]) -> None:
        """Set custom tags."""
        tags_str = ', '.join(tags)
        if isinstance(audio, MP3):
            audio['tag'] = tags_str
        elif isinstance(audio, MP4):
            audio['tag'] = tags_str
        else:
            if audio.get('comment'):
                comment = audio['comment']
                if isinstance(comment, list):
                    comment.append(f"Tags: {tags_str}")
                else:
                    audio['comment'] = [comment, f"Tags: {tags_str}"]
            else:
                audio['comment'] = f"Tags: {tags_str}"
    
    def read_tags(self, audio_path: str) -> Dict[str, Any]:
        """Read existing tags from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary of existing tags
        """
        path = Path(audio_path)
        if not path.exists():
            raise MetadataWriterError(f"Audio file not found: {audio_path}")
        
        try:
            audio = File(str(path))
            if audio is None:
                return {}
            
            tags = {}
            
            # Read common tags
            if audio.get('title'):
                tags['title'] = str(audio['title'])
            if audio.get('artist'):
                tags['artist'] = str(audio['artist'])
            if audio.get('album'):
                tags['album'] = str(audio['album'])
            if audio.get('genre'):
                tags['genre'] = str(audio['genre'])
            
            return tags
            
        except Exception as e:
            raise MetadataWriterError(f"Failed to read metadata: {e}")
    
    def write_from_analysis(self, audio_path: str, 
                           analysis: Dict[str, Any],
                           dry_run: bool = False) -> bool:
        """Write all tags from audio analysis results.
        
        Args:
            audio_path: Path to the audio file
            analysis: Complete analysis dictionary
            dry_run: If True, don't actually write (for testing)
            
        Returns:
            True if successful
        """
        if dry_run:
            return True
        
        genre = analysis.get('primary_genre')
        bpm = analysis.get('bpm')
        key = analysis.get('key')
        mood = analysis.get('moods', [])
        tags = analysis.get('tags', [])
        
        return self.write_tags(
            audio_path=audio_path,
            genre=genre,
            bpm=bpm,
            key=key,
            mood=mood,
            tags=tags
        )
