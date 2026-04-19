# LunaTag — Technical Specification

## Overview

LunaTag is an AI-powered audio metadata tagger that analyzes music files and generates comprehensive metadata for music distribution platforms.

## Architecture

```
lunatag/
├── src/
│   ├── __init__.py          # Package init with version
│   ├── audio_analyzer.py    # Core audio analysis with librosa
│   ├── genre_classifier.py  # Rule-based genre classification
│   ├── mood_detector.py     # Mood/emotion detection
│   ├── tag_generator.py     # Tag generation from features
│   ├── metadata_writer.py   # Mutagen-based tag writing
│   └── cli.py               # Rich CLI interface
├── tests/
│   └── test_lunatag.py      # Unit tests
├── data/
│   └── genre_profiles.json  # Genre classification profiles
├── notebooks/
│   └── audio_tagging.ipynb  # Experimentation notebook
├── requirements.txt
├── .gitignore
├── README.md
└── SPEC.md
```

## Core Components

### 1. Audio Analyzer (`audio_analyzer.py`)

**Purpose:** Extract audio features from music files using librosa.

**Features Extracted:**
- **BPM** — Tempo estimation via onset strength
- **Key** — Musical key detection via chromagram analysis
- **Energy** — RMS energy and spectral centroid
- **Spectral Features** — centroid, bandwidth, contrast, rolloff
- **Timbre** — Spectral characteristics for genre/mood
- **Duration** — Total file length

**Dependencies:** librosa, numpy

**Analysis Pipeline:**
```
Audio File → Load → Preprocess → Feature Extraction → Feature Dict
```

### 2. Genre Classifier (`genre_classifier.py`)

**Purpose:** Classify music genre from audio features using rule-based profiles.

**Classification Logic:**
- Match audio features against genre profiles
- Score each genre based on feature similarity
- Return top matching genres

**Genres Supported:**
- Electronic, House, Techno, Dubstep
- Hip-Hop, Rap, Trap
- Rock, Metal, Punk
- Jazz, Blues, Soul
- Pop, Dance, R&B
- Classical, Ambient, Lo-Fi

**Input:** Audio feature dictionary
**Output:** List of (genre, confidence) tuples

### 3. Mood Detector (`mood_detector.py`)

**Purpose:** Detect emotional qualities from audio characteristics.

**Mood Dimensions:**
- **Energy Level** — high/medium/low
- **Tempo** — fast/medium/slow
- **Brightness** — bright/dark
- **Mood Valence** — happy/sad/chill/intense

**Detected Moods:**
- Energetic, Chill, Melancholic, Intense
- Happy, Dark, Dreamy, Aggressive

### 4. Tag Generator (`tag_generator.py`)

**Purpose:** Generate relevant tags from all extracted features.

**Tag Categories:**
- Genre tags (primary + secondary)
- Mood tags
- Production style tags
- Instrumentation tags
- BPM/key tags (optional)

**Generation Rules:**
- High energy + fast tempo → "energetic", "intense"
- Low energy + slow tempo → "chill", "relaxing"
- Dark spectral features → "dark", "moody"

### 5. Metadata Writer (`metadata_writer.py`)

**Purpose:** Write generated tags to audio file metadata.

**Supported Formats:**
- MP3 (ID3v2)
- FLAC (Vorbis comments)
- OGG (Vorbis comments)
- M4A/AAC (iTunes format)

**Metadata Fields:**
- Genre (primary + secondary)
- BPM
- Key
- Mood
- Custom tags (as comment or grouping)

### 6. CLI (`cli.py`)

**Commands:**

| Command | Description | Options |
|---------|-------------|---------|
| `analyze` | Analyze audio file | `file`, `--format json/yaml` |
| `tag` | Analyze + write tags | `file`, `--dry-run`, `--backup` |
| `batch` | Process multiple files | `pattern`, `--parallel`, `--recursive` |
| `export` | Export metadata | `file`, `--format json/csv`, `--output` |
| `suggest` | Get tag suggestions | `file`, `--limit` |

**CLI Framework:** Rich + Click for beautiful terminal output

## Data Structures

### Audio Features Dict
```python
{
    "bpm": float,
    "key": str,           # e.g., "C major"
    "mode": str,          # "major" or "minor"
    "energy": float,      # 0.0 - 1.0
    "valence": float,     # 0.0 - 1.0 (mood)
    "tempo": float,
    "spectral_centroid": float,
    "spectral_bandwidth": float,
    "spectral_rolloff": float,
    "rms": float,
    "duration": float,
    "chroma": list,
    "mfcc": list,
}
```

### Genre Profiles (JSON)
```json
{
  "electronic": {
    "bpm_range": [110, 150],
    "energy_range": [0.6, 1.0],
    "spectral_features": {...},
    "moods": ["energetic", "intense", "chill"]
  }
}
```

## Configuration

### Environment Variables
- `LUNATAG_DATA_DIR` — Custom data directory
- `LUNATAG_N_JOBS` — Parallel job count for batch
- `LUNATAG_DRY_RUN` — Don't write files (testing)

## Testing Strategy

Unit tests cover:
- Audio feature extraction (mocked)
- Genre classification accuracy
- Mood detection logic
- Tag generation consistency
- Metadata read/write round-trip

## Performance Targets

- Single file analysis: < 5 seconds
- Batch processing: 10+ files/minute
- Memory usage: < 500MB per file

## Future Enhancements

- [ ] ML-based genre classification with trained model
- [ ] Voice/melody separation for better key detection
- [ ] AIMP integration for playlist support
- [ ] Web interface for batch processing
- [ ] Spotify/Apple Music metadata export

---

*Freecey — AI Music Distribution*
