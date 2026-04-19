# 🎵 LunaTag — by Luna for Luna 🦋

*AI-powered music metadata tagger for the independent artist*

---

## ✨ Qu'est-ce que c'est?

LunaTag is mon petit outil intelligent pour tagger automatiquement mes fichiers audio avec tout ce dont j'ai besoin pour la distribution : **genre**, **mood**, **BPM**, **key**, et **tags** pertinents.

Plus besoin de tagger à la main chaque track — je laisse l'IA analyser et générer les métadonnées pour moi! 🎧✨

## 🎯 Ce que je peux faire

| Commande | Description |
|----------|-------------|
| `analyze` | Analyser un fichier audio et extraire les features |
| `tag` | Générer et écrire les tags dans le fichier |
| `batch` | Traiter plusieurs fichiers d'un coup |
| `export` | Exporter les métadonnées en JSON/CSV |
| `suggest` | Suggérer des tags basés sur les features extraites |

## 🚀 Démarrage rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Analyser un fichier
python -m src.cli analyze mon_track.mp3

# Tagger un fichier avec métadonnées
python -m src.cli tag mon_track.mp3

# Batch processing
python -m src.cli batch ./tracks/*.mp3

# Exporter les métadonnées
python -m src.cli export mon_track.mp3 --format json
```

## 🧠 Comment ça marche

LunaTag utilise **librosa** pour analyser les caractéristiques audio :

- **BPM** — détecté via onset detection
- **Key** — estimé via chromagram et harmonic analysis  
- **Energy** — RMS et spectral contrast
- **Genre** — classification basée sur les profiles audio
- **Mood** — détecté via tempo, timbre, et dynamics
- **Tags** — générés automatiquement depuis les features extraites

## 📋 Requirements

- Python 3.9+
- librosa
- mutagen
- rich
- click
- Pillow
- pytest
- numpy

## 💾 Installation

```bash
git clone git@github.com:Freecey/LunaTag.git
cd LunaTag
pip install -r requirements.txt
```

## 🎼 Features

- 🎵 Analyse multi-fichier avec batch processing
- 🏷️ Écriture native des métadonnées ID3/Vorbis
- 📊 Export JSON/CSV pour catalogues
- 🎨 CLI riche avec colors et émojis
- 🔮 Suggestions intelligentes de tags
- 🌍 Style Luna — avec amour et attention aux détails

---

*Créé avec 💜 par Luna, pour Luna*

*Freecey — AI Music Distribution*
