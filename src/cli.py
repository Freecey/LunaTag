"""Rich CLI for LunaTag."""

import sys
import json
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .audio_analyzer import AudioAnalyzer
from .genre_classifier import GenreClassifier
from .mood_detector import MoodDetector
from .tag_generator import TagGenerator
from .metadata_writer import MetadataWriter


console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """🎵 LunaTag — AI-powered music metadata tagger by Luna 🦋"""
    pass


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
def analyze(file: str, output_format: str):
    """Analyze an audio file and extract features.
    
    Args:
        FILE: Path to the audio file to analyze
    """
    console.print(f"\n🎧 Analyzing [bold cyan]{file}[/bold cyan]...\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting audio features...", total=None)
            
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(file)
            progress.update(task, completed=True)
        
        # Display results
        _display_features(features, output_format)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Show what would be tagged without writing')
@click.option('--backup', is_flag=True, help='Create backup of original file')
def tag(file: str, dry_run: bool, backup: bool):
    """Analyze and write tags to an audio file.
    
    Args:
        FILE: Path to the audio file to tag
    """
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]writing[/green]"
    console.print(f"\n🏷️  Analyzing and {mode} tags to [bold cyan]{file}[/bold cyan]...\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Analysis phase
            progress.add_task("Analyzing audio...", total=None)
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(file)
            
            # Classification phase
            progress.add_task("Classifying genre...", total=None)
            classifier = GenreClassifier()
            genres = classifier.classify(features)
            
            # Mood detection
            progress.add_task("Detecting mood...", total=None)
            mood_detector = MoodDetector()
            moods = mood_detector.detect(features)
            
            # Tag generation
            progress.add_task("Generating tags...", total=None)
            tag_generator = TagGenerator()
            tags = tag_generator.generate_tags(features)
            
            progress.update(progress.tasks[0].id, completed=True)
        
        # Build analysis result
        analysis = {
            'bpm': features.get('bpm'),
            'key': f"{features.get('key', 'C')} {features.get('mode', 'major')}",
            'primary_genre': genres[0][0] if genres else 'unknown',
            'genres': [g for g, _ in genres],
            'moods': moods,
            'tags': tags,
            'energy': features.get('energy'),
            'valence': features.get('valence'),
            'duration': features.get('duration'),
        }
        
        # Display what will be written
        _display_analysis(analysis)
        
        if not dry_run:
            # Write metadata
            writer = MetadataWriter()
            writer.write_from_analysis(file, analysis)
            
            if backup:
                _create_backup(file)
            
            console.print("\n[bold green]✓ Tags written successfully![/bold green]")
        else:
            console.print("\n[yellow]Dry run complete — no files were modified[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('pattern', type=str)
@click.option('--parallel', '-p', is_flag=True, help='Process files in parallel')
@click.option('--recursive', '-r', is_flag=True, help='Search recursively')
def batch(pattern: str, parallel: bool, recursive: bool):
    """Process multiple audio files matching a pattern.
    
    Args:
        PATTERN: Glob pattern for files (e.g., '*.mp3' or './tracks/*.mp3')
    """
    console.print(f"\n📦 Batch processing [bold cyan]{pattern}[/bold cyan]...\n")
    
    # Expand pattern
    path = Path(pattern)
    if path.is_dir():
        files = list(path.rglob('*.mp3') if recursive else path.glob('*.mp3'))
    else:
        base = path.parent if path.parent != Path('.') else Path.cwd()
        files = list(base.glob(path.name))
    
    if not files:
        console.print("[yellow]No files found matching pattern[/yellow]")
        return
    
    console.print(f"[green]Found {len(files)} files to process[/green]\n")
    
    results = []
    for f in files:
        try:
            analyzer = AudioAnalyzer()
            features = analyzer.analyze(str(f))
            
            classifier = GenreClassifier()
            genres = classifier.classify(features)
            
            mood_detector = MoodDetector()
            moods = mood_detector.detect(features)
            
            tag_gen = TagGenerator()
            tags = tag_gen.generate_tags(features)
            
            results.append({
                'file': f.name,
                'bpm': features.get('bpm'),
                'genre': genres[0][0] if genres else 'unknown',
                'moods': moods[:2],
                'tags': tags[:5],
            })
            
            console.print(f"  ✓ [green]{f.name}[/green] — {genres[0][0] if genres else '?'}")
            
        except Exception as e:
            console.print(f"  ✗ [red]{f.name}[/red] — {e}")
            results.append({'file': f.name, 'error': str(e)})
    
    # Summary table
    _display_batch_results(results)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']),
              default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export(file: str, output_format: str, output: Optional[str]):
    """Export metadata from an audio file.
    
    Args:
        FILE: Path to the audio file to export metadata from
    """
    console.print(f"\n📤 Exporting metadata from [bold cyan]{file}[/bold cyan]...\n")
    
    try:
        # Analyze file
        analyzer = AudioAnalyzer()
        features = analyzer.analyze(file)
        
        classifier = GenreClassifier()
        genres = classifier.classify(features)
        
        mood_detector = MoodDetector()
        moods = mood_detector.detect(features)
        
        tag_gen = TagGenerator()
        tags = tag_gen.generate_tags(features)
        
        metadata = {
            'file': file,
            'bpm': features.get('bpm'),
            'key': f"{features.get('key', 'C')} {features.get('mode', 'major')}",
            'genre': genres[0][0] if genres else 'unknown',
            'genres': [g for g, _ in genres],
            'moods': moods,
            'tags': tags,
            'energy': features.get('energy'),
            'duration': features.get('duration'),
        }
        
        # Format output
        if output_format == 'json':
            content = json.dumps(metadata, indent=2)
        else:
            # CSV format
            content = _format_csv(metadata)
        
        if output:
            Path(output).write_text(content)
            console.print(f"[green]✓ Exported to {output}[/green]")
        else:
            console.print(content)
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--limit', type=int, default=10, help='Maximum number of tags to suggest')
def suggest(file: str, limit: int):
    """Suggest tags for an audio file.
    
    Args:
        FILE: Path to the audio file
    """
    console.print(f"\n🔮 Suggesting tags for [bold cyan]{file}[/bold cyan]...\n")
    
    try:
        analyzer = AudioAnalyzer()
        features = analyzer.analyze(file)
        
        tag_gen = TagGenerator()
        suggestions = tag_gen.suggest_tags(features, limit=limit)
        
        console.print("[bold]Suggested tags:[/bold]")
        for i, tag in enumerate(suggestions, 1):
            console.print(f"  {i}. {tag}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)


# Helper functions

def _display_features(features: dict, output_format: str):
    """Display audio features."""
    if output_format == 'json':
        console.print_json(json.dumps(features))
    else:
        table = Table(title="Audio Features")
        table.add_column("Feature", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("BPM", str(features.get('bpm', 'N/A')))
        table.add_row("Key", f"{features.get('key', 'C')} {features.get('mode', 'major')}")
        table.add_row("Energy", f"{features.get('energy', 0):.2f}")
        table.add_row("Valence", f"{features.get('valence', 0):.2f}")
        table.add_row("Duration", f"{features.get('duration', 0):.1f}s")
        table.add_row("Spectral Centroid", f"{features.get('spectral_centroid', 0):.0f}")
        table.add_row("BPM", str(features.get('bpm')))
        
        console.print(table)


def _display_analysis(analysis: dict):
    """Display analysis results."""
    console.print(Panel(
        f"[bold]BPM:[/bold] {analysis.get('bpm', 'N/A')}\n"
        f"[bold]Key:[/bold] {analysis.get('key', 'N/A')}\n"
        f"[bold]Genre:[/bold] {analysis.get('primary_genre', 'N/A')}\n"
        f"[bold]Mood:[/bold] {', '.join(analysis.get('moods', []))}\n"
        f"[bold]Tags:[/bold] {', '.join(analysis.get('tags', [])[:10])}",
        title="Analysis Results",
        border_style="cyan"
    ))


def _display_batch_results(results: List[dict]):
    """Display batch processing results as a table."""
    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("BPM", style="magenta")
    table.add_column("Genre", style="green")
    table.add_column("Mood", style="yellow")
    
    for r in results:
        if 'error' in r:
            table.add_row(r['file'], "[red]Error[/red]", "[red]✗[/red]", "[red]✗[/red]")
        else:
            table.add_row(
                r['file'],
                str(r.get('bpm', 'N/A')),
                r.get('genre', 'N/A'),
                ', '.join(r.get('moods', [])[:2])
            )
    
    console.print(table)


def _format_csv(metadata: dict) -> str:
    """Format metadata as CSV."""
    headers = ['file', 'bpm', 'key', 'genre', 'moods', 'tags', 'duration']
    values = [
        metadata.get('file', ''),
        str(metadata.get('bpm', '')),
        metadata.get('key', ''),
        metadata.get('primary_genre', ''),
        '|'.join(metadata.get('moods', [])),
        '|'.join(metadata.get('tags', [])),
        str(metadata.get('duration', '')),
    ]
    return ','.join(headers) + '\n' + ','.join(values)


def _create_backup(file: str):
    """Create a backup of the original file."""
    import shutil
    original = Path(file)
    backup = original.with_suffix(original.suffix + '.bak')
    shutil.copy2(original, backup)
    console.print(f"[dim]Backup created: {backup}[/dim]")


if __name__ == '__main__':
    cli()
