import os
import re
import sys
import copy
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

# Add project root to sys.path to allow importing from src and services
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import required project modules
try:
    from src.youtube_extractor import YouTubeMediaExtractor
    from services.category_service import CategoryPredictionService
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running the script from the project root or its subdirectories.")

# Custom theme for a "Premium" look
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "model_name": "bold magenta",
    "section_header": "bold yellow",
    "label": "bold blue",
    "prob_high": "bold green",
    "prob_med": "yellow",
    "prob_low": "dim white",
})

class ModelSpecsManager:
    """
    Singleton class to manage specs and run video category analysis.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelSpecsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, specs_path=None):
        if self._initialized:
            return
        
        self.console = Console(theme=custom_theme)
        self.project_root = PROJECT_ROOT
        
        if specs_path is None:
            self.specs_path = self.project_root / "model_specs.txt"
        else:
            self.specs_path = Path(specs_path)
            
        self.models = []
        self._parse_specs()
        
        # Initialize Services
        self.extractor = None
        self.category_service = None
        self._initialize_services()
        
        self._initialized = True

    def _initialize_services(self):
        """Initializes the extraction and prediction services."""
        try:
            # Initialize Extractor (configured to hit Deno server at 127.0.0.1:4416)
            self.extractor = YouTubeMediaExtractor(output_dir=str(self.project_root / 'temp'))
            
            # Initialize Category Prediction Service (New LightGBM Model)
            new_model_path = self.project_root / 'models' / 'yt_category_model.pkl'
            
            self.category_service = CategoryPredictionService(
                model_path=str(new_model_path) if new_model_path.exists() else 'models/yt_category_model.pkl'
            )
        except Exception as e:
            self.console.print(f"[warning]Warning: Could not fully initialize analysis services: {e}[/warning]")

    def _parse_specs(self):
        """Parses the model_specs.txt file into a structured list of models."""
        if not self.specs_path.exists():
            return

        with open(self.specs_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r'---', content)
        for section in sections:
            section = section.strip()
            if not section: continue
            
            match = re.search(r'##\s*(?:[^\w\s]*)\s*\d+\.\s*(.*)', section)
            if match:
                model_data = {
                    "name": match.group(1).strip(),
                    "architecture": "N/A",
                    "input": {},
                    "output": {},
                    "classes": []
                }
                
                arch_match = re.search(r'\*\*Architecture:\*\*\s*(.*)', section)
                if arch_match: model_data["architecture"] = arch_match.group(1).strip()
                
                input_match = re.search(r'###\s*📥\s*Model Input(.*?)(?=###|---|$)', section, re.DOTALL)
                if input_match:
                    items = re.findall(r'\*\s+\*\*(.*?):\*\*\s*(.*)', input_match.group(1))
                    model_data["input"] = {k.strip(): v.strip() for k, v in items}
                
                output_match = re.search(r'###\s*📤\s*Model Output(.*?)(?=###|---|$)', section, re.DOTALL)
                if output_match:
                    items = re.findall(r'\*\s+\*\*(.*?):\*\*\s*(.*)', output_match.group(1))
                    model_data["output"] = {k.strip(): v.strip() for k, v in items}
                
                classes_match = re.search(r'\*   \*\*Classes \(16 total\):\*\*\n(.*?)(?=\n---|\n###|$)', section, re.DOTALL)
                if classes_match:
                    classes = re.findall(r'\d+\.\s*(.*)', classes_match.group(1))
                    model_data["classes"] = [c.strip() for c in classes]

                self.models.append(model_data)

    def display_specs(self):
        """Displays the parsed model specifications."""
        self.console.print("\n")
        self.console.print(Panel(
            Text("YouTube Content Analyzer AI - Model Specifications", justify="center", style="bold white on blue"),
            expand=False
        ))
        
        for model in self.models:
            self.console.print(f"\n[model_name]🛡️ {model['name']}[/model_name]")
            self.console.print(f"[label]Architecture:[/label] {model['architecture']}")
            
            table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 2))
            table.add_column("Property", style="label", width=20)
            table.add_column("Value", style="info")
            table.add_row("[section_header]INPUT SPECS[/section_header]", "")
            for key, val in model["input"].items(): table.add_row(key, val)
            table.add_row("", "")
            table.add_row("[section_header]OUTPUT SPECS[/section_header]", "")
            for key, val in model["output"].items(): table.add_row(key, val)
            self.console.print(table)
            self.console.print("-" * 50)

    def analyze_video(self, url, quiet=False):
        """Runs the full analysis pipeline for a given YouTube URL."""
        if not self.extractor or not self.category_service:
            if not quiet: self.console.print("[error]Analysis services not initialized properly.[/error]")
            return None

        if not quiet:
            self.console.print(f"\n[info]🚀 Starting analysis for:[/info] [bold white]{url}[/bold white]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                # Task 1: Metadata Extraction
                task1 = progress.add_task("Extracting metadata (auth-free)...", total=None)
                video_id = self.extractor.get_video_id(url)
                metadata = self.extractor._fetch_metadata_no_auth(url, video_id)
                progress.update(task1, completed=True)
                
                # Task 2: Thumbnail Extraction
                task2 = progress.add_task("Fetching thumbnail (auth-free)...", total=None)
                frame_paths = self.extractor._thumbnail_fallback(video_id, min_copies=1)
                thumb_path = frame_paths[0] if frame_paths else None
                progress.update(task2, completed=True)
                
                # Task 3: Pure ML Model Inference
                task3 = progress.add_task("Running LightGBM Inference pipeline...", total=None)
                result = self.category_service.predict_category(thumb_path, metadata)
                progress.update(task3, completed=True)

            if quiet:
                return (result, metadata)

            # --- Display Results ---
            self.console.print(Panel(
                Text(f"Analysis Results: {metadata['title']}", style="bold white"),
                subtitle=f"Channel: {metadata['channel']} | Duration: {metadata['duration']}s",
                expand=False,
                border_style="cyan"
            ))

            # Primary Category
            color = "prob_high" if result['primary_probability'] > 0.6 else "prob_med"
            self.console.print(f"\n[bold]Predicted Category:[/bold] [{color}]{result['primary_category']}[/{color}]")
            self.console.print(f"[bold]Confidence:[/bold] [{color}]{result['primary_probability']*100:.2f}%[/{color}]")

            # (Overrides Scorecard removed for Pure Model Inference)
            
            # Probability Distribution Table
            self.console.print("\n[section_header]Probability Distribution (Top 10)[/section_header]")
            table = Table(box=None, header_style="bold yellow")
            table.add_column("Category", style="white")
            table.add_column("Probability", justify="right")
            table.add_column("Confidence Visualizer", width=30)

            for cat in result['all_categories'][:10]:
                prob = cat['probability']
                bar_len = int(prob * 25)
                color = "green" if prob > 0.5 else ("yellow" if prob > 0.2 else "red")
                bar = f"[{color}]" + "█" * bar_len + "[/]" + "░" * (25 - bar_len)
                table.add_row(cat['category'], f"{prob*100:5.2f}%", bar)
            
            self.console.print(table)
            self.console.print("\n" + "="*50)

        except Exception as e:
            if not quiet: self.console.print(f"\n[error]❌ Analysis failed: {e}[/error]")
            # import traceback
            # traceback.print_exc()
            return None

def main():
    manager = ModelSpecsManager()
    
    while True:
        manager.console.print("\n[bold cyan]YouTube Content Analyzer AI[/bold cyan]")
        options = [
            "1. View Model Specifications",
            "2. Analyze Video Category (LightGBM Pipeline)",
            "3. Exit"
        ]
        for opt in options: manager.console.print(opt)
        
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3"])
        
        if choice == "1":
            manager.display_specs()
        elif choice == "2":
            url = Prompt.ask("[label]Enter YouTube URL[/label]")
            manager.analyze_video(url)
        else:
            manager.console.print("[info]Goodbye![/info]")
            break

if __name__ == "__main__":
    main()
