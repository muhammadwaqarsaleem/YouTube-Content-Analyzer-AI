import os
import re
import sys
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt

# Add project root and local directory to sys.path
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
PROJECT_ROOT = THIS_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import required project modules
try:
    from src.youtube_extractor import YouTubeMediaExtractor
    from services.violence_service import ViolenceDetectionService
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running the script from the project root.")

# Custom theme for a "Premium" look
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "model_name": "bold magenta",
    "section_header": "bold yellow",
    "label": "bold blue",
    "safe": "bold green",
    "violent": "bold red",
    "timestamp": "bold white",
})

class ViolenceSpecsManager:
    """
    Singleton class to manage specs and run video violence analysis.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ViolenceSpecsManager, cls).__new__(cls)
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
            
        self.model_spec = {}
        self._parse_specs()
        
        # Initialize Services
        self.extractor = None
        self.violence_service = None
        self._initialize_services()
        
        self._initialized = True

    def _initialize_services(self):
        """Initializes the extraction and violence detection services."""
        try:
            # Initialize Extractor
            self.extractor = YouTubeMediaExtractor(output_dir=str(self.project_root / 'temp'))
            # Initialize Violence Detection Service
            self.violence_service = ViolenceDetectionService()
        except Exception as e:
            self.console.print(f"[warning]Warning: Could not fully initialize services: {e}[/warning]")

    def _parse_specs(self):
        """Parses the Violence Detection section of model_specs.txt."""
        if not self.specs_path.exists():
            return

        with open(self.specs_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Target section #1
        match = re.search(r'##\s*🛡️\s*1\.\s*Violence Detection Model(.*?)(?=##\s*🏷️\s*2\.|---|$)', content, re.DOTALL)
        if match:
            section = match.group(1).strip()
            self.model_spec = {
                "name": "Violence Detection Model",
                "architecture": "N/A",
                "input": {},
                "output": {}
            }
            
            arch_match = re.search(r'\*\*Architecture:\*\*\s*(.*)', section)
            if arch_match: self.model_spec["architecture"] = arch_match.group(1).strip()
            
            input_match = re.search(r'###\s*📥\s*Model Input(.*?)(?=###|---|$)', section, re.DOTALL)
            if input_match:
                items = re.findall(r'\*\s+\*\*(.*?):\*\*\s*(.*)', input_match.group(1))
                self.model_spec["input"] = {k.strip(): v.strip() for k, v in items}
            
            output_match = re.search(r'###\s*📤\s*Model Output(.*?)(?=###|---|$)', section, re.DOTALL)
            if output_match:
                items = re.findall(r'\*\s+\*\*(.*?):\*\*\s*(.*)', output_match.group(1))
                self.model_spec["output"] = {k.strip(): v.strip() for k, v in items}

    def display_specs(self):
        """Displays the parsed model specifications."""
        if not self.model_spec:
            self.console.print("[warning]No violence model specs found.[/warning]")
            return

        self.console.print("\n")
        self.console.print(Panel(
            Text("YouTube Content Analyzer AI - Violence Model Specifications", justify="center", style="bold white on red"),
            expand=False
        ))
        
        self.console.print(f"\n[model_name]🛡️ {self.model_spec['name']}[/model_name]")
        self.console.print(f"[label]Architecture:[/label] {self.model_spec['architecture']}")
        
        table = Table(show_header=True, header_style="bold yellow", box=None, padding=(0, 2))
        table.add_column("Property", style="label", width=20)
        table.add_column("Value", style="info")
        table.add_row("[section_header]INPUT SPECS[/section_header]", "")
        for key, val in self.model_spec["input"].items(): table.add_row(key, val)
        table.add_row("", "")
        table.add_row("[section_header]OUTPUT SPECS[/section_header]", "")
        for key, val in self.model_spec["output"].items(): table.add_row(key, val)
        self.console.print(table)
        self.console.print("-" * 50)

    def analyze_violence(self, url, quiet=False):
        """Runs the full violence analysis pipeline (auth-free waterfall)."""
        if not self.extractor or not self.violence_service:
            if not quiet: self.console.print("[error]Services not initialized properly.[/error]")
            return None

        if not quiet:
            self.console.print(f"\n[info]🚀 Starting violence analysis for:[/info] [bold white]{url}[/bold white]")

        try:
            # ── Auth-Free Frame Fetching (4-tier waterfall) ───────────────
            if not quiet: self.console.print("\n[info]🔄 Running auth-free frame fetching waterfall...[/info]")
            frame_paths, metadata, tier_used = self.extractor.fetch_frames_no_auth(
                url, max_frames=150
            )

            tier_labels = {
                1: "Tier 1 — yt-dlp android/tv_embedded stream",
                2: "Tier 2 — yt-dlp ios/web_embedded stream",
                3: "Tier 3 — YouTube storyboard scraping",
                4: "Tier 4 — Static thumbnail fallback (limited data)",
            }
            tier_icon = "✅" if tier_used < 4 else "⚠️ "
            if not quiet:
                self.console.print(
                    f"[success]{tier_icon} Frame source: {tier_labels[tier_used]}[/success]"
                )

            if not frame_paths:
                if not quiet: self.console.print("[error]❌ No frames could be obtained from any tier.[/error]")
                return None

            if not quiet: self.console.print(f"[info]📸 {len(frame_paths)} frames ready for analysis[/info]")

            # ── Violence Inference ────────────────────────────────────────
            if not quiet: self.console.print("[info]🧠 Running ResNet CNN inference on frames...[/info]")
            result = self.violence_service.analyze_video_frames(
                frame_paths,
                metadata=metadata,
            )
            result['tier_used'] = tier_used
            
            if quiet:
                return (result, metadata)

            # ── Display Results ───────────────────────────────────────────
            title   = metadata.get('title', url)
            channel = metadata.get('channel', 'Unknown')
            dur     = metadata.get('duration', 0)

            status_style = "violent" if result['is_violent'] else "safe"
            status_text  = "VIOLENT CONTENT DETECTED" if result['is_violent'] else "NO SIGNIFICANT VIOLENCE"

            self.console.print(Panel(
                Text(f"Analysis Results: {title}", style="bold white"),
                subtitle=(
                    f"Channel: {channel} | Duration: {dur}s | "
                    f"Source: {tier_labels[tier_used]}"
                ),
                expand=False,
                border_style="red" if result['is_violent'] else "green",
            ))

            # Assessment
            self.console.print(f"\n[bold]Overall Assessment:[/bold] [{status_style}]{status_text}[/{status_style}]")
            self.console.print(f"[bold]Severity Level:[/bold] [bold]{result['severity']}[/bold]")
            self.console.print(f"[bold]Violence Score:[/bold] {result['violence_percentage']:.2f}% of frames")

            # Statistics table
            stat_table = Table(box=None, padding=(0, 2))
            stat_table.add_column("Metric",  style="label")
            stat_table.add_column("Value",   style="info")
            stat_table.add_row("Total Frames Analyzed",    str(result['total_frames']))
            stat_table.add_row("Violent Frames Detected",  f"[violent]{result['violent_frame_count']}[/violent]")
            stat_table.add_row("Peak Raw Confidence",      f"{result['max_confidence']*100:.2f}%")
            stat_table.add_row("Frame Source Tier",        tier_labels[tier_used])
            self.console.print(stat_table)

            # Timeline
            if result['is_violent'] and result.get('violent_frame_timestamps'):
                self.console.print("\n[section_header]🕒 Violence Timeline (Detected Moments)[/section_header]")
                timeline_table = Table(box=None, header_style="bold yellow")
                timeline_table.add_column("Approx Timestamp", style="timestamp")
                timeline_table.add_column("Status",           style="violent")

                shown = sorted(set(round(ts, 1) for ts in result['violent_frame_timestamps']))
                for ts in shown[:15]:
                    timeline_table.add_row(f"{int(ts // 60):02d}:{int(ts % 60):02d}s", "!!! VIOLENT")
                if len(shown) > 15:
                    timeline_table.add_row("...", "+ more moments")
                self.console.print(timeline_table)

            # Recommendation
            self.console.print(f"\n[section_header]📋 Recommendation[/section_header]")
            self.console.print(f" {result['recommendation']}")

            # Tier 4 warning
            if tier_used == 4:
                self.console.print(
                    "\n[warning]⚠️  Analysis ran on static thumbnail frames only "
                    "(all stream methods failed).\n"
                    "   Results may be unreliable. Install 'requests' and 'Pillow' "
                    "and ensure the video is publicly accessible.[/warning]"
                )

            self.console.print("\n" + "="*50)

        except Exception as e:
            if not quiet: self.console.print(f"\n[error]❌ Violence Analysis failed: {e}[/error]")
            return None

def main():
    manager = ViolenceSpecsManager()
    
    while True:
        manager.console.print("\n[bold red]YouTube Violence Detection AI[/bold red]")
        options = [
            "1. View Violence Model Specifications",
            "2. Analyze Video for Violence (ResNet Pipeline)",
            "3. Exit"
        ]
        for opt in options: manager.console.print(opt)
        
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3"])
        
        if choice == "1":
            manager.display_specs()
        elif choice == "2":
            url = Prompt.ask("[label]Enter YouTube URL[/label]")
            manager.analyze_violence(url)
        else:
            manager.console.print("[info]Goodbye![/info]")
            break

if __name__ == "__main__":
    main()
