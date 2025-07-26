"""Console script for llmjammer."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmjammer.llmjammer import Obfuscator, install_git_hooks, create_github_action

app = typer.Typer(help="LLMJammer: Obfuscate your code to confuse LLMs scraping public repositories.")
console = Console()


@app.command()
def jam(
    path: str = typer.Argument(
        ".",
        help="Path to a Python file or directory to obfuscate.",
    ),
    config: Optional[str] = typer.Option(
        None, 
        "--config", 
        "-c", 
        help="Path to custom config file. Default is .jamconfig in current directory."
    ),
    mapping: Optional[str] = typer.Option(
        None, 
        "--mapping", 
        "-m", 
        help="Path to custom mapping file. Default is .jammapping.json in current directory."
    )
):
    """Obfuscate Python code to confuse LLMs scraping public repositories."""
    config_path = Path(config) if config else None
    mapping_path = Path(mapping) if mapping else None
    
    try:
        obfuscator = Obfuscator(config_path, mapping_path)
        file_count = obfuscator.jam(path)
        
        if file_count > 0:
            console.print(f"[green]Successfully obfuscated {file_count} file(s) at {path}[/green]")
        else:
            console.print(f"[yellow]No files were obfuscated at {path}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def unjam(
    path: str = typer.Argument(
        ".",
        help="Path to a Python file or directory to deobfuscate.",
    ),
    config: Optional[str] = typer.Option(
        None, 
        "--config", 
        "-c", 
        help="Path to custom config file. Default is .jamconfig in current directory."
    ),
    mapping: Optional[str] = typer.Option(
        None, 
        "--mapping", 
        "-m", 
        help="Path to custom mapping file. Default is .jammapping.json in current directory."
    )
):
    """Deobfuscate previously obfuscated Python code."""
    config_path = Path(config) if config else None
    mapping_path = Path(mapping) if mapping else None
    
    try:
        obfuscator = Obfuscator(config_path, mapping_path)
        file_count = obfuscator.unjam(path)
        
        if file_count > 0:
            console.print(f"[green]Successfully deobfuscated {file_count} file(s) at {path}[/green]")
        else:
            console.print(f"[yellow]No files were deobfuscated at {path}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def install_hooks(
    hooks_dir: Optional[str] = typer.Option(
        None, 
        "--hooks-dir", 
        help="Path to Git hooks directory. Default is .git/hooks in current repository."
    )
):
    """Install Git hooks for automatic obfuscation/deobfuscation."""
    hooks_path = Path(hooks_dir) if hooks_dir else None
    
    try:
        success = install_git_hooks(hooks_path)
        
        if success:
            console.print("[green]Git hooks installed successfully.[/green]")
        else:
            console.print("[yellow]Failed to install Git hooks.[/yellow]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def setup_github_action():
    """Create a GitHub Action workflow for automatic obfuscation."""
    try:
        success = create_github_action(Path.cwd())
        
        if success:
            console.print("[green]GitHub Action workflow created successfully.[/green]")
        else:
            console.print("[yellow]Failed to create GitHub Action workflow.[/yellow]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def init():
    """Initialize LLMJammer configuration in the current directory."""
    try:
        # Create default config
        config = {
            "exclude": ["tests/", "docs/", "*.md", "*.rst", "setup.py"],
            "obfuscation_level": "medium",  # Options: light, medium, aggressive
            "preserve_docstrings": False,
            "use_encryption": False,
            "encryption_key": "",
        }
        
        with open(".jamconfig", "w") as f:
            import json
            json.dump(config, f, indent=2)
            
        console.print("[green]Initialized LLMJammer configuration in .jamconfig[/green]")
        
        # Offer to install Git hooks
        if typer.confirm("Would you like to install Git hooks for automatic obfuscation?"):
            install_git_hooks()
            
        # Offer to create GitHub Action
        if typer.confirm("Would you like to create a GitHub Action workflow for automatic obfuscation?"):
            create_github_action(Path.cwd())
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def status():
    """Show LLMJammer status and configuration."""
    try:
        # Check for config file
        config_path = Path(".jamconfig")
        config_exists = config_path.exists()
        
        # Check for mapping file
        mapping_path = Path(".jammapping.json")
        mapping_exists = mapping_path.exists()
        
        # Check for Git hooks
        git_dir = Path(".git")
        hooks_installed = False
        if git_dir.exists() and git_dir.is_dir():
            hooks_dir = git_dir / "hooks"
            pre_commit = hooks_dir / "pre-commit"
            post_checkout = hooks_dir / "post-checkout"
            post_merge = hooks_dir / "post-merge"
            
            hooks_installed = (
                pre_commit.exists() and 
                post_checkout.exists() and 
                post_merge.exists() and
                "llmjammer" in pre_commit.read_text()
            )
            
        # Check for GitHub Action
        github_action_path = Path(".github/workflows/llmjammer.yml")
        github_action_exists = github_action_path.exists()
        
        # Display status
        table = Table(title="LLMJammer Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row(
            "Configuration", 
            "Installed" if config_exists else "Missing"
        )
        table.add_row(
            "Mapping File", 
            "Present" if mapping_exists else "Not created yet"
        )
        table.add_row(
            "Git Hooks", 
            "Installed" if hooks_installed else "Not installed"
        )
        table.add_row(
            "GitHub Action", 
            "Configured" if github_action_exists else "Not configured"
        )
        
        console.print(table)
        
        # Show config details if available
        if config_exists:
            try:
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                console.print(Panel(
                    "\n".join([f"{k}: {v}" for k, v in config.items()]),
                    title="Configuration",
                    expand=False
                ))
            except:
                console.print("[yellow]Could not parse configuration file.[/yellow]")
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
