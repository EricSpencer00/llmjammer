"""Console script for llmjammer."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmjammer.llmjammer import (
    Obfuscator, 
    install_git_hooks, 
    create_github_action, 
    check_git_hooks_installed,
    find_git_hooks_dir
)

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
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if hooks already exist"
    )
):
    """Install Git hooks for automatic obfuscation/deobfuscation."""
    # If no hooks directory is specified, find it automatically
    hooks_path = None
    if hooks_dir is None:
        hooks_path = find_git_hooks_dir()
        if hooks_path is None:
            console.print("[red]Error: Not in a Git repository.[/red]")
            console.print("[yellow]Please specify a hooks directory with --hooks-dir or navigate to a Git repository.[/yellow]")
            raise typer.Exit(code=1)
    else:
        hooks_path = Path(hooks_dir)
    
    # Check if hooks are already installed
    if not force:
        hooks_status = check_git_hooks_installed()
        if hooks_status["installed"]:
            console.print("[yellow]Git hooks are already installed.[/yellow]")
            console.print("Use --force to reinstall.")
            return
    
    # Install the hooks
    try:
        success = install_git_hooks(hooks_path)
        if success:
            console.print("[green]Successfully installed Git hooks for automatic obfuscation/deobfuscation.[/green]")
            console.print(f"Hooks installed in: {hooks_path}")
            console.print("\nThese hooks will:")
            console.print(" - Obfuscate Python files when you commit")
            console.print(" - Ensure all code is obfuscated when you push")
            console.print(" - Deobfuscate code when you checkout, merge, pull, or rebase")
        else:
            console.print("[red]Failed to install Git hooks.[/red]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error installing Git hooks: {str(e)}[/red]")
        raise typer.Exit(code=1)
    
    # Install the hooks
    try:
        success = install_git_hooks(hooks_path)
        if success:
            console.print("[green]Successfully installed Git hooks for automatic obfuscation/deobfuscation.[/green]")
            console.print(f"Hooks installed in: {hooks_path}")
            console.print("\nThese hooks will:")
            console.print(" - Obfuscate Python files when you commit")
            console.print(" - Ensure all code is obfuscated when you push")
            console.print(" - Deobfuscate code when you checkout, merge, pull, or rebase")
        else:
            console.print("[red]Failed to install Git hooks.[/red]")
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error installing Git hooks: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def hooks_status():
    """Check the status of Git hooks for LLMJammer."""
    hooks_status = check_git_hooks_installed()
    
    if hooks_status["installed"]:
        console.print("[green]LLMJammer Git hooks are properly installed.[/green]")
        console.print(f"Hooks directory: {hooks_status['hooks_dir']}")
        
        # Display installed hooks
        table = Table(title="Installed Hooks")
        table.add_column("Hook", style="cyan")
        table.add_column("Status", style="green")
        
        for hook in hooks_status["installed_hooks"]:
            table.add_row(hook, "✓ Installed")
        
        console.print(table)
    else:
        console.print("[yellow]LLMJammer Git hooks are not fully installed.[/yellow]")
        
        if "hooks_dir" in hooks_status:
            console.print(f"Hooks directory: {hooks_status['hooks_dir']}")
            
            # Display missing hooks
            table = Table(title="Hook Status")
            table.add_column("Hook", style="cyan")
            table.add_column("Status", style="red")
            
            for hook in hooks_status["missing_hooks"]:
                table.add_row(hook, "✗ Missing")
                
            for hook in hooks_status["installed_hooks"]:
                table.add_row(hook, "✓ Installed")
                
            console.print(table)
            
            console.print("\nTo install the missing hooks, run:")
            console.print("[bold]llmjammer install-hooks[/bold]")
        else:
            console.print("[red]Not in a Git repository.[/red]")
            console.print("Please navigate to a Git repository to install hooks.")


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
            hooks_dir = find_git_hooks_dir()
            if hooks_dir is None:
                console.print("[yellow]Not in a Git repository, skipping hooks installation.[/yellow]")
            else:
                success = install_git_hooks(hooks_dir)
                if success:
                    console.print("[green]Git hooks installed successfully.[/green]")
                else:
                    console.print("[yellow]Failed to install Git hooks.[/yellow]")
            
        # Offer to create GitHub Action
        if typer.confirm("Would you like to create a GitHub Action workflow for automatic obfuscation?"):
            success = create_github_action(Path.cwd())
            if success:
                console.print("[green]GitHub Action workflow created successfully.[/green]")
            else:
                console.print("[yellow]Failed to create GitHub Action workflow.[/yellow]")
            
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
        hooks_status = check_git_hooks_installed()
        hooks_installed = hooks_status["installed"]
            
        # Check for GitHub Action
        github_action_path = Path(".github/workflows/llmjammer.yml")
        github_action_exists = github_action_path.exists()
        
        # Display status
        table = Table(title="LLMJammer Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        table.add_row(
            "Configuration", 
            "✓ Installed" if config_exists else "✗ Missing"
        )
        table.add_row(
            "Mapping File", 
            "✓ Present" if mapping_exists else "✗ Not created yet"
        )
        table.add_row(
            "Git Hooks", 
            "✓ Installed" if hooks_installed else "✗ Not installed"
        )
        table.add_row(
            "GitHub Action", 
            "✓ Configured" if github_action_exists else "✗ Not configured"
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
        
        # Show detailed hooks status if in a Git repository
        if "hooks_dir" in hooks_status:
            if not hooks_installed:
                console.print("\n[yellow]Git hooks are not fully installed.[/yellow]")
                console.print("Run [bold]llmjammer install-hooks[/bold] to install them.")
            else:
                console.print("\n[green]Git hooks are properly installed.[/green]")
                console.print("Your code will be automatically:")
                console.print(" - Obfuscated before commits and pushes")
                console.print(" - Deobfuscated after checkouts, merges, and pulls")
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


@app.command()
def git_ready(
    operation: str = typer.Argument(
        ...,
        help="Git operation: 'push' to obfuscate before pushing, 'pull' to deobfuscate after pulling"
    ),
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to the repository root or specific file/directory"
    ),
    config: Optional[str] = typer.Option(
        None, 
        "--config", 
        "-c", 
        help="Path to custom config file"
    ),
    mapping: Optional[str] = typer.Option(
        None, 
        "--mapping", 
        "-m", 
        help="Path to custom mapping file"
    )
):
    """Prepare code for Git operations (manually trigger obfuscation/deobfuscation)."""
    config_path = Path(config) if config else None
    mapping_path = Path(mapping) if mapping else None
    target_path = Path(path)
    
    try:
        obfuscator = Obfuscator(config_path, mapping_path)
        
        if operation.lower() == "push":
            # Obfuscate before pushing
            file_count = obfuscator.jam(target_path)
            if file_count > 0:
                console.print(f"[green]Successfully obfuscated {file_count} file(s) at {target_path}[/green]")
                console.print("[green]Code is ready to be pushed![/green]")
            else:
                console.print(f"[yellow]No files were obfuscated at {target_path}[/yellow]")
        
        elif operation.lower() == "pull":
            # Deobfuscate after pulling
            file_count = obfuscator.unjam(target_path)
            if file_count > 0:
                console.print(f"[green]Successfully deobfuscated {file_count} file(s) at {target_path}[/green]")
                console.print("[green]Code is ready for local development![/green]")
            else:
                console.print(f"[yellow]No files were deobfuscated at {target_path}[/yellow]")
        
        else:
            console.print(f"[red]Invalid operation: {operation}[/red]")
            console.print("[yellow]Use 'push' to obfuscate before pushing or 'pull' to deobfuscate after pulling[/yellow]")
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1)
