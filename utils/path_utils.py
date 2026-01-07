"""
Path utilities for finding the repository root, even from git worktrees.

This module provides a robust function to find the real git repository root,
which works correctly even when running from Cursor/VS Code worktrees.
"""
from pathlib import Path


def find_repo_root(start_path: Path = None) -> Path:
    """
    Find the real git repository root, even when running from a worktree.
    
    This function walks up from the start path until it finds:
    1. A .git directory (main repo)
    2. A .git file (worktree) - then resolves to the main repo
    
    Args:
        start_path: Starting path to search from. If None, uses Path(__file__).parent.
        
    Returns:
        Path to the real repository root
        
    Examples:
        >>> from utils.path_utils import find_repo_root
        >>> from pathlib import Path
        >>> root = find_repo_root(Path(__file__).parent)
        >>> # root will be /home/essashah/smo_model_new regardless of worktree
    """
    if start_path is None:
        # Default to this file's parent directory
        start_path = Path(__file__).resolve().parent
    else:
        start_path = Path(start_path).resolve()
    
    current = start_path
    
    while current != current.parent:  # Stop at filesystem root
        git_path = current / ".git"
        
        # Check if .git exists as a directory (main repo)
        if git_path.is_dir():
            # Check if it's actually a worktree by looking for worktrees subdirectory
            worktrees_dir = git_path / "worktrees"
            if worktrees_dir.exists():
                # This is the main repo, return it
                return current
            else:
                # Regular .git directory, this is the repo root
                return current
        
        # Check if .git exists as a file (worktree)
        if git_path.is_file():
            # Read the .git file to find the main repo
            try:
                with open(git_path, 'r') as f:
                    content = f.read().strip()
                    # Format: "gitdir: /path/to/main/.git/worktrees/worktree-name"
                    if content.startswith("gitdir: "):
                        gitdir_path = Path(content[8:].strip())
                        # Navigate to the main repo (parent of .git/worktrees)
                        main_git_dir = gitdir_path.parent.parent
                        # The repo root is the parent of .git
                        return main_git_dir
            except Exception:
                pass
        
        # Also check for Models directory as a fallback indicator
        if (current / "Models").exists() and (current / "Models" / "Trained").exists():
            return current
        
        current = current.parent
    
    # Fallback: return the original start path's parent
    return start_path.parent if start_path != start_path.parent else start_path
