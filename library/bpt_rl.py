"""Importable interface for the BPT-RL library script."""

from pathlib import Path
from runpy import run_path


_implementation = run_path(str(Path(__file__).with_name("bpt-rl.py")))

PackOptions = _implementation["PackOptions"]
complete_analysis = _implementation["complete_analysis"]

__all__ = ["PackOptions", "complete_analysis"]
