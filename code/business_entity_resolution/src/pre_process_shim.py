"""Import shim: pre-process.py has a dash in the filename."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "pre-process.py"
_spec = importlib.util.spec_from_file_location("pre_process_mod", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update(_mod.__dict__)
