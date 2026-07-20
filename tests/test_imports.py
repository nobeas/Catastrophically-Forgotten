import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_core_modules_import():
    modules = [
        "src.data",
        "src.models.base",
        "src.experiments.forgetting",
        "src.analysis.accuracy",
        "src.analysis.weights",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None
