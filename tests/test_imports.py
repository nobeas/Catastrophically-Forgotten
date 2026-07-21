import importlib
import importlib.util
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


def test_backpropagation_module_exposes_minimal_mlp_interface():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "backpropagation.py"

    assert module_path.exists()

    spec = importlib.util.spec_from_file_location("backpropagation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "MultiLayerPerceptron")
    assert hasattr(module, "BasicOptimizer")
    assert hasattr(module, "train_model")
