import importlib


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
