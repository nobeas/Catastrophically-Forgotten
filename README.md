# Catastrophically Forgotten

This repository contains a modular PyTorch project for studying catastrophic forgetting in a NeuroAI setting using MNIST.

## Project goals

- Compare backpropagation against alternative learning rules in a forgetting experiment.
- Keep the code organized by concern so the project is easy for a group to extend.
- Support both sequential and interleaved training conditions.

## Repository structure

- `src/data.py` – dataset download and class-restriction helpers.
- `src/models/` – model implementations and training utilities.
- `src/experiments/` – forgetting experiment harnesses.
- `src/analysis/` – plotting and metric helpers.
- `tests/` – smoke tests for the package structure.

## Getting started

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the smoke tests:
   ```bash
   python -m pytest -q
   ```

## Suggested workflow for collaborators

- Keep experiments in `src/experiments/`.
- Keep model code in `src/models/`.
- Keep visualization helpers in `src/analysis/`.
- Add tests for new functionality under `tests/`.

## Notes

The project is intentionally modular so new learning rules can be added without rewriting the experiment harness.
