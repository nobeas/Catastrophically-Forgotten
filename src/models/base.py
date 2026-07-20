"""Baseline backprop implementation used by the forgetting experiment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiLayerPerceptron(nn.Module):
    """A simple two-layer MLP for MNIST-style classification."""

    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, bias: bool = True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.lin1 = nn.Linear(num_inputs, num_hidden, bias=bias)
        self.lin2 = nn.Linear(num_hidden, num_outputs, bias=bias)
        self._initialize_parameters()
        self.init_lin1_weight = self.lin1.weight.detach().clone()
        self.init_lin1_bias = self.lin1.bias.detach().clone()
        self.init_lin2_weight = self.lin2.weight.detach().clone()
        self.init_lin2_bias = self.lin2.bias.detach().clone()

    def _initialize_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

    def forward_backprop(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def list_parameters(self) -> list[str]:
        return ["lin1_weight", "lin1_bias", "lin2_weight", "lin2_bias"]

    def gather_gradient_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "lin1_weight": self.lin1.weight.grad.detach().clone() if self.lin1.weight.grad is not None else None,
            "lin1_bias": self.lin1.bias.grad.detach().clone() if self.lin1.bias.grad is not None else None,
            "lin2_weight": self.lin2.weight.grad.detach().clone() if self.lin2.weight.grad is not None else None,
            "lin2_bias": self.lin2.bias.grad.detach().clone() if self.lin2.bias.grad is not None else None,
        }


class BasicOptimizer:
    """Thin wrapper around torch.optim.SGD for the project code."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter] | Iterable[dict], lr: float):
        self.optimizer = torch.optim.SGD(parameters, lr=lr)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


def update_results_by_class_in_place(results: Dict[str, Any], correct_by_class: Counter, seen_by_class: Counter, prefix: str) -> None:
    results[f"{prefix}_correct_by_class"] = correct_by_class
    results[f"{prefix}_seen_by_class"] = seen_by_class


def train_epoch(model: nn.Module, loader, optimizer: BasicOptimizer, criterion=None, device: Optional[str] = None) -> Dict[str, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    correct_by_class: Counter = Counter()
    seen_by_class: Counter = Counter()

    for inputs, targets in loader:
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)

        outputs = model(inputs.view(inputs.size(0), -1))
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_seen += len(targets)

        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == targets).sum().item()

        for pred, target in zip(preds.tolist(), targets.tolist()):
            seen_by_class[target] += 1
            if pred == target:
                correct_by_class[target] += 1

    return {
        "loss": total_loss / total_seen,
        "accuracy": 100.0 * total_correct / total_seen,
        "correct_by_class": correct_by_class,
        "seen_by_class": seen_by_class,
    }


def train_model(model, train_loader, valid_loader, optimizer, num_epochs: int = 5, criterion=None, verbose: bool = False):
    results: Dict[str, Any] = {
        "avg_train_losses": [],
        "avg_train_accuracies": [],
        "avg_valid_losses": [],
        "avg_valid_accuracies": [],
    }

    for epoch in range(num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, criterion=criterion)
        valid_stats = evaluate_accuracy(model, valid_loader, criterion=criterion)

        results["avg_train_losses"].append(train_stats["loss"])
        results["avg_train_accuracies"].append(train_stats["accuracy"])
        results["avg_valid_losses"].append(valid_stats["loss"])
        results["avg_valid_accuracies"].append(valid_stats["accuracy"])

        update_results_by_class_in_place(results, train_stats["correct_by_class"], train_stats["seen_by_class"], "train")
        update_results_by_class_in_place(results, valid_stats["correct_by_class"], valid_stats["seen_by_class"], "valid")

        if verbose:
            print(f"epoch {epoch + 1}: train acc = {train_stats['accuracy']:.2f}%, valid acc = {valid_stats['accuracy']:.2f}%")

    return results


def evaluate_accuracy(model: nn.Module, loader, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    correct_by_class: Counter = Counter()
    seen_by_class: Counter = Counter()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            total_seen += len(targets)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == targets).sum().item()

            for pred, target in zip(preds.tolist(), targets.tolist()):
                seen_by_class[target] += 1
                if pred == target:
                    correct_by_class[target] += 1

    return {
        "loss": total_loss / total_seen,
        "accuracy": 100.0 * total_correct / total_seen,
        "correct_by_class": correct_by_class,
        "seen_by_class": seen_by_class,
    }
