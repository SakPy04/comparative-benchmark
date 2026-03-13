
import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


METRICS = ["psnr", "ssim", "lpips", "gmsd"]


def _clear_repo_modules() -> None:
    """Remove cached repo modules from sys.modules to prevent import collisions."""
    for module_name in list(sys.modules.keys()):
        if module_name.startswith(('data', 'models', 'options')):
            if not module_name.startswith(('dataclasses', 'dataloader')):
                del sys.modules[module_name]


def _push_repo_path(repo_path: Path) -> None:
    """Add repo to sys.path if not already present."""
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _pop_repo_path(repo_path: Path) -> None:
    """Remove repo from sys.path."""
    repo_str = str(repo_path)
    if repo_str in sys.path:
        sys.path.remove(repo_str)


def aggregate_metrics(all_metrics: Dict[str, List[float]]) -> Dict[str, float]:

    aggregated = {}
    for key, values in all_metrics.items():
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
    return aggregated


def print_aggregated_metrics(aggregated: Dict[str, float]) -> None:
    print("\nAggregated Metrics:")
    print("-" * 30)
    for metric in METRICS:
        mean_key = f"{metric}_mean"
        if mean_key in aggregated:
            print(f"  {metric.upper():6s}: {aggregated[mean_key]:.4f} ± {aggregated[f'{metric}_std']:.4f}")


def save_results(
    model_name: str,
    data_dir: str,
    input_key: str,
    target_key: str,
    n_samples: int,
    aggregated: Dict[str, float],
    save_dir: str
) -> None:

    print("\n" + "-" * 50)
    print("Saving Results")
    print("-" * 50)
    
    results = {
        "model": model_name,
        "config": {
            "data_dir": str(data_dir),
            "input_key": input_key,
            "target_key": target_key,
            "n_samples": n_samples,
        },
        "metrics": aggregated
    }
    
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
