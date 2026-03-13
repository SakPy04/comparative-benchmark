import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

from evaluators.common import (
    aggregate_metrics,
    print_aggregated_metrics,
    save_results,
    METRICS,
    _clear_repo_modules,
    _push_repo_path,
    _pop_repo_path,
)


def compute_baseline_metrics(
    dataset,
    evaluator,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    
    baseline_metrics: Dict[str, List[float]] = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "gmsd": []
    }
    
    n_batches = len(dataset)
    if max_batches:
        n_batches = min(n_batches, max_batches)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            input_img = data["A"].to(device)
            target_img = data["B"].to(device)
            
            batch_metrics = evaluator.evaluate_batch(input_img, target_img)
            
            for key in baseline_metrics:
                if key in batch_metrics and not np.isnan(batch_metrics[key]):
                    baseline_metrics[key].append(batch_metrics[key])
            
            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
                print(f"  Processed batch {batch_idx + 1}/{n_batches}")
                if "psnr" in batch_metrics:
                    print(f"    PSNR: {batch_metrics['psnr']:.2f} dB | "
                          f"SSIM: {batch_metrics['ssim']:.4f} | "
                          f"LPIPS: {batch_metrics['lpips']:.4f} | "
                          f"GMSD: {batch_metrics['gmsd']:.4f}")
    
    baseline_aggregated = aggregate_metrics(baseline_metrics)
    
    print("\nBASELINE METRICS (lq vs hq):")
    print("-" * 30)
    for metric in METRICS:
        mean_key = f"{metric}_mean"
        if mean_key in baseline_aggregated:
            print(f"  {metric.upper():6s}: {baseline_aggregated[mean_key]:.4f} ± {baseline_aggregated[f'{metric}_std']:.4f}")
    
    return baseline_aggregated


def run_baseline_evaluation(
    data_dir: str,
    input_key: str = "1_10",
    target_key: str = "full",
    batch_size: int = 1,
    max_batches: Optional[int] = None,
    device: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:

    repo_path = Path(__file__).parent.parent / "pytorch-CycleGAN-and-pix2pix"
    saved_cwd = os.getcwd()
    saved_argv = sys.argv.copy()
    
    try:
        os.chdir(repo_path)
        _clear_repo_modules()
        _push_repo_path(repo_path)
        
        from options.train_options import TrainOptions
        from data import create_dataset
        from eval.metrics import MetricsEvaluator
        
        sys.argv = [
            "eval",
            "--dataroot", str(data_dir),
            "--phase", "train",
            "--dataset_mode", "h5_aligned",
            "--model", "pix2pix",
            "--input_key", input_key,
            "--target_key", target_key,
            "--input_nc", "1",
            "--output_nc", "1",
            "--batch_size", str(batch_size),
            "--preprocess", "none",
            "--num_threads", "0",
            "--netG", "resnet_6blocks"
        ]

        opt = TrainOptions().parse()
        opt.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        dataset = create_dataset(opt)
        evaluator = MetricsEvaluator(device=opt.device)
        
        n_batches = len(dataset)
        if max_batches:
            n_batches = min(n_batches, max_batches)
        baseline_aggregated = compute_baseline_metrics(dataset, evaluator, opt.device, max_batches=max_batches)
        
        if save_dir:
            save_results(
                "baseline",
                data_dir,
                input_key,
                target_key,
                n_batches,
                baseline_aggregated,
                save_dir,
            )
        
        return baseline_aggregated
    
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        _pop_repo_path(repo_path)
