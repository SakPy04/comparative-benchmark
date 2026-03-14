
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


def run_model_evaluation(
    data_dir: str,
    model_name: str = "pix2pix",
    input_key: str = "1_10",
    target_key: str = "full",
    batch_size: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    max_batches: Optional[int] = None
) -> Dict[str, float]:

    repo_path = Path(__file__).parent.parent / "pytorch-CycleGAN-and-pix2pix"
    saved_cwd = os.getcwd()
    saved_argv = sys.argv.copy()
    
    # Convert to absolute path since we'll change directory
    abs_data_dir = str(Path(data_dir).resolve())
    
    try:
        os.chdir(repo_path)
        _clear_repo_modules()
        _push_repo_path(repo_path)
        
        from options.train_options import TrainOptions
        from data import create_dataset
        from models import create_model
        from eval.metrics import MetricsEvaluator
        
        print("=" * 70)
        print(f"PET EVALUATION: {model_name.upper()}")
        print("=" * 70)
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        print(f"\nUsing device: {device}")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Results will be saved to: {save_dir}")
        
        sys.argv = [
            "eval",
            "--dataroot", abs_data_dir,
            "--phase", "train",
            "--dataset_mode", "h5_aligned",
            "--model", model_name,
            "--input_key", input_key,
            "--target_key", target_key,
            "--input_nc", "1",
            "--output_nc", "1",
            "--batch_size", str(batch_size),
            "--preprocess", "none",
            "--num_threads", str(num_workers),
            "--netG", "resnet_6blocks"
        ]
        
        if checkpoint_dir:
            sys.argv.extend(["--checkpoints_dir", checkpoint_dir])
        
        opt = TrainOptions().parse()
        opt.device = device

        dataset = create_dataset(opt)
        print(f"Dataset created with {len(dataset)} samples")
        
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        print(f"Model {model_name} created and set to eval mode")
        
        # Check if checkpoint was actually loaded
        checkpoint_status = "Model initialized from repo (weights status unknown)"
        if hasattr(model, 'netG'):
            if hasattr(model.netG, 'load_state_dict'):
                checkpoint_status = "Model architecture ready (checkpoint loading depends on repo defaults)"
        elif hasattr(model, 'netG_A'):
            if hasattr(model.netG_A, 'load_state_dict'):
                checkpoint_status = "Model architecture ready (checkpoint loading depends on repo defaults)"
        
        print(f"Checkpoint status: {checkpoint_status}")
        
        evaluator = MetricsEvaluator(device=device)
        
        all_metrics: Dict[str, List[float]] = {
            "nmse": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "gmsd": [],
            "vifp": []
        }
        
        n_batches = len(dataset)
        if max_batches:
            n_batches = min(n_batches, max_batches)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(dataset):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                try:
                    A = data["A"].to(device)
                    B = data["B"].to(device)
                    
                    if batch_idx == 0:
                        print(f"\nFirst sample shape - A: {A.shape}, B: {B.shape}")
                    
                    data["A"] = A
                    data["B"] = B
                    
                    model.set_input(data)
                    model.forward()
                    
                    pred = model.fake_B
                    target = model.real_B
                    
                    batch_metrics = evaluator.evaluate_batch(pred, target)
                    
                    for key in all_metrics:
                        if key in batch_metrics and not np.isnan(batch_metrics[key]):
                            all_metrics[key].append(batch_metrics[key])
                    
                    if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
                        print(f"  Processed batch {batch_idx + 1}/{n_batches}")
                        if "psnr" in batch_metrics:
                            print(f"    NMSE: {batch_metrics['nmse']:.4f} | "
                                  f"PSNR: {batch_metrics['psnr']:.2f} dB | "
                                  f"SSIM: {batch_metrics['ssim']:.4f} | "
                                  f"LPIPS: {batch_metrics['lpips']:.4f} | "
                                  f"GMSD: {batch_metrics['gmsd']:.4f} | "
                                  f"VIFp: {batch_metrics['vifp']:.4f}")
                except Exception as e:
                    print(f"  Warning: Skipping batch {batch_idx} due to error: {str(e)[:100]}")
                    continue
        
        print("\n" + "-" * 50)
        print("Aggregating Results")
        print("-" * 50)
        
        aggregated = aggregate_metrics(all_metrics)
        print_aggregated_metrics(aggregated)
        
        if save_dir:
            save_results(model_name, data_dir, input_key, target_key, n_batches, aggregated, save_dir)
        
        print("\n" + "=" * 70)
        print(f"EVALUATION COMPLETE ({model_name})")
        print("=" * 70)
        
        return aggregated
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        _pop_repo_path(repo_path)
