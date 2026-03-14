import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluators.common import (
    aggregate_metrics,
    print_aggregated_metrics,
    save_results,
    _clear_repo_modules,
    _push_repo_path,
    _pop_repo_path,
)
from eval.metrics import MetricsEvaluator


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _detect_phase(data_dir: str) -> str:
    root = Path(data_dir)
    if (root / "test").is_dir():
        return "test"
    if (root / "val").is_dir():
        return "val"
    if (root / "train").is_dir():
        return "train"
    return "train"


def _validate_required_npy_files(models_dir: Path) -> None:
    """
    I2I-Mamba uses hardcoded absolute np.load paths inside modules.py.
    We do NOT edit repo files; we only make sure the needed local files exist.
    
    Verified against I2I-Mamba/models/path_generate.py:
      - Generates: spiral_eye.npy, despiral_eye.npy, despiral_r_eye.npy
      - Model loads: spiral_eye.npy, despiral_eye.npy, despiral_r_eye.npy
    """
    required = [
        "spiral_eye.npy",
        "despiral_eye.npy",
        "despiral_r_eye.npy",
    ]
    
    found = []
    missing = []
    for name in required:
        if (models_dir / name).is_file():
            found.append(name)
        else:
            missing.append(name)
    
    print(f"\n[I2I-Mamba] Checking required .npy files in {models_dir}")
    print(f"  Found: {len(found)}/{len(required)} files")
    if found:
        for f in found:
            print(f"    ✓ {f}")
    
    if missing:
        missing_str = ", ".join(missing)
        print(f"  Missing: {missing_str}")
        raise FileNotFoundError(
            f"\nMissing required Step 0 files in {models_dir}:\n"
            f"  {missing_str}\n\n"
            f"To generate these files, run from inside I2I-Mamba/:\n"
            f"  cd {models_dir.parent}\n"
            f"  python3 models/path_generate.py\n\n"
            f"This generates the spiral ordering matrices needed by the model."
        )
    print("  All required files found!\n")


def _patch_numpy_load(models_dir: Path):
    """
    Redirect repo hardcoded absolute paths like:
      /auto/.../models/spiral_eye.npy
    to:
      <repo>/models/spiral_eye.npy
    """
    original_np_load = np.load

    def patched_np_load(file, *args, **kwargs):
        file_str = str(file)
        basename = os.path.basename(file_str)

        redirect_names = {
            "spiral_eye.npy",
            "despiral_eye.npy",
            "despiral_r_eye.npy",
        }

        if basename in redirect_names:
            local_path = models_dir / basename
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"Required file missing: {basename}\n"
                    f"Expected at: {local_path}"
                )
            return original_np_load(str(local_path), *args, **kwargs)

        return original_np_load(file, *args, **kwargs)

    np.load = patched_np_load
    return original_np_load


def _build_i2i_opt(
    data_dir: str,
    input_key: str,
    target_key: str,
    batch_size: int,
    num_workers: int,
    resolved_device: torch.device,
    checkpoint_dir: Optional[str],
):
    """
    Build I2I-Mamba options without modifying the repo.
    """
    from options.test_options import TestOptions

    # Convert to absolute path since we'll change directory
    abs_data_dir = str(Path(data_dir).resolve())
    
    # Don't append phase subdirectory - use dataroot directly
    sys.argv = [
        "eval",
        "--dataroot", abs_data_dir,
        "--phase", "train",  # Empty phase means no subdirectory appended
        "--model", "i2i_mamba_one",
        "--which_model_netG", "i2i_mamba",
        "--dataset_mode", "h5_aligned",
        "--which_direction", "AtoB",
        "--batchSize", str(batch_size),
        "--nThreads", str(num_workers),
        "--serial_batches",
        "--no_flip",
        "--input_nc", "1",
        "--output_nc", "1",
        "--fineSize", "256",
        "--loadSize", "256",
        "--crop_size_h", "256",
        "--crop_size_w", "256",
    ]

    if resolved_device.type == "cpu":
        sys.argv.extend(["--gpu_ids", "-1"])

    if checkpoint_dir:
        sys.argv.extend(["--checkpoints_dir", checkpoint_dir])

    opt = TestOptions().parse()
    opt.device = resolved_device

    # custom args for your H5 dataset fallback
    opt.input_key = input_key
    opt.target_key = target_key

    return opt


def _create_i2i_dataset(opt):
    """
    Try repo-native loader first.
    If h5_aligned is not registered in I2I-Mamba's data/__init__.py,
    fall back to your H5AlignedDataset directly.
    """
    from data import CreateDataLoader

    try:
        data_loader = CreateDataLoader(opt)
        dataset_iterable = data_loader.load_data()
        sample_count = len(data_loader)
        batch_count = len(data_loader.dataloader) if hasattr(data_loader, "dataloader") else sample_count
        print("Using I2I-Mamba CreateDataLoader")
        return dataset_iterable, sample_count, batch_count
    except Exception as e:
        if opt.dataset_mode != "h5_aligned":
            raise

        from data.h5_aligned_dataset import H5AlignedDataset

        ds = H5AlignedDataset()
        ds.initialize(opt)

        loader = DataLoader(
            ds,
            batch_size=opt.batchSize,
            shuffle=not getattr(opt, "serial_batches", False),
            num_workers=int(getattr(opt, "nThreads", 0)),
        )

        print(f"CreateDataLoader fallback activated for h5_aligned: {e}")
        return loader, len(ds), len(loader)


def _get_checkpoint_status(opt) -> str:
    ckpt_root = Path(opt.checkpoints_dir) / opt.name

    # most likely generator file names
    candidates = [
        ckpt_root / f"{opt.which_epoch}_net_G.pth",
        ckpt_root / "latest_net_G.pth",
    ]

    for path in candidates:
        if path.is_file():
            return f"Found checkpoint: {path}"

    return (
        "No generator checkpoint found under "
        f"{ckpt_root}. Model may be random-initialized."
    )


def run_i2i_mamba_evaluation(
    data_dir: str,
    input_key: str = "1_10",
    target_key: str = "full",
    batch_size: int = 1,
    num_workers: int = 0,
    device: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate I2I-Mamba on PET HDF5 data using your H5 dataset fallback when needed.

    Assumptions:
      - Step 0 has already been run
      - required .npy files exist in I2I-Mamba/models/
      - dependencies like ml_collections and mamba_ssm are installed
    """
    repo_path = Path(__file__).parent.parent / "I2I-Mamba"
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    
    models_dir = repo_path / "models"

    saved_cwd = os.getcwd()
    saved_argv = sys.argv.copy()
    original_np_load = np.load
    original_load_network = None

    try:
        os.chdir(repo_path)
        _clear_repo_modules()
        _push_repo_path(repo_path)

        _validate_required_npy_files(models_dir)
        original_np_load = _patch_numpy_load(models_dir)

        from models import create_model
        from models.base_model import BaseModel

        resolved_device = _resolve_device(device)

        print("=" * 70)
        print("PET EVALUATION: I2I-MAMBA")
        print("=" * 70)
        print(f"\nUsing device: {resolved_device}")
        print(f"Using phase: {_detect_phase(data_dir)}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Results will be saved to: {save_dir}")

        opt = _build_i2i_opt(
            data_dir=data_dir,
            input_key=input_key,
            target_key=target_key,
            batch_size=batch_size,
            num_workers=num_workers,
            resolved_device=resolved_device,
            checkpoint_dir=checkpoint_dir,
        )

        dataset_iterable, sample_count, batch_count = _create_i2i_dataset(opt)
        print(f"Dataset ready: {sample_count} samples, {batch_count} batches")

        # Check if checkpoint exists before creating model
        ckpt_root = Path(opt.checkpoints_dir) / opt.name
        ckpt_path = ckpt_root / f"{opt.which_epoch}_net_G.pth"
        checkpoint_exists = ckpt_path.is_file()

        if not checkpoint_exists:
            print(f"\n[WARNING] Checkpoint not found at: {ckpt_path}")
            print("          Running I2I-Mamba with random initialized weights.\n")

            # Monkey-patch load_network to skip loading missing checkpoints
            original_load_network = BaseModel.load_network

            def safe_load_network(self, network, network_label, epoch_label):
                save_path = os.path.join(self.save_dir, f"{epoch_label}_net_{network_label}.pth")
                if not os.path.exists(save_path):
                    return  # Skip loading, use random weights
                return original_load_network(self, network, network_label, epoch_label)

            BaseModel.load_network = safe_load_network

        model = create_model(opt)
        print("Model I2I-Mamba created")
        print(f"Checkpoint status: {_get_checkpoint_status(opt)}")

        evaluator = MetricsEvaluator(device=resolved_device)

        all_metrics: Dict[str, List[float]] = {
            "nmse": [],
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "gmsd": [],
            "vifp": [],
        }

        n_batches = batch_count
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)

        with torch.no_grad():
            for batch_idx, data in enumerate(dataset_iterable):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                try:
                    if batch_idx == 0:
                        print(f"\nFirst sample shape - A: {data['A'].shape}, B: {data['B'].shape}")

                    model.set_input(data)
                    model.test()

                    if not hasattr(model, "fake_B") or not hasattr(model, "real_B"):
                        raise AttributeError(
                            "I2I-Mamba model does not expose expected attributes "
                            "'fake_B' and 'real_B'."
                        )

                    pred = model.fake_B
                    target = model.real_B

                    batch_metrics = evaluator.evaluate_batch(pred, target)

                    for key in all_metrics:
                        if key in batch_metrics and not np.isnan(batch_metrics[key]):
                            all_metrics[key].append(batch_metrics[key])

                    if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
                        print(f"  Processed batch {batch_idx + 1}/{n_batches}")
                        if "psnr" in batch_metrics:
                            print(
                                f"    NMSE: {batch_metrics['nmse']:.4f} | "
                                f"PSNR: {batch_metrics['psnr']:.2f} dB | "
                                f"SSIM: {batch_metrics['ssim']:.4f} | "
                                f"LPIPS: {batch_metrics['lpips']:.4f} | "
                                f"GMSD: {batch_metrics['gmsd']:.4f} | "
                                f"VIFp: {batch_metrics['vifp']:.4f}"
                            )
                except Exception as e:
                    print(f"  Warning: Skipping batch {batch_idx} due to error: {str(e)[:100]}")
                    continue

        aggregated = aggregate_metrics(all_metrics)
        print_aggregated_metrics(aggregated)

        if save_dir:
            save_results(
                "i2i_mamba",
                data_dir,
                input_key,
                target_key,
                n_batches,
                aggregated,
                save_dir,
            )

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE (I2I-MAMBA)")
        print("=" * 70)

        return aggregated

    finally:
        # Restore original load_network if we patched it
        if original_load_network is not None:
            from models.base_model import BaseModel
            BaseModel.load_network = original_load_network

        np.load = original_np_load
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        _pop_repo_path(repo_path)
        _clear_repo_modules()