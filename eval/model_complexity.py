"""

Usage:
    from model_complexity import compute_gflops, format_gflops_summary, compare_models
    
    # Default: 1 MAC = 2 FLOPs (theoretical FLOPs)
    stats = compute_gflops(model, input_tensor, model_name="MyModel")
    print(format_gflops_summary(stats))
    
    # Paper-style: some papers report MACs as "GFLOPs" (factor=1.0)
    stats = compute_gflops(model, input_tensor, mac_to_flop_factor=1.0)

"""

import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def compute_gflops(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_name: Optional[str] = None,
    mac_to_flop_factor: float = 2.0,
) -> Dict[str, Union[str, float, None, Tuple]]:
    """
    Compute GFLOPs for a PyTorch model.
    """
    # ---- Input validation ----
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module for model, got {type(model).__name__}")
    
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for input_tensor, got {type(input_tensor).__name__}")
    
    if input_tensor.ndim < 2:
        raise ValueError(
            f"input_tensor must have at least 2 dimensions (batch, ...), "
            f"got shape {tuple(input_tensor.shape)}"
        )
    
    # ---- Preserve model state ----
    original_training_state = model.training
    original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    try:
        # Move input to model's device
        input_tensor = input_tensor.to(original_device)
  
        model.eval()
        

        macs, profiler = _compute_macs_thop(model, input_tensor)
                
        # Compute GFLOPs from MACs
        if macs is not None:
            gflops = (macs * mac_to_flop_factor) / 1e9
        else:
            gflops = None
        
        return {
            'model_name': model_name if model_name else 'Unknown',
            'macs': macs,
            'gflops': float(gflops) if gflops is not None else None,
            'input_shape': tuple(input_tensor.shape),
            'mac_to_flop_factor': mac_to_flop_factor,
            'profiler': profiler,
        }
    
    finally:
        if original_training_state:
            model.train()
        else:
            model.eval()


def _compute_macs_thop(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute MACs using thop library.
    
    Returns:
        Tuple of (macs, profiler_name) where profiler_name is 'thop' or None
    """
    try:
        from thop import profile
    except ImportError:
        warnings.warn(
            "thop is not installed. Install it with: pip install thop\n"
            "Falling back to fvcore if available.",
            ImportWarning
        )
        return None, None
    
    try:
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return float(macs), 'thop'
    
    except Exception as e:
        warnings.warn(
            f"Failed to compute MACs with thop: {e}\n"
            "This may happen with custom layers or dynamic architectures.",
            RuntimeWarning
        )
        return None, None


def format_gflops_summary(stats: Dict[str, Union[str, float, None]]) -> str:
    """
    Format GFLOPs statistics into a human-readable string.
    
    Args:
        stats: Dictionary returned by compute_gflops()
    
    Returns:
        Formatted string with model complexity summary
    """
    model_name = stats.get('model_name', 'Unknown')
    input_shape = stats.get('input_shape', 'N/A')
    macs = stats.get('macs')
    gflops = stats.get('gflops')
    factor = stats.get('mac_to_flop_factor', 2.0)
    profiler = stats.get('profiler', 'N/A')
    
    macs_str = f"{macs / 1e9:.2f}G" if macs is not None else "N/A"
    gflops_str = f"{gflops:.2f}" if gflops is not None else "N/A"
    
    lines = [
        "=" * 50,
        f"Model: {model_name}",
        f"Input Shape: {input_shape}",
        "-" * 50,
        f"MACs: {macs_str}",
        f"GFLOPs: {gflops_str}",
        f"MAC-to-FLOP Factor: {factor}",
        f"Profiler: {profiler}",
        "=" * 50,
    ]
    
    return "\n".join(lines)


def compare_models(
    models: Dict[str, nn.Module],
    input_tensor: torch.Tensor,
    mac_to_flop_factor: float = 2.0,
) -> str:
    """
    Compare GFLOPs across multiple models.
    
    Args:
        models: Dictionary mapping model names to model instances
        input_tensor: Sample input tensor for profiling (same for all models)
        mac_to_flop_factor: Conversion factor from MACs to FLOPs (default: 2.0)
    
    Returns:
        Formatted comparison table as a string
    """
    if not models:
        return "No models provided for comparison."
    
    all_stats = []
    for name, model in models.items():
        stats = compute_gflops(
            model, input_tensor, 
            model_name=name, 
            mac_to_flop_factor=mac_to_flop_factor
        )
        all_stats.append(stats)

    width = 55
    separator = "=" * width
    
    lines = [
        separator,
        "MODEL COMPLEXITY COMPARISON",
        separator,
        f"Input Shape: {tuple(input_tensor.shape)}",
        f"MAC-to-FLOP Factor: {mac_to_flop_factor}",
        "-" * width,
        f"{'Model':<25}{'MACs (G)':<15}{'GFLOPs':<15}",
        "-" * width,
    ]
    
    for stats in all_stats:
        name = stats['model_name']
        macs = stats.get('macs')
        gflops = stats.get('gflops')
        macs_str = f"{macs / 1e9:.2f}" if macs is not None else "N/A"
        gflops_str = f"{gflops:.2f}" if gflops is not None else "N/A"
        lines.append(f"{name:<25}{macs_str:<15}{gflops_str:<15}")
    
    lines.append(separator)
    
    return "\n".join(lines)
