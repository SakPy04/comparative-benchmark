"""
Metrics included:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index) using pytorch-msssim
- LPIPS (Learned Perceptual Image Patch Similarity) using lpips
- GMSD (Gradient Magnitude Similarity Deviation) using piq
"""

import torch
from typing import Dict, Optional, Union
import warnings


# ============================================================================
# PSNR (Peak Signal-to-Noise Ratio)
# ============================================================================

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    eps: float = 1e-8
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Computes PSNR per-image then averages across the batch.
    """
    _validate_tensor_shapes(pred, target, "PSNR")

    # Per-image MSE, then per-image PSNR, then average
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # Shape: (B,)
    
    # Handle case where MSE is very small (near-identical images)
    psnr = torch.where(
        mse < eps,
        torch.tensor(float("inf"), device=mse.device),
        10 * torch.log10((data_range ** 2) / (mse + eps))
    )

    return float(psnr.mean().item())

# ============================================================================
# SSIM (Structural Similarity Index)
# ============================================================================

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute Structural Similarity Index (SSIM) using pytorch-msssim.
    """
    _validate_tensor_shapes(pred, target, "SSIM")

    try:
        from pytorch_msssim import ssim
    except ImportError as e:
        raise ImportError(
            "pytorch-msssim is required for SSIM computation. "
            "Install it with: pip install pytorch-msssim"
        ) from e

    pred, target = _ensure_compatible(pred, target)
    
    with torch.no_grad():
        ssim_val = ssim(pred, target, data_range=data_range, size_average=True)
    
    return float(ssim_val.item())


# ============================================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================================
class LPIPSMetric:
    """
    LPIPS metric wrapper for grayscale images.
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[torch.device] = None
    ) -> None:
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "lpips is required for LPIPS computation. "
                "Install it with: pip install lpips"
            ) from e

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.loss_fn.eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        _validate_tensor_shapes(pred, target, "LPIPS")
        pred, target = _ensure_compatible(pred, target)

        pred_3ch = _grayscale_to_3channel(pred).to(self.device)
        target_3ch = _grayscale_to_3channel(target).to(self.device)

        pred_min, pred_max = pred_3ch.min().item(), pred_3ch.max().item()
        target_min, target_max = target_3ch.min().item(), target_3ch.max().item()

        # Convert only if tensors are in [0, 1]
        if (
            pred_min >= 0.0 and pred_max <= 1.0 and
            target_min >= 0.0 and target_max <= 1.0
        ):
            pred_3ch = pred_3ch * 2.0 - 1.0
            target_3ch = target_3ch * 2.0 - 1.0

        with torch.no_grad():
            lpips_val = self.loss_fn(pred_3ch, target_3ch)

        return float(lpips_val.mean().item())

def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'alex',
    device: Optional[torch.device] = None
) -> float:
    """
    Convenience function to compute LPIPS.
    """
    metric = LPIPSMetric(net=net, device=device)
    return metric(pred, target)


# ============================================================================
# GMSD (Gradient Magnitude Similarity Deviation)
# ============================================================================

def compute_gmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD) using piq.
    
    Lower is better. Typical range: [0, 0.3] for good quality.
    
    Args:
        pred: Predicted image tensor of shape (B, 1, H, W)
        target: Target image tensor of shape (B, 1, H, W)
        reduction: Reduction method ('mean' or 'sum')
    
    Returns:
        GMSD value (float). Lower is better.
    
    Raises:
        ValueError: If tensor shapes don't match or are invalid
        ImportError: If piq is not installed
    """
    _validate_tensor_shapes(pred, target, "GMSD")
    
    try:
        from piq import gmsd
    except ImportError as e:
        raise ImportError(
            "piq is required for GMSD computation. "
            "Install it with: pip install piq"
        ) from e
    
    pred, target = _ensure_compatible(pred, target)

    pred_proc = pred
    target_proc = target
    
    pred_min, pred_max = pred_proc.min().item(), pred_proc.max().item()
    target_min, target_max = target_proc.min().item(), target_proc.max().item()
    
    if pred_min >= -1.0 and pred_max <= 1.0 and target_min >= -1.0 and target_max <= 1.0:
        if pred_min < 0.0 or target_min < 0.0:
            pred_proc = (pred_proc + 1.0) / 2.0
            target_proc = (target_proc + 1.0) / 2.0
    
    # Final clamp to ensure [0, 1]
    pred_proc = torch.clamp(pred_proc, 0, 1)
    target_proc = torch.clamp(target_proc, 0, 1)
    
    with torch.no_grad():
        gmsd_val = gmsd(pred_proc, target_proc, reduction=reduction)
    
    return float(gmsd_val.item())



def compute_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> float:

    _validate_tensor_shapes(pred, target, "NMSE")
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    
    nmse_per_image = numerator / denominator
    
    return float(nmse_per_image.mean().item())

def compute_vifp(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:

    _validate_tensor_shapes(pred, target, "VIFp")
    
    try:
        from piq import vif_p
    except ImportError as e:
        raise ImportError(
            "piq is required for VIFp computation. "
            "Install it with: pip install piq"
        ) from e
    
    pred, target = _ensure_compatible(pred, target)

    pred_3ch = _grayscale_to_3channel(pred)
    target_3ch = _grayscale_to_3channel(target)
    
    pred_min, pred_max = pred_3ch.min().item(), pred_3ch.max().item()
    target_min, target_max = target_3ch.min().item(), target_3ch.max().item()
    
    if (pred_min >= -1.0 and pred_max <= 1.0 and
        target_min >= -1.0 and target_max <= 1.0):
        if pred_min < 0.0 or target_min < 0.0:
            pred_3ch = (pred_3ch + 1.0) / 2.0
            target_3ch = (target_3ch + 1.0) / 2.0

    pred_3ch = torch.clamp(pred_3ch, 0, 1)
    target_3ch = torch.clamp(target_3ch, 0, 1)
    
    with torch.no_grad():
        vifp_val = vif_p(pred_3ch, target_3ch, reduction='mean')
    
    return float(vifp_val.item())


def _validate_tensor_shapes(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_name: str
) -> None:
    """
    Validate that input tensors have correct shape for metric computation.
    
    Expected shape: (B, 1, H, W) or (B, C, H, W)
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        metric_name: Name of the metric for error messages
    
    Raises:
        ValueError: If shapes are invalid or don't match
    """
    if not isinstance(pred, torch.Tensor):
        raise ValueError(
            f"{metric_name}: pred must be a torch.Tensor, got {type(pred)}"
        )
    
    if not isinstance(target, torch.Tensor):
        raise ValueError(
            f"{metric_name}: target must be a torch.Tensor, got {type(target)}"
        )
    
    if pred.ndim != 4:
        raise ValueError(
            f"{metric_name}: pred must have 4 dimensions (B, C, H, W), "
            f"got shape {pred.shape}"
        )
    
    if target.ndim != 4:
        raise ValueError(
            f"{metric_name}: target must have 4 dimensions (B, C, H, W), "
            f"got shape {target.shape}"
        )
    
    if pred.shape != target.shape:
        raise ValueError(
            f"{metric_name}: pred and target must have the same shape. "
            f"pred: {pred.shape}, target: {target.shape}"
        )
    
    if pred.shape[1] not in [1, 3]:
        warnings.warn(
            f"{metric_name}: Expected 1 or 3 channels, got {pred.shape[1]}. "
            "Results may be unexpected."
        )


def _ensure_compatible(
    pred: torch.Tensor,
    target: torch.Tensor
) -> tuple:
    """
    Ensure pred and target are on the same device with compatible dtypes.
    """
    # Ensure float32
    pred = pred.float()
    target = target.float()
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    return pred, target


def _grayscale_to_3channel(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1-channel grayscale tensor to 3-channel by repeating.
    
    Args:
        tensor: Input tensor of shape (B, 1, H, W) or (B, 3, H, W)
    
    Returns:
        Tensor of shape (B, 3, H, W)
    """
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] == 3:
        return tensor
    else:
        raise ValueError(
            f"Expected 1 or 3 channels, got {tensor.shape[1]}"
        )


# ============================================================================
# Batch Evaluation Function
# ============================================================================

class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for image quality assessment.
    
    Initializes all metric functions once for efficient repeated evaluation.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        lpips_net: str = 'alex'
    ):
        """
        Initialize the metrics evaluator.
        
        Args:
            device: Device for computation (auto-detected if None)
            lpips_net: Network for LPIPS ('alex', 'vgg', 'squeeze')
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        
        # Initialize LPIPS (this loads a network, so we do it once)
        try:
            self.lpips_metric = LPIPSMetric(net=lpips_net, device=device)
            self._has_lpips = True
        except ImportError as e:
            warnings.warn(f"LPIPS not available: {e}")
            self._has_lpips = False
        
        # Check for other dependencies
        try:
            from pytorch_msssim import ssim
            self._has_ssim = True
        except ImportError:
            warnings.warn("pytorch-msssim not available. SSIM will not be computed.")
            self._has_ssim = False
        
        try:
            from piq import gmsd
            self._has_gmsd = True
        except ImportError:
            warnings.warn("piq not available. GMSD will not be computed.")
            self._has_gmsd = False
        
        try:
            from piq import vif_p
            self._has_vifp = True
        except ImportError:
            warnings.warn("piq not available. VIFp will not be computed.")
            self._has_vifp = False
    
    def evaluate_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 2.0
    ) -> Dict[str, float]:
        """
        Compute all available metrics for a batch.
        
        Args:
            pred: Predicted images of shape (B, 1, H, W)
            target: Target images of shape (B, 1, H, W)
            data_range: Data range for PSNR/SSIM computation. Default 2.0 for [-1, 1] normalized data.
        
        Returns:
            Dictionary with metric names and values:
            {
                "psnr": float,
                "ssim": float,
                "lpips": float,
                "gmsd": float,
                "nmse": float,
                "vifp": float
            }
        """
        results = {}
        
        # PSNR (always available, pure PyTorch)
        results["psnr"] = compute_psnr(pred, target, data_range=data_range)
        
        # SSIM
        if self._has_ssim:
            results["ssim"] = compute_ssim(pred, target, data_range=data_range)
        else:
            results["ssim"] = float('nan')
        
        # LPIPS
        if self._has_lpips:
            results["lpips"] = self.lpips_metric(pred, target)
        else:
            results["lpips"] = float('nan')
        
        # GMSD
        if self._has_gmsd:
            results["gmsd"] = compute_gmsd(pred, target)
        else:
            results["gmsd"] = float('nan')
        
        # NMSE (always available, pure PyTorch)
        results["nmse"] = compute_nmse(pred, target)
        
        # VIFp
        if self._has_vifp:
            results["vifp"] = compute_vifp(pred, target)
        else:
            results["vifp"] = float('nan')
        
        return results


def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate a batch with all metrics.
    
    For repeated evaluation, use MetricsEvaluator class instead.
    
    Args:
        pred: Predicted images of shape (B, 1, H, W)
        target: Target images of shape (B, 1, H, W)
        data_range: Data range for PSNR/SSIM. Default 2.0 for [-1, 1] normalized data.
        device: Device for computation
    
    Returns:
        Dictionary with metric names and values:
        {
            "nmse": float,
            "psnr": float,
            "ssim": float,
            "lpips": float,
            "gmsd": float,
            "vifp": float
        }
    """
    evaluator = MetricsEvaluator(device=device)
    return evaluator.evaluate_batch(pred, target, data_range=data_range)

