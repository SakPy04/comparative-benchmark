import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from natsort import natsorted


def center_crop_or_pad(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Center-crop or zero-pad a tensor to (target_h, target_w).

    Supports:
        (H, W), (1, H, W), (C, H, W)
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    elif img.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(img.shape)}")

    _, h, w = img.shape
    dh = target_h - h
    dw = target_w - w

    pad_top = max(dh // 2, 0)
    pad_bottom = max(dh - pad_top, 0)
    pad_left = max(dw // 2, 0)
    pad_right = max(dw - pad_left, 0)

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

    h, w = img.shape[-2:]
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)

    return img[..., start_h:start_h + target_h, start_w:start_w + target_w]


class UnifiedH5PairDataset(Dataset):
    """
    Unified HDF5 dataset for all reconstruction models.

    Returns a common batch structure:
        {
            "lq": Tensor (1, H, W),
            "hq": Tensor (1, H, W),
            "case_id": str,
            "input_key": str,
            "target_key": str,
            "file_path": str,
        }
    """

    def __init__(
        self,
        data_dir: str,
        input_key: str,
        target_key: str = "full",
        crop_size: Optional[Tuple[int, int]] = (640, 320),
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        preload: bool = False,
        allowed_exts: Tuple[str, ...] = (".h5", ".hdf5"),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.input_key = input_key
        self.target_key = target_key
        self.crop_size = crop_size
        self.normalize_range = normalize_range
        self.preload = preload

        files = natsorted(os.listdir(self.data_dir))

        candidates = [
            str(self.data_dir / f)
            for f in files
            if f.endswith(allowed_exts)
        ]

        valid_files: List[str] = []
        for path in candidates:
            try:
                with h5py.File(path, "r") as h5:
                    if self.input_key in h5 and self.target_key in h5:
                        valid_files.append(path)
            except OSError:
                continue

        if not valid_files:
            raise RuntimeError(
                f"No valid HDF5 files found in {data_dir} with keys "
                f"'{self.input_key}' and '{self.target_key}'."
            )

        self.files = valid_files
        self.data: Optional[List[Dict[str, Any]]] = None

        if self.preload:
            self.data = [self._load_item(path) for path in self.files]

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def min_max_scale(
        x: torch.Tensor,
        out_min: float = -1.0,
        out_max: float = 1.0
    ) -> torch.Tensor:
        x_min = x.min()
        x_max = x.max()
        denom = x_max - x_min

        if denom <= 0:
            return torch.full_like(x, (out_min + out_max) / 2.0)

        x_std = (x - x_min) / denom
        return x_std * (out_max - out_min) + out_min

    def _load_item(self, file_path: str) -> Dict[str, Any]:
        case_id = Path(file_path).stem

        with h5py.File(file_path, "r") as h5:
            hq = torch.from_numpy(h5[self.target_key][()]).float()
            lq = torch.from_numpy(h5[self.input_key][()]).float()

        out_min, out_max = self.normalize_range
        hq = self.min_max_scale(hq, out_min, out_max).unsqueeze(0).contiguous()
        lq = self.min_max_scale(lq, out_min, out_max).unsqueeze(0).contiguous()

        if self.crop_size is not None:
            h, w = self.crop_size
            hq = center_crop_or_pad(hq, h, w)
            lq = center_crop_or_pad(lq, h, w)

        if not torch.isfinite(hq).all():
            raise ValueError(f"Non-finite values in target for file {file_path}")
        if not torch.isfinite(lq).all():
            raise ValueError(f"Non-finite values in input for file {file_path}")

        return {
            "hq": hq,
            "lq": lq,
            "case_id": case_id,
            "input_key": self.input_key,
            "target_key": self.target_key,
            "file_path": file_path,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data is not None:
            return self.data[idx]
        return self._load_item(self.files[idx])