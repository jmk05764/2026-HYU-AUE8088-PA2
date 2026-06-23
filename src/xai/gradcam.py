"""Grad-CAM (Selvaraju et al., ICCV 2017) — multi-task aware.

We attach hooks to the last conv layer (or the final feature map of a
ViT) and compute a separate CAM for each of the three attribute heads.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Single-target Grad-CAM. Use one instance per attribute."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self,
        x: torch.Tensor,
        score_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Compute a CAM for the score returned by ``score_fn``.

        Example for ``weather`` head, predicted class:

            cam = gc(x, lambda out: out["weather"].max(dim=-1).values.sum())
        """
        self.model.zero_grad()
        out = self.model(x)
        score = score_fn(out)
        score.backward(retain_graph=True)

        # Activations/Gradients: (B, C, H, W) for CNN, (B, N+1, D) for ViT.
        a = self._activations
        g = self._gradients

        if a.dim() == 4:
            # CNN path: global-average-pool gradients over spatial dims.
            weights = g.mean(dim=(2, 3), keepdim=True)            # (B, C, 1, 1)
            cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # (B, 1, H, W)
        else:
            # ViT path: token sequence (B, N+1, D) — index 0 is CLS token.
            # Element-wise product summed over embedding dim gives per-token score.
            cam = F.relu((g * a).sum(dim=-1))  # (B, N+1)
            cam = cam[:, 1:]                    # drop CLS → (B, N)
            n = cam.shape[1]
            h = w = int(n ** 0.5)
            cam = cam.reshape(cam.shape[0], 1, h, w)  # (B, 1, h_p, w_p)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Per-image normalization to [0, 1].
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(1)
