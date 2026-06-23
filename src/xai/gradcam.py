"""Grad-CAM (Selvaraju et al., ICCV 2017) — multi-task aware.

We attach hooks to the last conv layer (or the final feature map of a
ViT) and compute a separate CAM for each of the three attribute heads.
"""
from __future__ import annotations

from typing import Callable
import math

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

    @staticmethod
    def _tokens_to_feature_map(x: torch.Tensor) -> torch.Tensor:
        """Convert ViT tokens ``(B, N, D)`` to ``(B, D, H, W)``.

        If a CLS token is present, it is removed before reshaping patch
        tokens into a square grid.
        """
        b, n, d = x.shape
        grid = int(math.sqrt(n))
        if grid * grid == n:
            patches = x
        else:
            grid = int(math.sqrt(n - 1))
            if grid * grid != n - 1:
                raise ValueError(f"Cannot reshape {n} tokens into a square Grad-CAM map.")
            patches = x[:, 1:, :]

        return patches.transpose(1, 2).reshape(b, d, grid, grid)

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

        a = self._activations
        g = self._gradients
        if a is None or g is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # CNN target layers produce (B, C, H, W). ViT target layers produce
        # (B, N, D), where N may include a leading CLS token.
        if a.ndim == 3:
            a = self._tokens_to_feature_map(a)
            g = self._tokens_to_feature_map(g)
        elif a.ndim != 4:
            raise ValueError(f"Unsupported activation shape for Grad-CAM: {tuple(a.shape)}")

        weights = g.mean(dim=(2, 3), keepdim=True)            # (B, C, 1, 1)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # (B, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Per-image normalization to [0, 1].
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(1)
