"""Concrete reservoirs for normalized ridge epiplexity."""

from __future__ import annotations

import torch
import torch.nn as nn

from .core import RCEpiplexity, RCEpiplexityOutput


class PreActNorm(nn.Module):
    """Normalize over the feature/channel axis right before a nonlinearity.

    For an MLP feature vector ``(batch, channels)`` this is a LayerNorm over the
    channels; for a conv map ``(batch, channels, ...)`` it normalizes each
    spatial site's channel vector. Inserting it before every activation pins the
    pre-activation variance, which holds a random reservoir at the edge of chaos
    (per-layer perturbation multiplier ~ 1) independent of depth, input scale,
    and architecture -- the feedforward analogue of the echo-state spectral
    radius. No learnable affine: the reservoir stays fixed and random.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)


class RCEpiplexity1D(RCEpiplexity):
    """Circular 1D reservoir with one readout shared across spatial sites.

    Inputs have shape ``(batch, input_channels, space)`` and targets have shape
    ``(batch, target_channels, space)``; both default to a single channel (ECA).
    Each target channel gets its own ridge readout from the shared per-site
    reservoir features.
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        kernel_size: int,
        input_channels: int = 1,
        ridge_lambda: float = 0.1,
        sigma_y: float = 1.0,
        lambda_code: float = 1.0,
        eps: float = 1e-8,
        device: str | torch.device = "cpu",
    ):
        super().__init__(channels, ridge_lambda, sigma_y, lambda_code, eps, device)
        self.channels = channels
        layers = []

        in_channels = input_channels
        for layer_idx in range(depth):
            is_last = layer_idx == depth - 1
            current_kernel = 1 if is_last else kernel_size
            padding = current_kernel // 2
            layers.append(
                nn.Conv1d(
                    in_channels,
                    channels,
                    current_kernel,
                    padding=padding,
                    padding_mode="circular" if padding > 0 else "zeros",
                    bias=True,
                    device=self.device,
                )
            )
            if not is_last:
                layers.append(PreActNorm())
                layers.append(nn.ELU())
            in_channels = channels
        self.phi = nn.Sequential(*layers)

    def epiplexity(self, x: torch.Tensor, y: torch.Tensor) -> RCEpiplexityOutput:
        # x.shape = (batch, input_channels, space); y.shape = (batch, target_channels, space)
        self.eval()
        h = self.phi(x)  # shape (batch, channels, space)
        # One scalar target per (site, target channel) -> (batch * space, channels)
        # features and a (batch * space, target_channels) target.
        features = h.movedim(1, 2).reshape(-1, self.channels)
        target = y.movedim(1, 2).reshape(-1, y.shape[1])
        return self.multioutput_epiplexity(features, target)


class RCEpiplexityMLP(RCEpiplexity):
    """MLP reservoir with one independent readout per output component.

    The program length is the singular-value log-volume of the stacked
    per-component readouts, so duplicated or linearly dependent output
    coordinates are not charged repeatedly (see ``svd_log_volume``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_dim: int,
        ridge_lambda: float = 0.1,
        sigma_y: float = 1.0,
        lambda_code: float = 1.0,
        eps: float = 1e-8,
        device: str | torch.device = "cpu",
    ):
        super().__init__(hidden_dim, ridge_lambda, sigma_y, lambda_code, eps, device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=self.device),
            PreActNorm(),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim, device=self.device),
            PreActNorm(),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim, device=self.device),
            PreActNorm(),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim, device=self.device),
        )

    def epiplexity(self, x: torch.Tensor, y: torch.Tensor) -> RCEpiplexityOutput:
        # x.shape = (batch, input_dim)
        # y.shape = (batch, target_dim)
        self.eval()
        h = self.phi(x)  # shape (batch, hidden_dim)
        return self.multioutput_epiplexity(h, y)
