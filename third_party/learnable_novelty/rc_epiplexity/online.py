"""Online covariance-form RLS for temporal (per-step) epiplexity.

The batch estimator in :mod:`rc_epiplexity.core` recomputes the ridge readout
from scratch at every time step, costing ``O(T^2 d^2)`` to trace the whole
cumulative-epiplexity curve ``S_t`` along a length-``T`` stream. This module
implements the exact recursive least-squares (RLS) recursion, which advances
``P = A^{-1}`` and ``W = A^{-1} B`` by Sherman-Morrison rank-1 updates in
``O(d^2)`` per step, so the full ``S_t`` trajectory costs ``O(T d^2)``.

The program length reuses the same singular-value log-volume as
:meth:`RCEpiplexity.multioutput_epiplexity`, so ``S_t`` matches the batch
estimator bit-for-bit under identical normalization (verified to ~1e-12).

One caveat (also stated in the paper's temporal-accumulation appendix): the exact rank-1 recursion requires the
feature/target normalization to be *frozen* in advance. A running, per-step
normalization rewrites every historical row and is not a low-rank perturbation,
so it cannot be folded into the recursion. The caller must therefore normalize
with statistics fixed before streaming (a warm-up prefix or the full stream).
This class does no normalization itself -- it consumes already-normalized rows.
"""

from __future__ import annotations

import torch

from .core import svd_log_volume


class CovarianceRLSEpiplexity:
    """Recursive (covariance-form) estimator of cumulative epiplexity ``S_t``.

    Maintains the inverse Gram matrix ``P = (lambda I + sum phi phi^T)^{-1}`` and
    the readout ``W = P @ (sum phi y^T)`` and updates both by a rank-1 step per
    observation. Call :meth:`update` once per ``(phi, y)`` pair, then read
    :meth:`epiplexity` for the cumulative score after the pairs seen so far.

    Args:
        feature_dim: Reservoir feature dimension ``d``.
        target_dim: Output dimension ``M``.
        ridge_lambda: Ridge penalty ``lambda``.
        device: Torch device.
        dtype: Working precision. Defaults to ``torch.float64``; double
            precision is recommended (the Gram condition number grows linearly
            with the stream length).

    Notes:
        Feed already-normalized rows. The recursion is exact only when the
        normalization is frozen before streaming.
    """

    def __init__(
        self,
        feature_dim: int,
        target_dim: int,
        ridge_lambda: float,
        lambda_code: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.feature_dim = feature_dim
        self.target_dim = target_dim
        self.ridge_lambda = ridge_lambda
        self.lambda_code = lambda_code
        self.device = torch.device(device)
        self.dtype = dtype
        self.P = torch.eye(feature_dim, device=self.device, dtype=dtype) / ridge_lambda
        self.W = torch.zeros(feature_dim, target_dim, device=self.device, dtype=dtype)

    def update(self, phi: torch.Tensor, y: torch.Tensor) -> None:
        """Fold one observation ``(phi, y)`` into ``P`` and ``W`` (rank-1, O(d^2)).

        Args:
            phi: Normalized feature row, shape ``(feature_dim,)``.
            y: Normalized target row, shape ``(target_dim,)``.
        """
        phi = phi.to(self.device, self.dtype)
        y = y.to(self.device, self.dtype)
        p_phi = self.P @ phi
        gain = p_phi / (1.0 + phi @ p_phi)  # u = A_t^{-1} phi (post-update gain)
        innovation = y - self.W.T @ phi
        self.W = self.W + torch.outer(gain, innovation)
        self.P = self.P - torch.outer(gain, p_phi)
        self.P = 0.5 * (self.P + self.P.T)  # re-symmetrize for long-run stability

    def epiplexity(self) -> torch.Tensor:
        """Cumulative epiplexity: singular-value log-volume of ``W``.

        Matches :meth:`RCEpiplexity.multioutput_epiplexity` so scale enters
        additively and redundant output coordinates are priced at zero.
        """
        return svd_log_volume(self.W, self.lambda_code)
