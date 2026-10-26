"""Core normalized ridge-readout epiplexity utilities.

The program length of the learned readout is scored by the spectral
log-determinant code (in bits, via a base-2 logarithm)

    S_RC = alpha * log2 det(I + eta * w w^T)
         = alpha * sum_i log2(1 + eta * s_i^2),

where ``s_i`` are the singular values of the readout matrix ``w*`` (one column
per output coordinate). In the code ``alpha`` is the fixed prefactor ``0.5`` and
``eta`` is the resolution parameter named ``lambda_code``. This form has two
properties the earlier ``0.5 * lambda * ||w*||^2`` lacked:

* Scale enters additively. Doubling the signal sends ``s_i -> 2 s_i`` and the
  score grows by ``~log 2`` per active direction (the algorithmic-information
  behaviour ``S -> S + log k``), instead of the quadratic ``k^2`` blow-up of a
  squared norm.
* Redundant output coordinates are free. A duplicated target column adds a zero
  singular value, and ``log(1 + 0) = 0`` contributes nothing -- so the explicit
  column-pivoted QR that used to deduplicate directions is no longer needed.

Targets are centered and divided by a *fixed* unit ``sigma_y`` rather than by
their empirical standard deviation, so the target's own scale survives as
information (this matters for tasks where magnitude is meaningful). Features are
per-coordinate standardized and then divided by ``sqrt(d)`` so that a random
readout produces an ``O(1)`` output regardless of the feature dimension ``d``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# Program length is reported in bits, so the singular-value log-volume uses a
# base-2 logarithm. We compute it as log1p(.) / ln(2) to keep log1p's accuracy
# near zero while converting to base 2.
_LN2 = math.log(2.0)


@dataclass(frozen=True)
class RCEpiplexityOutput:
    epiplexity: torch.Tensor
    residual: torch.Tensor


class RCEpiplexity(nn.Module):
    """Minimal base class for concrete reservoir epiplexity estimators."""

    def __init__(
        self,
        feature_dim: int,
        ridge_lambda: float = 0.1,
        sigma_y: float = 1.0,
        lambda_code: float = 1.0,
        eps: float = 1e-8,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.ridge_lambda = ridge_lambda
        self.sigma_y = sigma_y
        self.lambda_code = lambda_code
        self.eps = eps
        self.device = torch.device(device)

    def init(self, seed: int | None = None) -> "RCEpiplexity":
        """Reset random reservoir parameters."""
        if seed is not None:
            torch.manual_seed(seed)
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        return self

    def multioutput_epiplexity(
        self, features: torch.Tensor, target: torch.Tensor
    ) -> "RCEpiplexityOutput":
        """Normalized ridge + singular-value log-volume epiplexity.

        Each output coordinate gets its own ridge readout from the shared
        per-sample reservoir features (one ``vmap``-ed solve per coordinate),
        giving a readout matrix ``w*`` of shape ``(feature_dim, output_dim)``.
        The program length is the singular-value log-volume of ``w*`` (see the
        module docstring), which charges scale additively and prices redundant
        output coordinates at zero.

        Args:
            features (torch.Tensor): Per-sample reservoir features, shape
                ``(batch, feature_dim)``.
            target (torch.Tensor): Multi-output target, shape
                ``(batch, output_dim)``.

        Returns:
            RCEpiplexityOutput: Epiplexity and the per-sample readout residual.
        """
        feature_dim = features.shape[1]

        # Target: center, then divide by a FIXED unit sigma_y (not the data std),
        # so the target's own scale is preserved as information.
        target = (target - torch.mean(target, dim=0, keepdim=True)) / self.sigma_y
        # Features: per-coordinate standardize, then /sqrt(d) so a random readout
        # produces an O(1) output regardless of the feature dimension.
        features = (features - torch.mean(features, dim=0, keepdim=True)) / (
            torch.std(features, dim=0, correction=0, keepdim=True)
            * (feature_dim**0.5)
            + self.eps
        )

        # One ridge readout per output coordinate -> w shape (feature_dim, output_dim).
        w = torch.vmap(
            lambda target_component: ridge_least_squares(
                features, target_component, self.ridge_lambda
            )
        )(target.mT).mT
        residual = target - features @ w
        epiplexity = svd_log_volume(w, self.lambda_code)
        return RCEpiplexityOutput(
            epiplexity=epiplexity,
            residual=residual,
        )


def ridge_least_squares(
    features: torch.Tensor,
    y: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Solve a no-bias scalar ridge readout.

    Finds one readout vector ``w`` by solving

        minimize_w ||features @ w - y||_2^2 + ridge_lambda * ||w||_2^2.

    Args:
        features (torch.Tensor): Feature matrix with shape ``(N, C)``.
            ``N`` is the number of observations. ``C`` is the reservoir feature
            dimension. Each row is one observation.
        y (torch.Tensor): Target vector with shape ``(N,)``. This is one
            scalar target for each observation.
        ridge_lambda (float): Ridge penalty coefficient.

    Shape:
        - features: ``(N, C)``
        - y: ``(N,)``
        - Output: ``(C,)``

    Returns:
        torch.Tensor: Readout vector ``w`` with shape ``(C,)``.

    Notes:
        This function solves one scalar target. For vector-valued targets such
        as MLP outputs, call it once per target component and keep the
        component-wise readouts outside core code.

        The caller is responsible for any experiment-specific normalization.
        Internally, the ridge problem is solved as a stable least-squares
        problem rather than by explicitly forming the normal equations.
    """
    feature_dim = features.shape[1]
    identity = torch.eye(feature_dim, dtype=features.dtype, device=features.device)
    zeros = torch.zeros(feature_dim, dtype=y.dtype, device=y.device)
    augmented_features = torch.cat([features, (ridge_lambda**0.5) * identity], dim=0)
    augmented_y = torch.cat([y, zeros], dim=0)
    q, r = torch.linalg.qr(augmented_features, mode="reduced")
    rhs = q.mT @ augmented_y
    return torch.linalg.solve_triangular(r, rhs.unsqueeze(1), upper=True).squeeze(1)


def svd_log_volume(w: torch.Tensor, lambda_code: float) -> torch.Tensor:
    """Spectral log-determinant program length of a readout matrix.

    Returns ``0.5 * sum_i log2(1 + lambda_code * s_i^2)`` where ``s_i`` are the
    singular values of ``w``; equivalently ``0.5 * log2 det(I + lambda_code * w
    w^T)``. The base-2 logarithm puts the program length in bits. For a
    single-output readout (``w`` a column vector) this reduces to
    ``0.5 * log2(1 + lambda_code * ||w||_2^2)``.

    ``lambda_code`` is the resolution parameter ``eta``: directions with
    ``s_i^2`` well above ``1 / lambda_code`` each contribute about
    ``log(lambda_code * s_i^2)`` bits, while directions far below it contribute
    almost nothing. ``torch.linalg.svdvals`` is differentiable, so the result is
    differentiable w.r.t. ``w``.

    Args:
        w (torch.Tensor): Readout matrix, shape ``(feature_dim, output_dim)``.
        lambda_code (float): Resolution parameter ``eta``.

    Returns:
        torch.Tensor: Scalar program length.
    """
    s = torch.linalg.svdvals(w)
    return 0.5 * torch.sum(torch.log1p(lambda_code * s.square())) / _LN2
