"""Train-only augmentation for side ribbon pairs (Phase 5).

These transforms are applied ONLY to training pairs. Validation and test
pairs must remain unaugmented to keep metrics comparable.
"""

from __future__ import annotations

import numpy as np


class RibbonAugmenter:
    """Stochastic augmentations for ribbon image strips.

    Parameters
    ----------
    flip_prob : Probability of horizontal flip.
    jitter_range : Max additive brightness jitter (symmetric, per-channel).
    noise_std : Gaussian noise standard deviation.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        jitter_range: float = 15.0,
        noise_std: float = 5.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.flip_prob = flip_prob
        self.jitter_range = jitter_range
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()

    def __call__(self, ribbon: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a ribbon array (N, 3) or (H, W, C)."""
        out = ribbon.astype(np.float64).copy()

        # Horizontal flip
        if self.rng.random() < self.flip_prob:
            out = out[::-1].copy()

        # Brightness jitter (per-channel)
        if self.jitter_range > 0:
            jitter = self.rng.uniform(-self.jitter_range, self.jitter_range, size=out.shape[-1])
            out += jitter

        # Gaussian noise
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, size=out.shape)
            out += noise

        return np.clip(out, 0, 255)
