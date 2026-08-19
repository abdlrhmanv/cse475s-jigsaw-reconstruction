"""Train SiameseNet on synthetic complementary ribbons + optional SidePair lists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.core.ribbons import RIBBON_LEN, pack_ribbon
from src.ml.pair_dataset import SidePair
from src.ml.siamese import SiameseNet


class RibbonPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray, int, int]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, y, rot = self.pairs[idx]
        return (
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(rot, dtype=torch.long),
        )


def side_pairs_to_tensors(pairs: list[SidePair]) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    out: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for p in pairs:
        a = p.ribbon_a if p.ribbon_a.ndim == 2 and p.ribbon_a.shape[0] == 4 else pack_ribbon(p.ribbon_a, p.profile_a)
        b = p.ribbon_b if p.ribbon_b.ndim == 2 and p.ribbon_b.shape[0] == 4 else pack_ribbon(p.ribbon_b, p.profile_b)
        rot = (p.side_idx_a - p.side_idx_b) % 4 if p.label == 1 else 0
        out.append((a.astype(np.float32), b.astype(np.float32), int(p.label), int(rot)))
    return out


def make_synthetic_pairs(n_pos: int = 256, n_neg: int = 768, rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Geometric positives: profile_b ≈ −flip(profile_a), colour_b ≈ flip(colour_a)."""
    rng = rng or np.random.default_rng(0)
    pairs: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    t = np.linspace(0, 2 * np.pi, RIBBON_LEN)

    def one_side() -> tuple[np.ndarray, np.ndarray]:
        profile = 8.0 * np.sin(t) * rng.uniform(0.6, 1.4) + rng.normal(0, 0.3, RIBBON_LEN)
        colour = np.stack(
            [
                rng.uniform(40, 90) + 5 * np.sin(t),
                rng.uniform(-20, 20) + 3 * np.cos(t),
                rng.uniform(-20, 20) + 3 * np.sin(2 * t),
            ],
            axis=-1,
        )
        return colour, profile

    for _ in range(n_pos):
        ca, pa = one_side()
        cb = ca[::-1].copy() + rng.normal(0, 1.0, ca.shape)
        pb = -pa[::-1].copy() + rng.normal(0, 0.2, pa.shape)
        rot = int(rng.integers(0, 4))
        pairs.append((pack_ribbon(ca, pa), pack_ribbon(cb, pb), 1, rot))

    for _ in range(n_neg):
        ca, pa = one_side()
        cb, pb = one_side()
        pairs.append((pack_ribbon(ca, pa), pack_ribbon(cb, pb), 0, 0))
    return pairs


def train_siamese(
    pairs: list[tuple[np.ndarray, np.ndarray, int, int]] | None = None,
    val_pairs: list[tuple[np.ndarray, np.ndarray, int, int]] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_rot: float = 0.5,
    ckpt_path: str | Path = "checkpoints/siamese.pt",
    device: str | None = None,
) -> SiameseNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if pairs is None:
        all_pairs = make_synthetic_pairs()
        rng = np.random.default_rng(1)
        rng.shuffle(all_pairs)
        split = int(0.8 * len(all_pairs))
        pairs, val_pairs = all_pairs[:split], all_pairs[split:]
    if val_pairs is None:
        val_pairs = pairs[-max(1, len(pairs) // 10) :]

    model = SiameseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    train_loader = DataLoader(RibbonPairDataset(pairs), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RibbonPairDataset(val_pairs), batch_size=batch_size)

    best_f1 = -1.0
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    patience, stalled = 7, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for a, b, y, rot in train_loader:
            a, b, y, rot = a.to(device), b.to(device), y.to(device), rot.to(device)
            p, rot_logits, e1, e2 = model(a, b)
            bce = F.binary_cross_entropy(p, y)
            pos = y > 0.5
            rot_loss = torch.tensor(0.0, device=device)
            if pos.any():
                rot_loss = F.cross_entropy(rot_logits[pos], rot[pos])
            dist = torch.norm(e1 - e2, dim=-1)
            contrast = (y * dist.pow(2) + (1 - y) * F.relu(1.0 - dist).pow(2)).mean()
            loss = bce + lambda_rot * rot_loss + 0.1 * contrast
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * len(y)

        model.eval()
        tp = fp = fn = 0
        val_bce = 0.0
        n_val = 0
        with torch.no_grad():
            for a, b, y, rot in val_loader:
                a, b, y = a.to(device), b.to(device), y.to(device)
                p, _, _, _ = model(a, b)
                val_bce += float(F.binary_cross_entropy(p, y)) * len(y)
                n_val += len(y)
                pred = (p >= 0.5).cpu().numpy()
                yt = y.cpu().numpy() >= 0.5
                tp += int(np.sum(pred & yt))
                fp += int(np.sum(pred & ~yt))
                fn += int(np.sum(~pred & yt))
        val_bce /= max(n_val, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        sched.step(val_bce)
        print(
            f"epoch {epoch+1:02d}  train_loss={train_loss/max(len(pairs),1):.4f}  "
            f"val_bce={val_bce:.4f}  val_f1={f1:.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            stalled = 0
            torch.save({"model": model.state_dict(), "val_f1": f1}, ckpt_path)
        else:
            stalled += 1
            if stalled >= patience:
                print("early stop")
                break

    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(blob["model"])
    return model
