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
from src.ml.train_metrics import binary_pr_metrics


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


def side_pairs_to_tensors(
    pairs: list[SidePair],
    *,
    rot_known: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Pack ribbons. Positive relative ori is ``(si − sj) mod 4``, not photo rot."""
    from src.ml.pose_gt import relative_orient

    out: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for p in pairs:
        a = p.ribbon_a if p.ribbon_a.ndim == 2 and p.ribbon_a.shape[0] == 4 else pack_ribbon(p.ribbon_a, p.profile_a)
        b = p.ribbon_b if p.ribbon_b.ndim == 2 and p.ribbon_b.shape[0] == 4 else pack_ribbon(p.ribbon_b, p.profile_b)
        a = np.nan_to_num(a.astype(np.float32), nan=0.0)
        b = np.nan_to_num(b.astype(np.float32), nan=0.0)
        if p.label == 1 and rot_known:
            rot = p.rel_orient if p.rel_orient >= 0 else relative_orient(p.side_idx_a, p.side_idx_b)
        elif p.label == 1:
            rot = -1
        else:
            rot = 0
        out.append((a, b, int(p.label), int(rot)))
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
        rot = 0
        pairs.append((pack_ribbon(ca, pa), pack_ribbon(cb, pb), 1, rot))

    for _ in range(n_neg):
        ca, pa = one_side()
        cb, pb = one_side()
        pairs.append((pack_ribbon(ca, pa), pack_ribbon(cb, pb), 0, 0))
    return pairs


def load_real_pair_tensors(
    data_dir: str | Path = "data",
    max_boards: int = 80,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int, int]], list[tuple[np.ndarray, np.ndarray, int, int]]]:
    from src.ml.real_dataset import pairs_from_split

    print(f"Building real Siamese pairs (max_boards={max_boards})...")
    train = pairs_from_split("train", data_dir, max_boards=max_boards, augment=True)
    val = pairs_from_split("val", data_dir, max_boards=max(8, max_boards // 5), augment=False)
    print(f"Real pairs: train={len(train)}  val={len(val)}")
    return side_pairs_to_tensors(train, rot_known=True), side_pairs_to_tensors(val, rot_known=True)


def train_siamese(
    pairs: list[tuple[np.ndarray, np.ndarray, int, int]] | None = None,
    val_pairs: list[tuple[np.ndarray, np.ndarray, int, int]] | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_rot: float = 0.5,
    ckpt_path: str | Path = "checkpoints/siamese.pt",
    device: str | None = None,
    real: bool = False,
    max_boards: int = 80,
    data_dir: str | Path = "data",
) -> SiameseNet:
    torch.manual_seed(42)
    np.random.seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if pairs is None and real:
        pairs, val_pairs = load_real_pair_tensors(data_dir, max_boards=max_boards)
        if not pairs:
            print("No real pairs found; falling back to synthetic ribbons.")
            pairs = None
        else:
            synth = make_synthetic_pairs(n_pos=512, n_neg=512, rng=np.random.default_rng(7))
            pairs = list(pairs) + synth
            print(f"Mixed in {len(synth)} synthetic complementary pairs for geometry.")
    if pairs is None:
        all_pairs = make_synthetic_pairs()
        rng = np.random.default_rng(1)
        rng.shuffle(all_pairs)
        split = int(0.8 * len(all_pairs))
        pairs, val_pairs = all_pairs[:split], all_pairs[split:]
    if val_pairs is None:
        val_pairs = pairs[-max(1, len(pairs) // 10) :]

    n_pos = sum(1 for _a, _b, y, _r in pairs if y == 1)
    n_neg = max(len(pairs) - n_pos, 1)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Train labels: pos={n_pos} neg={n_neg}  pos_weight={pos_weight:.2f}")

    model = SiameseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    train_loader = DataLoader(RibbonPairDataset(pairs), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RibbonPairDataset(val_pairs), batch_size=batch_size)

    best_ap = -1.0
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    patience, stalled = 7, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for a, b, y, rot in train_loader:
            a, b, y, rot = a.to(device), b.to(device), y.to(device), rot.to(device)
            p, rot_logits, e1, e2 = model(a, b)
            p = p.clamp(1e-6, 1 - 1e-6)
            sample_w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
            bce = F.binary_cross_entropy(p, y, weight=sample_w)
            pos = y > 0.5
            rot_ok = pos & (rot >= 0)
            rot_loss = torch.tensor(0.0, device=device)
            if rot_ok.any():
                rot_loss = F.cross_entropy(rot_logits[rot_ok], rot[rot_ok])
            dist = torch.norm(e1 - e2, dim=-1)
            contrast = (y * dist.pow(2) + (1 - y) * F.relu(1.0 - dist).pow(2)).mean()
            loss = bce + lambda_rot * rot_loss + 0.1 * contrast
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * len(y)

        model.eval()
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        val_bce = 0.0
        n_val = 0
        ori_ok = 0
        ori_n = 0
        with torch.no_grad():
            for a, b, y, rot in val_loader:
                a, b, y, rot = a.to(device), b.to(device), y.to(device), rot.to(device)
                p, rot_logits, _, _ = model(a, b)
                p = p.clamp(1e-6, 1 - 1e-6)
                val_bce += float(F.binary_cross_entropy(p, y)) * len(y)
                n_val += len(y)
                scores.append(p.cpu().numpy())
                labels.append(y.cpu().numpy())
                mask = (y > 0.5) & (rot >= 0)
                if mask.any():
                    pred = rot_logits.argmax(dim=-1)
                    ori_ok += int((pred[mask] == rot[mask]).sum().item())
                    ori_n += int(mask.sum().item())
        val_bce /= max(n_val, 1)
        metrics = binary_pr_metrics(np.concatenate(scores), np.concatenate(labels))
        sched.step(val_bce)
        ori_acc = ori_ok / ori_n if ori_n else float("nan")
        print(
            f"epoch {epoch+1:02d}  train_loss={train_loss/max(len(pairs),1):.4f}  "
            f"val_bce={val_bce:.4f}  val_f1@0.5={metrics['f1']:.3f}  "
            f"val_f1*={metrics['f1_best']:.3f}@t={metrics['t_best']:.2f}  "
            f"val_ap={metrics['ap']:.3f}  val_ori={ori_acc:.3f}  "
            f"p_pos={metrics['p_pos']:.3f}  p_neg={metrics['p_neg']:.3f}"
        )
        if metrics["ap"] > best_ap:
            best_ap = metrics["ap"]
            stalled = 0
            torch.save(
                {"model": model.state_dict(), "val_f1": metrics["f1_best"], "val_ap": metrics["ap"]},
                ckpt_path,
            )
        else:
            stalled += 1
            if stalled >= patience:
                print("early stop")
                break

    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(blob["model"])
    return model
