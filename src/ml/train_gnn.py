"""Train PuzzleGNN on synthetic grid graphs (known adjacency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.core.ribbons import pack_ribbon
from src.core.types import Piece, Side
from src.ml.gnn import PuzzleGNN
from src.ml.gnn_graph import SideGraph, build_side_graph


def _synthetic_piece(pid: int, profiles: list[np.ndarray], classes: list[str], rng: np.random.Generator) -> Piece:
    sides = []
    for i, (prof, cls) in enumerate(zip(profiles, classes)):
        colour = rng.uniform(30, 80, size=(32, 3))
        if cls != "flat":
            colour += (i * 3)
        sides.append(Side(
            index=i,
            cls=cls,  # type: ignore[arg-type]
            profile=prof,
            colour=colour,
            ribbon=pack_ribbon(colour, prof),
            contour_pts=np.zeros((8, 2)),
        ))
    return Piece(
        id=pid,
        image=np.zeros((8, 8, 3)),
        mask=np.zeros((8, 8), dtype=np.uint8),
        contour=np.zeros((8, 2), dtype=np.int32),
        bbox=(0, 0, 8, 8),
        pca_theta=0.0,
        corners=np.zeros((4, 2)),
        sides=sides,
    )


def make_synthetic_grid(rows: int = 3, cols: int = 3, rng: np.random.Generator | None = None) -> tuple[list[Piece], dict]:
    """Build an R×C puzzle with complementary tab/blank profiles on internal edges."""
    rng = rng or np.random.default_rng(0)
    n = rows * cols
    # side 0=N 1=E 2=S 3=W
    profiles = [[np.zeros(32) for _ in range(4)] for _ in range(n)]
    classes = [["flat"] * 4 for _ in range(n)]
    adjacency: dict[tuple[int, int], tuple[int, int]] = {}

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if r == 0:
                classes[i][0] = "flat"
            if r == rows - 1:
                classes[i][2] = "flat"
            if c == 0:
                classes[i][3] = "flat"
            if c == cols - 1:
                classes[i][1] = "flat"
            if c + 1 < cols:
                bump = rng.uniform(4, 10) * np.sin(np.linspace(0, 2 * np.pi, 32))
                classes[i][1] = "tab"
                classes[idx(r, c + 1)][3] = "blank"
                profiles[i][1] = bump
                profiles[idx(r, c + 1)][3] = -bump[::-1]
                adjacency[(i, 1)] = (idx(r, c + 1), 3)
            if r + 1 < rows:
                bump = rng.uniform(4, 10) * np.sin(np.linspace(0, 2 * np.pi, 32))
                classes[i][2] = "tab"
                classes[idx(r + 1, c)][0] = "blank"
                profiles[i][2] = bump
                profiles[idx(r + 1, c)][0] = -bump[::-1]
                adjacency[(i, 2)] = (idx(r + 1, c), 0)

    pieces = [_synthetic_piece(i, profiles[i], classes[i], rng) for i in range(n)]
    return pieces, adjacency


def graph_to_tensors(g: SideGraph, device: str) -> tuple[torch.Tensor, ...]:
    ribbons = torch.from_numpy(g.ribbons).to(device)
    src = torch.from_numpy(g.edge_src).to(device)
    dst = torch.from_numpy(g.edge_dst).to(device)
    et = torch.from_numpy(g.edge_type).to(device)
    y = torch.from_numpy(g.labels).to(device) if g.labels is not None else None
    inter = torch.from_numpy(g.inter_mask.astype(np.bool_)).to(device)
    return ribbons, src, dst, et, y, inter


def train_gnn(
    n_train: int = 40,
    n_val: int = 8,
    epochs: int = 40,
    lr: float = 5e-4,
    ckpt_path: str | Path = "checkpoints/gnn.pt",
    device: str | None = None,
    rows: int = 3,
    cols: int = 3,
) -> PuzzleGNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = PuzzleGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_graphs(k: int, seed: int) -> list[SideGraph]:
        rng = np.random.default_rng(seed)
        graphs = []
        for i in range(k):
            pieces, adj = make_synthetic_grid(rows, cols, rng=np.random.default_rng(int(rng.integers(0, 1_000_000))))
            graphs.append(build_side_graph(pieces, adjacency=adj, top_k=8))
        return graphs

    train_g = batch_graphs(n_train, 0)
    val_g = batch_graphs(n_val, 99)
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best, stalled, patience = -1.0, 0, 10

    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for g in train_g:
            ribbons, src, dst, et, y, inter = graph_to_tensors(g, device)
            p, _ = model(ribbons, src, dst, et)
            # BCE on inter-piece edges only; intra edges have no neighbour label
            loss = F.binary_cross_entropy(p[inter], y[inter])
            # peaky softmax over candidates of each side (mutual exclusion)
            n_nodes = ribbons.shape[0]
            peak = torch.tensor(0.0, device=device)
            inter_src = src[inter]
            p_inter = p[inter]
            for u in torch.unique(inter_src):
                sl = p_inter[inter_src == u]
                if sl.numel() > 1:
                    peak = peak + (-(sl / sl.sum().clamp(min=1e-6)).clamp(min=1e-6).log() * (sl / sl.sum().clamp(min=1e-6))).sum()
            loss = loss + 0.05 * peak / max(n_nodes, 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss.detach())

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for g in val_g:
                ribbons, src, dst, et, y, inter = graph_to_tensors(g, device)
                p, _ = model(ribbons, src, dst, et)
                pred = (p[inter] >= 0.5).cpu().numpy()
                yt = y[inter].cpu().numpy() >= 0.5
                tp += int(np.sum(pred & yt))
                fp += int(np.sum(pred & ~yt))
                fn += int(np.sum(~pred & yt))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"epoch {epoch+1:02d}  train_loss={tr/len(train_g):.4f}  val_f1={f1:.3f}")
        if f1 > best:
            best, stalled = f1, 0
            torch.save({"model": model.state_dict(), "val_f1": f1}, ckpt_path)
        else:
            stalled += 1
            if stalled >= patience:
                print("early stop")
                break

    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(blob["model"])
    return model
