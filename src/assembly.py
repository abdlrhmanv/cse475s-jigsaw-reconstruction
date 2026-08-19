"""Greedy best-first assembly and canvas reconstruction (Phase 4)."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Assembler, ImageReconstructor
from src.core.types import AssemblyState, CompatibilityTensor, Piece, Placement, Puzzle
from src.edge_matching import ILLEGAL_COST

BORDER_PENALTY = 5.0e4
INTERIOR_FLAT_PENALTY = 5.0e4


class GreedyBestFirstAssembler(Assembler):
    """Grow the board from the most confident adjacent placement.

    Seeds the top-left cell with a corner piece (flats facing north+west).
    Illegal geometry (inf dissim) is skipped. If no legal move remains, the
    best partial arrangement is returned instead of force-filling cells.
    Tie-break: more pieces placed, then lower total dissim, then lower id/rot.
    """

    def __init__(self, beam_k: int = 8) -> None:
        self.beam_k = beam_k

    def assemble(
        self,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        rows: int,
        cols: int,
    ) -> AssemblyState:
        n = len(pieces)
        empty = AssemblyState(grid=[[None] * cols for _ in range(rows)], used=set())
        if n == 0:
            return empty

        seeds = self._seed_states(pieces, rows, cols)
        if not seeds:
            return empty
        seeds.sort(key=lambda s: (s.total_dissim, min(s.used), 0))
        beams = seeds[: max(self.beam_k, 1)]
        best = self._pick_best(beams)
        target = min(rows * cols, n)

        for _ in range(target - 1):
            next_beams: list[AssemblyState] = []
            grew = False
            for state in beams:
                if len(state.used) >= target:
                    next_beams.append(state)
                    continue
                children = self._expand_frontier(state, pieces, tensor, rows, cols)
                growing = [c for c in children if len(c.used) > len(state.used)]
                if growing:
                    next_beams.extend(growing)
                    grew = True
                else:
                    next_beams.append(state)
            if not next_beams:
                break
            next_beams.sort(
                key=lambda s: (-len(s.used), s.total_dissim, min(s.used) if s.used else 0)
            )
            beams = next_beams[: self.beam_k]
            cand = self._pick_best(beams)
            if self._better(cand, best):
                best = cand
            if not grew:
                break

        return best

    def _seed_states(self, pieces: list[Piece], rows: int, cols: int) -> list[AssemblyState]:
        n = len(pieces)
        corner_ids = [
            pid for pid in range(n)
            if pieces[pid].is_corner or (
                pieces[pid].sides
                and sum(s.cls == "flat" for s in pieces[pid].sides) >= 2
            )
        ]
        candidates = corner_ids if corner_ids else list(range(n))
        seeds: list[AssemblyState] = []
        for pid in candidates:
            for rot in range(4):
                cost = self._seed_cost(pieces[pid], rot, rows, cols)
                if not np.isfinite(cost):
                    continue
                grid = [[None] * cols for _ in range(rows)]
                grid[0][0] = Placement(piece_id=pid, row=0, col=0, rot=rot)
                seeds.append(AssemblyState(grid=grid, used={pid}, total_dissim=cost))
        if not seeds:
            pid, rot = candidates[0], 0
            grid = [[None] * cols for _ in range(rows)]
            grid[0][0] = Placement(piece_id=pid, row=0, col=0, rot=rot)
            seeds.append(AssemblyState(grid=grid, used={pid}, total_dissim=0.0))
        return seeds

    @staticmethod
    def _better(a: AssemblyState, b: AssemblyState) -> bool:
        if len(a.used) != len(b.used):
            return len(a.used) > len(b.used)
        return a.total_dissim < b.total_dissim

    def _pick_best(self, states: list[AssemblyState]) -> AssemblyState:
        best = states[0]
        for s in states[1:]:
            if self._better(s, best):
                best = s
        return best

    def _expand_frontier(
        self,
        state: AssemblyState,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        rows: int,
        cols: int,
    ) -> list[AssemblyState]:
        frontier = self._frontier_cells(state, rows, cols)
        if not frontier:
            return []
        moves: list[AssemblyState] = []
        for r, c in frontier:
            moves.extend(
                self._candidates_for_cell(state, pieces, tensor, r, c, rows, cols)
            )
        if not moves:
            return []
        moves.sort(key=lambda s: s.total_dissim)
        return moves[: max(self.beam_k, 1)]

    @staticmethod
    def _frontier_cells(
        state: AssemblyState, rows: int, cols: int
    ) -> list[tuple[int, int]]:
        empty: list[tuple[int, int]] = []
        adjacent: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if state.grid[r][c] is not None:
                    continue
                empty.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and state.grid[nr][nc] is not None:
                        adjacent.append((r, c))
                        break
        return adjacent if adjacent else empty

    @staticmethod
    def _side(piece: Piece, rot: int, board_dir: int) -> str | None:
        if not piece.sides:
            return None
        return piece.sides[(board_dir - rot) % 4].cls

    def _seed_cost(self, piece: Piece, rot: int, rows: int, cols: int) -> float:
        """Lower is better for the top-left cell: flats should face north and west."""
        cost = 0.0
        north = self._side(piece, rot, 0)
        west = self._side(piece, rot, 3)
        east = self._side(piece, rot, 1)
        south = self._side(piece, rot, 2)
        if north != "flat":
            cost += BORDER_PENALTY
        if west != "flat":
            cost += BORDER_PENALTY
        if cols > 1 and east == "flat":
            cost += INTERIOR_FLAT_PENALTY
        if rows > 1 and south == "flat":
            cost += INTERIOR_FLAT_PENALTY
        if piece.is_corner:
            cost -= 1.0e3
        return cost

    def _candidates_for_cell(
        self,
        state: AssemblyState,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        r: int,
        c: int,
        rows: int,
        cols: int,
    ) -> list[AssemblyState]:
        results: list[AssemblyState] = []
        n = len(pieces)

        for pid in range(n):
            if pid in state.used:
                continue
            for rot in range(4):
                cost = self._placement_cost(state, pieces, tensor, pid, rot, r, c, rows, cols)
                if not np.isfinite(cost) or cost >= ILLEGAL_COST:
                    continue
                new_grid = [row[:] for row in state.grid]
                new_grid[r][c] = Placement(piece_id=pid, row=r, col=c, rot=rot)
                results.append(AssemblyState(
                    grid=new_grid,
                    used=state.used | {pid},
                    total_dissim=state.total_dissim + cost,
                ))

        if not results:
            return []
        results.sort(key=lambda s: (s.total_dissim, min(s.used)))
        return results[: max(self.beam_k, 1)]

    def _placement_cost(
        self,
        state: AssemblyState,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        pid: int,
        rot: int,
        r: int,
        c: int,
        rows: int,
        cols: int,
    ) -> float:
        piece = pieces[pid]
        total = 0.0
        adjacencies = [
            (0, -1, 0, 2),
            (1, 0, 1, 3),
            (2, 1, 0, 0),
            (3, 0, -1, 1),
        ]
        for my_dir, dr, dc, nbr_dir in adjacencies:
            nr, nc = r + dr, c + dc
            outside = nr < 0 or nc < 0 or nr >= rows or nc >= cols
            my_cls = self._side(piece, rot, my_dir)

            if outside:
                if my_cls != "flat":
                    total += BORDER_PENALTY
                continue

            nbr_placement = state.grid[nr][nc]
            if nbr_placement is None:
                if my_cls == "flat":
                    total += INTERIOR_FLAT_PENALTY
                continue

            nid = nbr_placement.piece_id
            nrot = nbr_placement.rot
            if not piece.sides or not pieces[nid].sides:
                total += ILLEGAL_COST
                continue
            my_side_idx = (my_dir - rot) % 4
            nbr_side_idx = (nbr_dir - nrot) % 4
            total += tensor.pair(pid, my_side_idx, nid, nbr_side_idx)
        return total


class CanvasReconstructor(ImageReconstructor):
    """Paste rotated piece crops onto a blank canvas to visualise the assembly."""

    def reconstruct(self, puzzle: Puzzle, state: AssemblyState) -> np.ndarray:
        by_id = {p.id: p for p in puzzle.pieces}
        by_index = {i: p for i, p in enumerate(puzzle.pieces)}
        rows = len(state.grid)
        cols = len(state.grid[0]) if rows > 0 else 0
        if not puzzle.pieces:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        # Median extent so one oversized blob cannot stretch every cell.
        extents = [max(p.image.shape[0], p.image.shape[1]) for p in puzzle.pieces]
        cell = max(int(np.median(extents)), 1)
        cell_h = cell_w = cell
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                placement = state.grid[r][c]
                if placement is None:
                    continue
                piece = by_id.get(placement.piece_id, by_index.get(placement.piece_id))
                if piece is None:
                    continue
                img = piece.image.copy()
                mask = piece.mask.copy()
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                # Placement.rot is clockwise; np.rot90(k) is counter-clockwise.
                if placement.rot % 4:
                    k = (-placement.rot) % 4
                    img = np.rot90(img, k=k)
                    mask = np.rot90(mask, k=k)

                img, mask = _fit_in_cell(img, mask, cell_h, cell_w)
                ph, pw = img.shape[:2]
                cell_y0, cell_x0 = r * cell_h, c * cell_w
                y0 = cell_y0 + (cell_h - ph) // 2
                x0 = cell_x0 + (cell_w - pw) // 2
                y1 = min(y0 + ph, cell_y0 + cell_h)
                x1 = min(x0 + pw, cell_x0 + cell_w)
                y0c, x0c = max(y0, cell_y0), max(x0, cell_x0)
                sy, sx = y0c - y0, x0c - x0
                sh, sw = y1 - y0c, x1 - x0c
                if sh <= 0 or sw <= 0:
                    continue
                patch = np.clip(img[sy : sy + sh, sx : sx + sw], 0, 255).astype(np.uint8)
                m = mask[sy : sy + sh, sx : sx + sw] > 0
                dest = canvas[y0c:y1, x0c:x1]
                dest[m] = patch[m]
                canvas[y0c:y1, x0c:x1] = dest

        return canvas


def _fit_in_cell(
    img: np.ndarray, mask: np.ndarray, cell_h: int, cell_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour shrink so a crop cannot spill into neighbouring cells."""
    ph, pw = img.shape[:2]
    if ph <= cell_h and pw <= cell_w:
        return img, mask
    scale = min(cell_h / ph, cell_w / pw)
    nh, nw = max(1, int(ph * scale)), max(1, int(pw * scale))
    ys = (np.arange(nh) * (ph / nh)).astype(int)
    xs = (np.arange(nw) * (pw / nw)).astype(int)
    return img[np.ix_(ys, xs)], mask[np.ix_(ys, xs)]
