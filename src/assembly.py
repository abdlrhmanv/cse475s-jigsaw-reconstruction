"""Greedy best-first assembly and canvas reconstruction (Phase 4)."""

from __future__ import annotations

import numpy as np

from src.core.protocols import Assembler, ImageReconstructor
from src.core.types import AssemblyState, CompatibilityTensor, Piece, Placement, Puzzle


class GreedyBestFirstAssembler(Assembler):
    """Place pieces greedily, trying each rotation (0-3).

    Beam search with beam_k candidates prevents a single bad early choice
    from derailing the entire assembly.
    """

    def __init__(self, beam_k: int = 3) -> None:
        self.beam_k = beam_k

    def assemble(
        self,
        pieces: list[Piece],
        tensor: CompatibilityTensor,
        rows: int,
        cols: int,
    ) -> AssemblyState:
        n = len(pieces)
        if n == 0:
            return AssemblyState(grid=[[None] * cols for _ in range(rows)], used=set())

        # --- seed: try each piece at (0,0) with each rotation ---------
        beams: list[AssemblyState] = []
        for pid in range(n):
            for rot in range(4):
                grid = [[None] * cols for _ in range(rows)]
                grid[0][0] = Placement(piece_id=pid, row=0, col=0, rot=rot)
                beams.append(AssemblyState(grid=grid, used={pid}, total_dissim=0.0))

        # Keep only top-k by total dissimilarity
        beams.sort(key=lambda s: s.total_dissim)
        beams = beams[: self.beam_k]

        # --- expand cell by cell in raster order ---------------------
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        for r, c in cells[1:]:  # skip (0,0) already placed
            next_beams: list[AssemblyState] = []
            for state in beams:
                candidates = self._candidates_for_cell(
                    state, pieces, tensor, r, c, rows, cols
                )
                for cand_state in candidates:
                    next_beams.append(cand_state)

            if not next_beams:
                break  # no valid placements; return best partial

            next_beams.sort(key=lambda s: s.total_dissim)
            beams = next_beams[: self.beam_k]

        # Return best beam
        beams.sort(key=lambda s: s.total_dissim)
        return beams[0]

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
        """Generate candidate states by placing each unused piece (all rotations) at (r, c)."""
        results: list[AssemblyState] = []
        n = len(pieces)

        for pid in range(n):
            if pid in state.used:
                continue
            for rot in range(4):
                cost = self._placement_cost(state, pieces, tensor, pid, rot, r, c, rows, cols)
                if cost == np.inf:
                    continue
                # Deep-copy grid
                new_grid = [row[:] for row in state.grid]
                new_grid[r][c] = Placement(piece_id=pid, row=r, col=c, rot=rot)
                new_used = state.used | {pid}
                results.append(AssemblyState(
                    grid=new_grid, used=new_used,
                    total_dissim=state.total_dissim + cost,
                ))

        # If nothing fits, allow skipping (leave cell empty)
        if not results:
            results.append(state)
        return results

    @staticmethod
    def _placement_cost(
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
        """Sum of dissimilarities between (pid, rot) at (r,c) and already-placed neighbours."""
        piece = pieces[pid]
        total = 0.0
        # (board_dir, dr, dc, neighbour_board_dir)
        adjacencies = [
            (0, -1, 0, 2),  # north neighbour is above; my north side faces their south
            (3, 0, -1, 1),  # west neighbour is left; my west faces their east
        ]
        for my_dir, dr, dc, nbr_dir in adjacencies:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                continue
            nbr_placement = state.grid[nr][nc]
            if nbr_placement is None:
                continue
            nid = nbr_placement.piece_id
            nrot = nbr_placement.rot
            if not piece.sides or not pieces[nid].sides:
                return np.inf
            my_side_idx = (my_dir - rot) % 4
            nbr_side_idx = (nbr_dir - nrot) % 4
            d = tensor.pair(pid, my_side_idx, nid, nbr_side_idx)
            if d == np.inf:
                return np.inf
            total += d
        return total


class CanvasReconstructor(ImageReconstructor):
    """Paste rotated piece crops onto a blank canvas to visualise the assembly."""

    def reconstruct(self, puzzle: Puzzle, state: AssemblyState) -> np.ndarray:
        pieces_by_id = {p.id: p for p in puzzle.pieces}
        rows = len(state.grid)
        cols = len(state.grid[0]) if rows > 0 else 0

        # Estimate cell size from largest piece
        max_h = max((p.image.shape[0] for p in puzzle.pieces), default=1)
        max_w = max((p.image.shape[1] for p in puzzle.pieces), default=1)
        cell_h, cell_w = max_h, max_w

        ch = 3 if puzzle.pieces and puzzle.pieces[0].image.ndim == 3 else 0
        if ch:
            canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((rows * cell_h, cols * cell_w), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                placement = state.grid[r][c]
                if placement is None:
                    continue
                piece = pieces_by_id.get(placement.piece_id)
                if piece is None:
                    continue
                img = piece.image.copy()
                # Apply rotation (k * 90° counter-clockwise = numpy rot90)
                if placement.rot > 0:
                    img = np.rot90(img, k=placement.rot)

                ph, pw = img.shape[:2]
                # Centre in cell
                y0 = r * cell_h + (cell_h - ph) // 2
                x0 = c * cell_w + (cell_w - pw) // 2
                # Clamp
                y1 = min(y0 + ph, canvas.shape[0])
                x1 = min(x0 + pw, canvas.shape[1])
                y0 = max(y0, 0)
                x0 = max(x0, 0)
                sh = y1 - y0
                sw = x1 - x0
                if sh <= 0 or sw <= 0:
                    continue
                canvas[y0:y1, x0:x1] = np.clip(img[:sh, :sw], 0, 255).astype(np.uint8)

        return canvas
