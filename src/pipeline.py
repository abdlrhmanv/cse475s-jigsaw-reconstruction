"""End-to-end reconstruction. Constructor takes protocols so M2 is a matcher swap (DIP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.io import ImageStore
from src.core.protocols import (
    Assembler,
    CompatibilityMatcher,
    Evaluator,
    ImageReconstructor,
    Labeler,
    PieceDescriptor,
    PieceExtractor,
    Thresholder,
)
from src.core.types import Puzzle, ReconstructionResult
from src.core.viz import StageVisualizer
from src.enhancement import FilterChain, _ensure_gray


class ReconstructionPipeline:
    def __init__(
        self,
        filters: FilterChain,
        thresholder: Thresholder,
        labeler: Labeler,
        extractor: PieceExtractor,
        descriptor: PieceDescriptor,
        matcher: CompatibilityMatcher,
        assembler: Assembler,
        reconstructor: ImageReconstructor,
        evaluator: Evaluator | None = None,
        visualizer: StageVisualizer | None = None,
    ) -> None:
        self.filters = filters
        self.thresholder = thresholder
        self.labeler = labeler
        self.extractor = extractor
        self.descriptor = descriptor
        self.matcher = matcher
        self.assembler = assembler
        self.reconstructor = reconstructor
        self.evaluator = evaluator
        self.visualizer = visualizer or StageVisualizer()
        self._io = ImageStore()

    def run(self, image_path: str, config: dict) -> ReconstructionResult:
        out_dir = Path(config.get("output_dir", "results"))
        rows = int(config.get("rows", 3))
        cols = int(config.get("cols", 3))

        # 1 — Load
        raw = self._io.load(image_path).astype(np.float64)
        raw_colour = raw.copy()

        # 2 — Enhancement filter chain
        enhanced = self.filters.apply(raw)
        self.visualizer.save_side_by_side(
            out_dir / "01_enhanced.png",
            [np.clip(raw, 0, 255), np.clip(enhanced, 0, 255)],
            titles=["Original", "Enhanced"],
        )

        # 3 — Threshold
        gray = _ensure_gray(enhanced)
        binary = self.thresholder.threshold(gray)
        self.visualizer.save_side_by_side(
            out_dir / "02_binary.png",
            [gray, binary],
            titles=["Grayscale", "Binary"],
        )

        # 4 — Connected component labelling
        labels = self.labeler.label(binary)
        label_vis = (labels.astype(np.float64) / max(labels.max(), 1)) * 255
        self.visualizer.save_side_by_side(
            out_dir / "03_labels.png",
            [binary, label_vis],
            titles=["Binary", "Labels"],
        )

        # 5 — Piece extraction
        pieces = self.extractor.extract(raw_colour, labels)

        # 6 — Piece description (corners, sides, colour strips)
        for i in range(len(pieces)):
            pieces[i] = self.descriptor.describe(pieces[i])

        # 7 — Compatibility tensor
        tensor = self.matcher.build(pieces)

        # 8 — Assembly
        puzzle = Puzzle(
            image=raw,
            pieces=pieces,
            rows=rows,
            cols=cols,
            raw_colour=raw_colour,
        )
        state = self.assembler.assemble(pieces, tensor, rows, cols)

        # 9 — Canvas reconstruction
        canvas = self.reconstructor.reconstruct(puzzle, state)
        self._io.save(out_dir / "04_reconstructed.png", canvas)

        return ReconstructionResult(
            state=state,
            image=canvas,
        )
