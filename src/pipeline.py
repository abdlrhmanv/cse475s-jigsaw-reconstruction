"""End-to-end reconstruction. Constructor takes protocols so M2 is a matcher swap (DIP)."""

from __future__ import annotations

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
from src.core.types import ReconstructionResult
from src.core.viz import StageVisualizer
from src.enhancement import FilterChain


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
        self.visualizer = visualizer

    def run(self, image_path: str, config: dict) -> ReconstructionResult:
        raise NotImplementedError(
            "Later phases: enhance → threshold → CCL → describe → match → assemble."
        )
