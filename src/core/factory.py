"""Build a `ReconstructionPipeline` from YAML.

This is the only module allowed to import concrete operators (DIP). `main.py`
and tests must not construct Sobel/Canny/GNN directly. `--method` swaps the
matcher only; piece extraction stays shared so M1 vs M2 is a fair comparison.
"""

from __future__ import annotations

from src.assembly import CanvasReconstructor, GreedyBestFirstAssembler
from src.contour_extraction import PieceExtractorImpl
from src.core.convolution import ConvolutionEngine
from src.core.viz import StageVisualizer
from src.edge_matching import ClassicalCompatibilityMatcher
from src.enhancement import ContrastStretcher, FilterChain, GaussianFilter, MedianFilter
from src.evaluation import ReconstructionEvaluator
from src.piece_description import PieceDescriptorImpl
from src.pipeline import ReconstructionPipeline
from src.segmentation import ConnectedComponentLabeler
from src.thresholding import AdaptiveThreshold, GlobalThreshold, OtsuThreshold


class PipelineFactory:
    @staticmethod
    def from_config(config: dict) -> ReconstructionPipeline:
        method = config.get("method", "classical")
        engine = ConvolutionEngine()
        filt = config.get("filter", {})
        # Median first (impulse noise), then Gaussian, then stretch. Colour copy
        # for strips is kept separately on Puzzle.raw_colour, not in this chain.
        chain = FilterChain(
            [
                MedianFilter(k=int(filt.get("median_k", 3))),
                GaussianFilter(
                    k=int(filt.get("gaussian_k", 5)),
                    sigma=filt.get("gaussian_sigma", 1.0),
                    engine=engine,
                ),
                ContrastStretcher(),
            ]
        )

        kind = str(config.get("thresholder", "otsu")).lower()
        if kind == "otsu":
            thresholder = OtsuThreshold()
        elif kind == "global":
            thresholder = GlobalThreshold(t=float(config.get("global_t", 128)))
        elif kind == "adaptive":
            thresholder = AdaptiveThreshold(w=15, c=2.0, kind="gaussian")
        else:
            raise ValueError(f"unknown thresholder: {kind}")

        matching = config.get("matching", {})
        if method == "classical":
            matcher = ClassicalCompatibilityMatcher(
                ws=float(matching.get("ws", 0.3)),
                wc=float(matching.get("wc", 0.7)),
            )
        elif method in {"siamese", "gnn"}:
            raise NotImplementedError(f"Milestone 2 matcher: {method}")
        else:
            raise ValueError(f"unknown method: {method}")

        beam_k = int(config.get("assembly", {}).get("beam_k", 3))
        return ReconstructionPipeline(
            filters=chain,
            thresholder=thresholder,
            labeler=ConnectedComponentLabeler(),
            extractor=PieceExtractorImpl(),
            descriptor=PieceDescriptorImpl(),
            matcher=matcher,
            assembler=GreedyBestFirstAssembler(beam_k=beam_k),
            reconstructor=CanvasReconstructor(),
            evaluator=ReconstructionEvaluator(),
            visualizer=StageVisualizer(),
        )
