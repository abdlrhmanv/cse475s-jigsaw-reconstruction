"""Build a `ReconstructionPipeline` from YAML.

This is the only module allowed to import concrete operators (DIP). `main.py`
and tests must not construct Sobel/Canny/GNN directly. `--method` swaps the
matcher only; piece extraction stays shared so M1 vs M2 is a fair comparison.
Siamese is the ML matcher; GNN is a weak extra (real val_ap ~0.26).
"""

from __future__ import annotations

from src.assembly import CanvasReconstructor, GreedyBestFirstAssembler
from src.contour_extraction import PieceExtractorImpl, YoloBoxExtractor
from src.core.convolution import ConvolutionEngine
from src.core.viz import StageVisualizer
from src.edge_detection import CannyEdgeDetector
from src.edge_matching import ClassicalCompatibilityMatcher
from src.enhancement import (
    BinaryCloser,
    BinaryHoleFiller,
    ContrastStretcher,
    FilterChain,
    GaussianFilter,
    MedianFilter,
)
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
                ws=float(matching.get("ws", 0.85)),
                wc=float(matching.get("wc", 0.15)),
            )
        elif method == "siamese":
            from src.ml.siamese_matcher import SiameseCompatibilityMatcher
            matcher = SiameseCompatibilityMatcher(
                weights=matching.get("weights", "checkpoints/siamese.pt"),
            )
        elif method == "gnn":
            from src.ml.gnn_matcher import GNNCompatibilityMatcher
            matcher = GNNCompatibilityMatcher(
                weights=matching.get("weights", "checkpoints/gnn.pt"),
                top_k=int(matching.get("top_k", 0)),
            )
        else:
            raise ValueError(f"unknown method: {method}")

        beam_k = int(config.get("assembly", {}).get("beam_k", 3))
        seg = config.get("segmentation", {})
        keep_n = seg.get("keep_n")
        return ReconstructionPipeline(
            filters=chain,
            thresholder=thresholder,
            hole_filler=FilterChain(
                [BinaryCloser(k=int(seg.get("close_k", 7))), BinaryHoleFiller()]
            )
            if seg.get("fill_holes", True)
            else FilterChain([]),
            labeler=ConnectedComponentLabeler(
                min_area=int(seg.get("min_area", 2000)),
                min_area_frac=float(seg.get("min_area_frac", 0.001)),
                max_aspect=float(seg.get("max_aspect", 5.0)),
                min_solidity=float(seg.get("min_solidity", 0.45)),
                min_rel_area=float(seg.get("min_rel_area", 0.40)),
                max_area_frac=float(seg.get("max_area_frac", 0.25)),
                keep_n=int(keep_n) if keep_n is not None else None,
            ),
            extractor=PieceExtractorImpl(),
            box_extractor=YoloBoxExtractor(),
            descriptor=PieceDescriptorImpl(),
            matcher=matcher,
            assembler=GreedyBestFirstAssembler(beam_k=beam_k),
            reconstructor=CanvasReconstructor(),
            evaluator=ReconstructionEvaluator(),
            visualizer=StageVisualizer(),
            edge_detector=CannyEdgeDetector(),
        )
