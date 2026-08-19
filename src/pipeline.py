"""End-to-end reconstruction. Constructor takes protocols so M2 is a matcher swap (DIP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.io import ImageStore
from src.core.protocols import (
    Assembler,
    CompatibilityMatcher,
    EdgeDetector,
    Evaluator,
    ImageFilter,
    ImageReconstructor,
    Labeler,
    PieceDescriptor,
    PieceExtractor,
    Thresholder,
)
from src.contour_extraction import YoloBoxExtractor, count_yolo_classes, gt_label_path
from src.core.grid import count_for_grid, infer_board_shape
from src.core.types import Puzzle, ReconstructionResult
from src.core.viz import StageVisualizer
from src.core.image_utils import _ensure_gray
from src.enhancement import FilterChain
from src.evaluation import canonical_Q, geometry_quality, identity_neighbour_quality, identity_orientation_quality
from src.segmentation import orient_foreground


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
        hole_filler: ImageFilter | None = None,
        box_extractor: YoloBoxExtractor | None = None,
        edge_detector: EdgeDetector | None = None,
    ) -> None:
        self.filters = filters
        self.thresholder = thresholder
        self.hole_filler = hole_filler
        self.labeler = labeler
        self.extractor = extractor
        self.box_extractor = box_extractor
        self.edge_detector = edge_detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.assembler = assembler
        self.reconstructor = reconstructor
        self.evaluator = evaluator
        self.visualizer = visualizer or StageVisualizer()
        self._io = ImageStore()

    def run(self, image_path: str, config: dict) -> ReconstructionResult:
        out_dir = Path(config.get("output_dir", "results"))
        enhanced_dir = out_dir / "enhanced_images"
        mask_dir = out_dir / "masks"
        contour_dir = out_dir / "contours"
        edge_dir = out_dir / "edge_visualisations"
        recon_dir = out_dir / "reconstructed_images"
        eval_dir = out_dir / "evaluation_results"
        ml_dir = out_dir / "ml"
        rows = int(config.get("rows", 3))
        cols = int(config.get("cols", 3))

        # 1 — Load
        raw = self._io.load(image_path).astype(np.float64)
        raw_colour = raw.copy()

        # 2 — Enhancement filter chain
        enhanced = self.filters.apply(raw)
        self.visualizer.save_side_by_side(
            enhanced_dir / "01_enhanced.png",
            [np.clip(raw, 0, 255), np.clip(enhanced, 0, 255)],
            titles=["Original", "Enhanced"],
        )
        self.visualizer.save_image(enhanced_dir / "enhanced.png", enhanced)
        self.visualizer.save_hist(enhanced_dir / "histogram.png", enhanced, title="Enhanced")

        # 3 — Threshold, polarity so pieces are white, optional hole fill
        gray = _ensure_gray(enhanced)
        binary = self.thresholder.threshold(gray)
        binary = orient_foreground(gray, binary)
        if self.hole_filler is not None:
            binary = self.hole_filler.apply(binary)
        self.visualizer.save_side_by_side(
            mask_dir / "02_binary.png",
            [gray, binary],
            titles=["Grayscale", "Binary"],
        )
        self.visualizer.save_image(mask_dir / "binary.png", binary)

        if self.edge_detector is not None:
            edge = self.edge_detector.detect(_downscale_for_viz(gray))
            mag = edge.magnitude
            mag_vis = mag * (255.0 / mag.max()) if mag.max() > 0 else mag
            self.visualizer.save_side_by_side(
                edge_dir / "canny.png",
                [edge.extras.get("smoothed", mag_vis), mag_vis, edge.edges],
                titles=["Smoothed", "Gradient magnitude", "Canny"],
            )
            self.visualizer.save_image(edge_dir / "magnitude.png", mag_vis)
            self.visualizer.save_image(edge_dir / "edges.png", edge.edges)

        # 4–5 — Piece extraction: CCL is the spec path; YOLO boxes are opt-in
        use_gt = bool(config.get("segmentation", {}).get("use_gt_boxes", False))
        identity_labels = gt_label_path(image_path)
        labelled = count_yolo_classes(identity_labels) if identity_labels is not None else 0
        expected_n = labelled if labelled > 0 else None
        label_file = identity_labels if use_gt else None
        source = "ccl"
        if label_file is not None and self.box_extractor is not None:
            pieces = self.box_extractor.extract(raw_colour, label_file)
            labels = self.box_extractor.label_map(raw_colour.shape, label_file)
            source = "yolo"
        else:
            keep_n_cfg = config.get("segmentation", {}).get("keep_n")
            n_keep = int(keep_n_cfg) if keep_n_cfg is not None else expected_n
            if n_keep is not None and hasattr(self.labeler, "keep_n"):
                prev_keep = self.labeler.keep_n
                self.labeler.keep_n = n_keep
                try:
                    labels = self.labeler.label(binary)
                finally:
                    self.labeler.keep_n = prev_keep
            else:
                labels = self.labeler.label(binary)
            pieces = self.extractor.extract(raw_colour, labels)

        label_vis = (labels.astype(np.float64) / max(float(labels.max()), 1.0)) * 255
        self.visualizer.save_side_by_side(
            mask_dir / "03_labels.png",
            [binary, label_vis],
            titles=["Binary", f"Labels ({source})"],
        )
        self.visualizer.save_image(mask_dir / "labels.png", label_vis)
        overlay = self.visualizer.overlay_contours(raw_colour, pieces)
        self.visualizer.save_image(contour_dir / "contours.png", overlay)

        if source != "yolo":
            if identity_labels is not None and self.box_extractor is not None:
                from src.ml.pose_gt import attach_class_ids

                boxed = self.box_extractor.extract(raw_colour, identity_labels)
                attach_class_ids(pieces, boxed)

        # 6 — Piece description (corners, sides, colour strips)
        for i in range(len(pieces)):
            pieces[i] = self.descriptor.describe(pieces[i])

        # 7 — Compatibility tensor
        tensor = self.matcher.build(pieces)

        if bool(config.get("infer_grid", True)):
            rows, cols = infer_board_shape(
                count_for_grid(len(pieces), expected_n),
                prefer_rows=int(config.get("rows", rows)),
                prefer_cols=int(config.get("cols", cols)),
            )

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
        self._io.save(recon_dir / "04_reconstructed.png", canvas)
        self._io.save(out_dir / "04_reconstructed.png", canvas)

        metrics = self._evaluate(state, pieces, rows, cols, image_path, source, expected_n, config)
        self.visualizer.save_json(eval_dir / "last.json", metrics)
        matching = config.get("matching", {})
        self.visualizer.save_json(
            ml_dir / "last_run.json",
            {
                "method": config.get("method", "classical"),
                "weights": matching.get("weights"),
                "checkpoints_dir": "checkpoints",
            },
        )
        print(
            f"extracted {len(pieces)} pieces | grid {rows}×{cols} | "
            f"placed {metrics.get('n_placed', '?')} | Q={metrics.get('Q', 0):.3f} | source {source}"
        )

        return ReconstructionResult(
            state=state,
            image=canvas,
            quality=metrics.get("Q"),
            metrics=metrics,
        )


    def _evaluate(self, state, pieces, rows, cols, image_path, source, expected_n, config) -> dict:
        """Compute all quality metrics for a reconstruction."""
        placed = sum(1 for row in state.grid for cell in row if cell is not None)
        quality = geometry_quality(state, pieces, rows, cols)
        if self.evaluator is not None:
            quality.setdefault("evaluator", type(self.evaluator).__name__)
        metrics: dict = {
            "input": str(image_path),
            "method": config.get("method", "classical"),
            "extraction": source,
            "n_pieces": len(pieces),
            "n_labelled": expected_n,
            "n_placed": placed,
            "rows": rows,
            "cols": cols,
            **quality,
            "geometry_Q": quality.get("Q"),
            "pose_gt": False,
            "orientation_accuracy": None,
        }
        ident = identity_neighbour_quality(state, pieces)
        ori = identity_orientation_quality(state, pieces)
        if ori:
            metrics.update(ori)
        if ident:
            metrics.update(ident)
            metrics["neighbour_accuracy"] = ident["identity_neighbour_accuracy"]
            metrics["edge_accuracy"] = ident["identity_neighbour_accuracy"]
            metrics["Q"] = canonical_Q(
                metrics.get("position_accuracy"),
                metrics.get("orientation_accuracy"),
                ident["identity_neighbour_accuracy"],
            )
            metrics["quality_kind"] = "identity_neighbour"
            metrics["complete_reconstruction"] = (
                1.0
                if ident["identity_neighbour_accuracy"] >= 1.0 - 1e-12
                and placed == len(pieces)
                and placed == rows * cols
                else 0.0
            )

        from src.ml.pose_gt import pose_from_pieces

        gt, gt_rows, gt_cols = pose_from_pieces(pieces)
        compact = (
            gt is not None
            and (gt_rows, gt_cols) == (rows, cols)
            and all(info.get("pose_note") == "canonical_compact" for info in gt.pieces.values())
        )
        if compact and self.evaluator is not None and gt is not None:
            pose = self.evaluator.evaluate(state, gt)
            id_edge = ident.get("identity_neighbour_accuracy", pose.get("edge_accuracy") or 0.0)
            metrics["position_accuracy"] = pose["position_accuracy"]
            if ori:
                metrics.update(ori)
            else:
                metrics["orientation_accuracy"] = None
            metrics["neighbour_accuracy"] = id_edge
            metrics["edge_accuracy"] = id_edge
            metrics["complete_reconstruction"] = 1.0 if pose["position_accuracy"] >= 1.0 - 1e-12 else 0.0
            metrics["Q"] = canonical_Q(
                pose["position_accuracy"],
                metrics.get("orientation_accuracy"),
                id_edge,
            )
            metrics["geometry_Q"] = quality.get("Q")
            metrics["quality_kind"] = "identity_pose"
            metrics["pose_gt"] = True
            metrics["pose_grid"] = [gt_rows, gt_cols]
            metrics["pose_note"] = "compact_cells; ori from identity+flats on border pieces"
        return metrics


def _downscale_for_viz(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Nearest-neighbour shrink so Canny viz stays cheap on 1080p photos."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = (np.arange(nh) * (h / nh)).astype(int)
    xs = (np.arange(nw) * (w / nw)).astype(int)
    return image[np.ix_(ys, xs)]
