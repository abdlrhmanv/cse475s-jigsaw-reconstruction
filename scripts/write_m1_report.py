"""Generate report/milestone_1_report.pdf from the implemented operators."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_detection import CannyEdgeDetector, SobelOperator
from src.enhancement import ContrastStretcher, GaussianFilter, MedianFilter
from src.thresholding import OtsuThreshold


OUT_PATH = PROJECT_ROOT / "report" / "milestone_1_report.pdf"
A4 = (8.27, 11.69)


def _wrap(text: str, width: int = 98) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def text_page(pdf: PdfPages, title: str, body: str, footer: str | None = None) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.955, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.08, 0.91, _wrap(body), fontsize=9.2, va="top", family="DejaVu Sans", linespacing=1.35)
    if footer:
        fig.text(0.08, 0.035, footer, fontsize=8, color="0.35")
    fig.text(0.92, 0.035, "CSE480 Milestone 1", fontsize=8, color="0.35", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def math_page(pdf: PdfPages, title: str, intro: str, formulas: list[str], notes: str) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.08, 0.955, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.08, 0.91, _wrap(intro, 95), fontsize=9.2, va="top", linespacing=1.35)
    y = 0.72
    for formula in formulas:
        fig.text(0.12, y, formula, fontsize=12, va="top")
        y -= 0.07
    fig.text(0.08, y - 0.02, _wrap(notes, 95), fontsize=9.2, va="top", linespacing=1.35)
    fig.text(0.92, 0.035, "CSE480 Milestone 1", fontsize=8, color="0.35", ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def _tab_piece(size: int = 160) -> np.ndarray:
    img = np.full((size, size), 30.0)
    img[30:130, 30:130] = 210.0
    img[20:30, 70:90] = 210.0  # north tab
    img[70:90, 130:145] = 30.0  # east blank (notch)
    rng = np.random.default_rng(0)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255)


def figure_operators(pdf: PdfPages) -> None:
    piece = _tab_piece()
    med = MedianFilter(k=3).apply(piece)
    gau = GaussianFilter(k=5, sigma=1.0).apply(med)
    stretched = ContrastStretcher().apply(gau)
    binary = OtsuThreshold().threshold(stretched)
    canny = CannyEdgeDetector()
    edge = canny.detect(gau)
    gx, gy = SobelOperator().gradients(gau)
    mag = np.hypot(gx, gy)

    fig, axes = plt.subplots(3, 3, figsize=A4)
    fig.suptitle("Enhancement, thresholding, and Canny stages (synthetic piece)", fontsize=12, fontweight="bold")
    panels = [
        (piece, "Original"),
        (med, "Median k=3"),
        (gau, "Gaussian k=5, σ=1"),
        (stretched, "Percentile stretch 1–99"),
        (binary, "Otsu binary"),
        (mag, "Sobel magnitude"),
        (edge.extras["nms"], "NMS"),
        (edge.extras["strong"], "Strong (T_high)"),
        (edge.edges, "Hysteresis output"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)

    kernel = GaussianFilter(k=5, sigma=1.0).kernel()
    fig, ax = plt.subplots(figsize=A4)
    fig.suptitle("Gaussian smoothing kernel used by Canny (k=5, σ=1)", fontsize=12, fontweight="bold")
    im = ax.imshow(kernel, cmap="viridis")
    ax.set_title(r"$G(x,y)=\frac{1}{Z}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$,  $\sigma=1$,  $k=5$")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{kernel[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)

    t = np.linspace(0, 1, 64)
    tab = 8 * np.sin(np.pi * t) ** 2
    blank = -tab[::-1]
    fig, axes = plt.subplots(2, 1, figsize=A4)
    fig.suptitle("Complementary shape profiles (tab fills blank)", fontsize=12, fontweight="bold")
    axes[0].plot(tab, label="p (tab)")
    axes[0].plot(blank, label="q (blank)")
    axes[0].legend()
    axes[0].set_title("Raw signed offsets")
    axes[1].plot(tab + blank[::-1], color="C2")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title(r"$p + \mathrm{flip}(q)$  (zero for a perfect mate)")
    for ax in axes:
        ax.set_xlabel("sample along side")
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def write_report(path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=A4)
        fig.text(0.5, 0.72, "CSE480 Machine Vision", ha="center", fontsize=14)
        fig.text(0.5, 0.66, "Milestone 1 Report", ha="center", fontsize=22, fontweight="bold")
        fig.text(0.5, 0.58, "Jigsaw Puzzle Reconstruction from Scrambled Photographs", ha="center", fontsize=12)
        fig.text(
            0.5,
            0.46,
            _wrap(
                "Abdlrhman Hisham Ismail, 2300343\n"
                "Ain Shams University, Faculty of Engineering, Mechatronics Engineering\n"
                "Instructor: Prof. Hossam El-Din Hassan Abd El Munim\n"
                "Summer 2026",
                70,
            ),
            ha="center",
            va="top",
            fontsize=11,
            linespacing=1.6,
        )
        fig.text(
            0.5,
            0.22,
            "Classical pipeline: enhancement, Canny, CCL, piece description,\n"
            "weighted shape+colour matching, greedy beam assembly.",
            ha="center",
            fontsize=10,
            color="0.25",
        )
        pdf.savefig(fig)
        plt.close(fig)

        text_page(
            pdf,
            "1. Introduction",
            "This project reconstructs a jigsaw from a photograph of scrambled pieces. "
            "Pieces may appear at unknown 90-degree rotations. The Milestone 1 system is a "
            "classical computer-vision pipeline: denoise and stretch the photo, threshold to a "
            "binary mask, label connected components, extract contours, describe each piece by "
            "four sides (tab / blank / flat plus a colour ribbon), score every legal side pair, "
            "and assemble a grid with a greedy beam search.\n\n"
            "Rotations are resolved in three layers. Continuous table tilt is removed by "
            "deskewing each crop to its min-area rectangle. The remaining 90-degree ambiguity "
            "is snapped from observed flats: corner pieces are rotated so flats face north and "
            "west in the crop; edge pieces so the single flat faces north. Assembly then "
            "searches r in {0,1,2,3} (clockwise 90-degree steps). The side that faces board "
            "direction D is piece side (D − r) mod 4, with D = 0,1,2,3 for N,E,S,W.\n\n"
            "Milestone 2 reuses the same extraction and assembler. Only the compatibility "
            "matcher is swapped (Siamese CNN or GNN). This report documents the classical "
            "operators, the exact matching formula, assembly tie / dead-end rules, and the "
            "limitations of the supplied labels.",
        )

        text_page(
            pdf,
            "2. Problem definition (I/O)",
            "Input: a colour photograph of mixed puzzle pieces on a table, JPEG/PNG, typically "
            "1080p. Optional YOLO label file with the same stem, format `class cx cy w h` "
            "(normalised), where class is a physical piece identity 1–35.\n\n"
            "Output: a reconstructed canvas of an R×C grid, a placement list "
            "(piece id, row, col, rot), intermediate visualisations (enhanced image, mask, "
            "labels, contours, Canny stages), and a JSON metrics file containing Q.\n\n"
            "Board size is inferred from the extracted piece count: 9 → 3×3, 16 → 4×4, "
            "35 → 7×5. Other counts use the closest factorisation, preferring the configured "
            "(7, 5) when the count already equals 35.\n\n"
            "Default extraction is connected-component labelling on the enhanced mask "
            "(the course spec). YOLO boxes are opt-in (`segmentation.use_gt_boxes: true`) and "
            "are otherwise used only to attach piece identities for pose evaluation and ML pair "
            "generation.",
        )

        text_page(
            pdf,
            "3. Dataset",
            "The dataset is a Roboflow YOLO export of a 35-piece (7×5) jigsaw photographed in "
            "scrambled layouts. Many images show a 9-piece subset; some show 16–21 pieces; a "
            "smaller set shows the full 35.\n\n"
            "Splits live under data/input/{train,val,test} with matching labels under "
            "data/ground_truth/{train,val,test}. Roboflow augmentations share a source prefix "
            "before `.rf.`; the resplit script keeps those variants in one split so train and "
            "test never share the same photograph.\n\n"
            "Important: YOLO classes are piece identities, not poses. The names list in "
            "data/data.yaml is not numeric order (['1','10',…,'9']). Mapping class index → "
            "printed number requires that list. There is no assembled-image ground truth and "
            "no rotation label for the photo. Canonical cells assume piece 1 sits at (0,0) "
            "row-major on the completed 7×5 board. When a photo’s pieces occupy a filled "
            "rectangle, that block is compacted (e.g. a 3×3 subset) for pose Q; scattered "
            "subsets keep geometry Q only.",
        )

        text_page(
            pdf,
            "4. Methodology overview",
            "Pipeline (shared by classical / Siamese / GNN):\n"
            "  1. Load RGB with Pillow (I/O only).\n"
            "  2. Enhancement: median 3×3 → Gaussian 5×5 σ=1 → percentile contrast stretch.\n"
            "  3. Grayscale luminance, Otsu threshold, polarity so pieces are white, "
            "morphological close + hole fill.\n"
            "  4. Connected-component labelling from scratch; drop dust by area / aspect / solidity.\n"
            "  5. Moore-neighbour contour on each blob; crop from the original colour image.\n"
            "  6. Corners: convex hull → Ramer–Douglas–Peucker → largest near-rectangular quad "
            "(PCA rectangle fallback). Four sides, signed profile, inward Lab ribbon.\n"
            "  7. Compatibility tensor D(i,si,j,sj). Illegal geometry is +∞.\n"
            "  8. Greedy beam assembly from a top-left corner seed (flats facing N and W).\n"
            "  9. Paste crops onto a canvas; write Q.\n\n"
            "All listed filters, Sobel/Prewitt/Canny, CCL, and contours are NumPy "
            "implementations. OpenCV is not used for those operators.",
        )

        text_page(
            pdf,
            "5. Enhancement operators",
            "Median (k=3). Non-linear order statistic. Each pixel is the median of its 3×3 "
            "window (reflect pad). Impulse noise (table specks, JPEG sparkle) is removed "
            "without the blur of a mean filter. It cannot be written as a kernel inner product, "
            "so the implementation is a justified nested loop (not OpenCV convolution). On this "
            "photo set that costs about 17 s per board; that is accepted for the course.\n\n"
            "Gaussian (k=5, σ=1). Separable in principle; implemented as a normalised 2-D kernel\n"
            "    G(x,y) = (1/Z) exp(−(x²+y²)/(2σ²)),   Z = Σ G.\n"
            "Used both in the enhancement chain and as Canny’s pre-smoothing stage.\n\n"
            "Contrast stretch. Let v_low and v_high be the 1st and 99th percentiles of the image.\n"
            "    I' = clip( 255 (I − v_low) / (v_high − v_low) , 0, 255 ).\n"
            "Percentiles ignore dust min/max so a few bright pixels cannot collapse the range.\n\n"
            "Histogram equalisation is implemented (CDF LUT) but not in the default chain: "
            "equalising the table photo over-amplifies wood grain and hurts Otsu.\n\n"
            "Unsharp mask I + α(I − Gσ ∗ I) is available for ablation; default α is unused.\n\n"
            "Colour strips for matching are taken from Puzzle.raw_colour (the unfiltered photo), "
            "so photometric matching does not see equalisation or blur.",
        )

        math_page(
            pdf,
            "6. Edge detection and Canny parameters",
            "Sobel kernels produce Gx, Gy through the same ConvolutionEngine as the Gaussian. "
            "Magnitude is hypot(Gx, Gy); orientation is atan2(Gy, Gx). Prewitt is a drop-in "
            "GradientOperator for ablation. Canny then uses the following fixed parameters.",
            [
                r"$G_\sigma$: Gaussian $k=5$, $\sigma=1$",
                r"$|\nabla I|=\sqrt{G_x^2+G_y^2}$,  $\theta=\mathrm{atan2}(G_y,G_x)$",
                r"NMS: 4 orientation bins $\{0^\circ,45^\circ,90^\circ,135^\circ\}$",
                r"$T_{\mathrm{high}}=P_{90}(\mathrm{NMS}>0)$,  "
                r"$T_{\mathrm{low}}=0.4\,T_{\mathrm{high}}$",
                r"Hysteresis: 8-connected BFS from strong into weak",
            ],
            "Strong pixels are NMS ≥ T_high; weak pixels satisfy T_low ≤ NMS < T_high. "
            "A weak pixel is promoted only if it is 8-adjacent to a strong edge (or to a "
            "previously promoted weak pixel). This is the only Canny configuration used in "
            "the reconstruction pipeline; t_low / t_high can be injected for unit tests. "
            "Contour extraction itself uses Moore tracing on the binary mask, not Canny, "
            "because Canny breaks on textured prints. Canny is retained as a required "
            "operator and is saved under results/edge_visualisations/.",
        )

        text_page(
            pdf,
            "7. Segmentation and piece IDs",
            "Otsu maximises between-class variance σ²_B(T) = w0(T) w1(T) (μ0 − μ1)² on the "
            "256-bin histogram of the enhanced grayscale image.\n\n"
            "Polarity: after Otsu, if the majority of pixels (or the brighter region) is "
            "labelled foreground, the mask is inverted so the table is background and pieces "
            "are white. Without this, a light table becomes one giant blob.\n\n"
            "Close (k=3) then hole-fill: flood-fill background from the image border; leftover "
            "zeros are interior holes (print texture) and are painted foreground.\n\n"
            "CCL is a two-pass union-find on 8-connected foreground. "
            "Components are rejected if area < max(min_area, min_area_frac × H × W), "
            "bbox aspect > 5, solidity < 0.35, area < 0.40 × median blob, or area > 0.25 of "
            "the frame (table remnant). When YOLO labels exist, keep_n is the unique class "
            "count so a 9-piece photo is not assembled on a 2×5 grid of 10 blobs. That uses "
            "identities only as a count, not as box crops (the spec path remains CCL).\n\n"
            "N = number of surviving components. Grid inference maps the labelled count (when "
            "extraction is close) to (R, C): 9→3×3, 16→4×4, 35→7×5. "
            "Extraction IDs are 0 … N−1 in CCL order. Physical piece numbers 1–35 come from "
            "YOLO class indices via data.yaml names, attached by box IoU when CCL is used.",
        )

        text_page(
            pdf,
            "8. Piece description and corners",
            "Moore-neighbour tracing walks the boundary of each labelled mask in the crop "
            "coordinate system.\n\n"
            "Corner method (hybrid, chosen over raw PCA rectangles):\n"
            "  1. Convex hull (Jarvis march) of the contour.\n"
            "  2. Ramer–Douglas–Peucker with ε = 0.02 × hull perimeter.\n"
            "  3. Among 4-point subsets, keep the quad with large area, similar side lengths, "
            "and interior angles in [55°, 125°]. Tab tips lie on the hull; a near-rectangular "
            "quad prefers the piece body over a tab apex.\n"
            "  4. Fallback: PCA-aligned rectangle snapped back onto the contour, then ordered "
            "clockwise from top-left.\n\n"
            "Each side is resampled along the contour between consecutive corners. The signed "
            "profile is the outward offset from the chord (positive = tab, negative = blank). "
            "Classification: mean |profile| near zero and low variance → flat; positive mean → "
            "tab; negative → blank. An inward Lab strip (length 32) is sampled just inside the "
            "edge. Out-of-mask samples are NaN, not Lab zeros, so missing colour cannot score "
            "as a perfect match.\n\n"
            "After sides are classified, the crop is rotated by k×90° so flats occupy a "
            "canonical local frame (corners: N+W; edges: N). Interior pieces stay in the "
            "deskewed frame. Assembly still tries all four rotations.",
        )

        math_page(
            pdf,
            "9. Matching formula and weights",
            "Every ordered pair of sides (i, si) and (j, sj) receives a dissimilarity D. "
            "The assembler minimises the sum of D over placed adjacencies. Weights are taken "
            "from configs/classical.yaml and sum to 1.",
            [
                r"$D = 0.85\,E_{\mathrm{shape}} + 0.15\,E_{\mathrm{colour}}$",
                r"$E_{\mathrm{shape}}=\mathrm{mean}((p + \mathrm{flip}(q))^2)$",
                r"$E_{\mathrm{colour}}=\mathrm{mean}\,\|Lab_a - \mathrm{flip}(Lab_b)\|_2^2$",
                r"$D = +\infty$  if $i=j$, either side is flat, or both are tab/tab or blank/blank",
            ],
            "Profiles are linearly resampled to 64 samples before the shape term; colour "
            "strips to 32. Complementary geometry means a tab’s positive bump cancels a "
            "blank’s negative pocket after reversal, so E_shape ≈ 0 for a true mate. "
            "Colour uses only samples that are finite on both sides; if no valid overlap "
            "remains, E_colour = +∞ (never 0). Illegal pairs stay IEEE infinity so they "
            "cannot win a greedy step. The Siamese matcher converts p_neighbour to a "
            "lower-better cost (−log p) and still marks class-illegal pairs as +∞. GNN uses "
            "the same cost convention but is a weak extra on this set (real val_ap ~0.26).",
        )

        text_page(
            pdf,
            "10. Assembly: seeds, ties, and dead ends",
            "Algorithm: greedy best-first with beam width K=8 (configs/classical.yaml).\n\n"
            "Seed. Top-left cell (0,0) is filled by a corner piece: two adjacent flats, "
            "rotated so those flats face north and west. If several corners exist, they all "
            "enter the beam, ranked by a dummy dissim of 0 then piece id.\n\n"
            "Expand. For each beam state, candidate cells are 4-neighbours of already placed "
            "pieces. Each unused piece is tried at each of 4 rotations. The placement cost is "
            "the sum of D to already-placed neighbours, plus a large finite penalty if a "
            "non-flat faces the outer border or a flat faces the interior. Any +∞ neighbour "
            "cost discards that move (illegal geometry is never placed).\n\n"
            "Ties. After each expansion the beam is sorted by:\n"
            "  1. more pieces placed (primary),\n"
            "  2. lower total dissimilarity,\n"
            "  3. lower minimum piece id as a deterministic last key.\n"
            "The global best follows the same order.\n\n"
            "Dead ends. If no legal growing move exists, the assembler does not force-fill "
            "empty cells. It returns the best partial arrangement seen so far. Incomplete "
            "boards are expected on hard photos; Q then reflects completeness < 1.",
        )

        math_page(
            pdf,
            "11. Quality scores",
            "Every run writes Q into results/evaluation_results/last.json. There is no "
            "assembled-image ground truth, so SSIM is not computed. Lead with "
            "identity-neighbour accuracy and complete-reconstruction (usually 0 on this "
            "set). Q is the same formula in code, the notebook, and this report. Missing "
            "position/orientation terms are 0, so without compact pose GT, Q ≤ 0.2. "
            "Completeness and border-flat accuracy are diagnostics, not Q.",
            [
                r"$Q=0.5\,A_{\mathrm{pos}}+0.3\,A_{\mathrm{ori}}+0.2\,A_{\mathrm{edge}}$",
                r"$A_{\mathrm{edge}}=A_{\mathrm{idn}}\ \mathrm{(identity\ neighbours)}$",
            ],
            "A_idn is the fraction of placed edges whose YOLO piece numbers are 4-adjacent "
            "on the completed 7×5. Orientation A_ori is the fraction of border pieces whose "
            "placement rot matches the unique identity+flats rot; interiors are omitted "
            "(None → 0 in Q). If identities compact onto the inferred grid, A_pos is "
            "reported; otherwise A_pos is 0 in Q. geometry_Q uses the same formula with "
            "tab↔blank accuracy as the edge term (still ≤ 0.2 without pose).",
        )

        text_page(
            pdf,
            "12. Experiments and protocol",
            "Unit tests cover filters, Otsu, Sobel/Canny, CCL polarity, corners, matching "
            "illegal=+∞, assembly partial boards, split leakage, and pose-GT mapping "
            "(piece 1 → (0,0), piece 35 → (6,4), compact 3×3).\n\n"
            "Reconstruction CLI:\n"
            "  python main.py reconstruct --method classical --config configs/classical.yaml \\\n"
            "      --input data/input/test/<file>\n\n"
            "Methods classical and siamese share extraction and assembly (GNN is an optional "
            "ablation). The Siamese matcher fails closed if checkpoints/siamese.pt is missing.\n\n"
            "Real-pair training (identity neighbours, relative ori from matching sides):\n"
            "  python main.py train-siamese --real --max-boards 80\n"
            "Neighbour identities come from canonical 7×5 adjacency of piece numbers. When "
            "both pieces have a unique identity+flats rotation, those rotations map canonical "
            "neighbour directions to local sides. Otherwise a unique tab↔blank pair is used, "
            "with lowest E_shape as a tie-break (not full classical D). Relative orientation "
            "is (si − sj) mod 4 from those sides — not unlabelled photo rotation. The ori "
            "head is trained on that label; inference down-weights pairs whose predicted ori "
            "disagrees.\n\n"
            "Do not train on data/input/test. After resplit, check_no_source_leakage must pass.",
        )

        text_page(
            pdf,
            "13. Limitations",
            "1. No assembled-image ground truth is provided. Visual inspection of the canvas "
            "cannot be replaced by an SSIM to a solved photo. Reported headline metrics "
            "are identity-neighbour accuracy and complete reconstruction; Q uses the "
            "canonical 0.5/0.3/0.2 formula with missing pose terms as 0.\n"
            "2. YOLO boxes are identities, not (row, col, rot) in the scrambled image. The "
            "canonical 7×5 map (piece 1 at top-left, row-major) is an assumption about how "
            "the physical puzzle was numbered; if the box numbering disagrees, pose Q is wrong.\n"
            "3. Photo rotation is unlabelled. Border placement ori is recovered from identity+"
            "flats. Relative ori for ML is (si − sj) mod 4 from matching sides. Interior "
            "piece pose remains unknown. Synthetic complementary ribbons remain a useful pretrain.\n"
            "4. CCL still fails when pieces touch or when print texture survives hole-fill as "
            "bridges. Over-segmentation is reduced by keeping the labelled piece count of "
            "largest blobs; YOLO crops avoid residual debris but are not the spec default.\n"
            "5. Colour is weak: most faces are near-white cardboard, which is why wc=0.15.\n"
            "6. Greedy beam search is not globally optimal; a wrong early seed cannot be undone.\n"
            "7. Median is a nested Python order-statistic loop (~17 s/board). Full 35-piece "
            "photos remain the expensive case.\n\n"
            "Despite these limits, the Milestone 1 operators, formula, weights, Canny "
            "parameters, and assembly rules are fully specified and implemented from scratch.",
        )

        text_page(
            pdf,
            "14. Conclusion",
            "Milestone 1 delivers a complete classical reconstruction path with explicit "
            "mathematics: D = 0.85 E_shape + 0.15 E_colour, illegal pairs at +∞, Canny with "
            "Gaussian k=5 σ=1, Sobel, 4-bin NMS, T_high = 90th percentile of NMS, "
            "T_low = 0.4 T_high, 8-connected hysteresis, and a beam assembler that returns "
            "the best partial board on dead ends, breaking ties by fill count then dissim.\n\n"
            "Q is numerical on every run: 0.5 A_pos + 0.3 A_ori + 0.2 A_idn, with missing "
            "terms as 0. Do not read a high completeness/border score as a solved puzzle. "
            "Milestone 2 plugs Siamese costs into the same tensor slot without forking "
            "assembly. GNN remains an optional weak extra.",
        )

        figure_operators(pdf)

    return path


if __name__ == "__main__":
    out = write_report()
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
