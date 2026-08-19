# CSE480 Jigsaw Reconstruction

![Ain Shams University Faculty of Engineering](assets/en_logo.png)

**Course:** CSE480 Machine Vision, Ain Shams University, Faculty of Engineering, Mechatronics Engineering, Summer 2026.

Classical computer-vision reconstruction (Milestone 1), then Siamese CNN matching (Milestone 2) that reuse the same assembly algorithm. GNN matching is implemented as a weak extra (real val_ap ~0.26) and is not the default ML matcher.

**Student:** Abdlrhman Hisham Ismail, 2300343.

**Instructor:** Prof. Hossam El-Din Hassan Abd El Munim, Computer and Systems Engineering.

**Teaching assistants:**
- Dina Zakaria Mahmaud Mahammed, Demonstrator, Mechatronics Engineering
- Mohamed Ahmed Mohamed Abdelhalim Mohanna, Demonstrator, Mechatronics Engineering

## Clone

```bash
git clone https://github.com/abdlrhmanv/cse475s-jigsaw-reconstruction.git
cd cse475s-jigsaw-reconstruction
```

If the GitHub repository uses a different URL, replace the clone line with the URL from the repo page.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Reconstruct a puzzle

```bash
python main.py reconstruct --method classical --config configs/classical.yaml --input data/input/test/<file>
```

Example:

```bash
python main.py reconstruct --method classical --config configs/classical.yaml \
  --input data/input/test/0718-11_Color_png.rf.d936d44a14098628d03f3c91ac6d1fc3.jpg
```

Methods: `classical` (default), `siamese` (the ML matcher), `gnn` (weak extra / ablation). Siamese requires `checkpoints/siamese.pt` (train first or the CLI exits). The board size is inferred from the labelled piece count when CCL is close (9→3×3, 16→4×4, 35→7×5). Default extraction is CCL from the enhanced mask, keeping the N largest blobs where N is the unique YOLO class count; set `segmentation.use_gt_boxes: true` to crop YOLO boxes instead.

Each run writes metrics under `results/evaluation_results/last.json`. **Lead with** `identity_neighbour_accuracy` and `complete_reconstruction`. There is no assembled-image ground truth, so SSIM is not computed.

One quality formula everywhere (code, notebook, report):

```text
Q = 0.5 * position_accuracy + 0.3 * orientation_accuracy + 0.2 * edge_accuracy
```

Missing terms are 0. Without compact pose GT, Q is at most 0.2 even if the grid is full and border flats look clean. Completeness / border-flat / tab↔blank are diagnostics, not Q. When YOLO identities compact onto the inferred grid, compact position accuracy is reported. Orientation is scored from identity+flats on border pieces (not from unlabelled photo rotation); interior pieces are omitted.

Median filtering is a from-scratch per-pixel order statistic (not convolution / not OpenCV) and takes ~17 s per board on this set.

Train the ML matcher on real boards. Neighbour labels are piece-ID adjacencies; matching sides prefer identity+flats when unique, else tab↔blank / E_shape. Relative ori is `(si − sj) mod 4`:

```bash
python main.py train-siamese --real --max-boards 80
```

GNN ablation (optional):

```bash
python main.py train-gnn --real --max-boards 40
```

## Dataset

Everything is under **`data/`** (no `detection/` folder). See [data/README.md](data/README.md).

| Path | Role | Count |
|---|---|---|
| `data/input/train` | scrambled images, training | 4555 |
| `data/input/val` | scrambled images, validation | 68 |
| `data/input/test` | scrambled images, held-out test | 61 |
| `data/ground_truth/{train,val,test}` | YOLO piece boxes, same stem as the image | matching |
| `data/sample_pieces/` | tiny fixtures for tests | few files |
| `data/data.yaml` | 35 piece-class names | — |

Do not train on `data/input/test`. Labels are `class cx cy w h` (piece IDs 1–35), not an assembled-grid pose.

## Repository layout

```
project-repository/
├── README.md
├── assets/en_logo.png
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── enhancement.py
│   ├── thresholding.py
│   ├── edge_detection.py
│   ├── segmentation.py
│   ├── contour_extraction.py
│   ├── piece_description.py
│   ├── edge_matching.py
│   ├── assembly.py
│   └── evaluation.py
├── tests/
│   ├── test_enhancement.py
│   ├── test_thresholding.py
│   ├── test_edge_detection.py
│   ├── test_segmentation.py
│   ├── test_piece_description.py
│   ├── test_edge_matching.py
│   └── test_assembly.py
├── data/
│   ├── data.yaml
│   ├── input/{train,val,test}/
│   ├── ground_truth/{train,val,test}/
│   └── sample_pieces/
├── results/
│   ├── enhanced_images/
│   ├── masks/
│   ├── contours/
│   ├── edge_visualisations/
│   ├── reconstructed_images/
│   └── evaluation_results/
├── notebooks/
│   └── demonstration.ipynb
└── report/
    └── milestone_1_report.pdf
```

Additional OOP modules live under `src/core/` and `src/ml/` and do not replace the required files above.
