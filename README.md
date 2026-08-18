# CSE475s Jigsaw Reconstruction

![Ain Shams University Faculty of Engineering](assets/en_logo.png)

**Course:** CSE475s Machine Vision, Ain Shams University, Faculty of Engineering, Computer and Systems, Summer 2026.

Classical computer-vision reconstruction (Milestone 1), then Siamese CNN and GNN matching (Milestone 2) that reuse the same assembly algorithm.

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
  --input data/input/test/1-LINE_ALBUM_-1_220521_0_jpg.rf.fe8a263c6682800d79f518484f4fe1f7.jpg
```

Methods later: `classical`, `siamese`, `gnn`. Milestone 1 uses `classical` on CPU only.

## Dataset

Everything is under **`data/`** (no `detection/` folder). See [data/README.md](data/README.md).

| Path | Role | Count |
|---|---|---|
| `data/input/train` | scrambled images, training | 4037 |
| `data/input/val` | scrambled images, validation | 622 |
| `data/input/test` | scrambled images, held-out test | 25 |
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
