# Dataset

All puzzle data lives under `data/` (course spec). There is no top-level `detection/` folder.

```
data/
├── data.yaml              # class names (piece IDs 1–35) and split paths
├── input/                 # scrambled puzzle images (model input)
│   ├── train/             # 4037 images — training only
│   ├── val/               # 622 images — validation only
│   └── test/              # 25 images — held-out test; never train on these
├── ground_truth/          # labels, same filename stem as the image
│   ├── train/
│   ├── val/
│   └── test/
└── sample_pieces/         # tiny fixtures for unit tests and the demo
```

**Pairing rule:** `data/input/<split>/<stem>.jpg` ↔ `data/ground_truth/<split>/<stem>.txt`.

**Label format:** YOLO line `class cx cy w h` (normalized 0–1). `class` is the piece identity (`1`–`35` in `data.yaml`). These are piece boxes, not a finished assembled grid.

**Split rule (Milestone 2):** train / val / test are frozen by folder. Do not mix a test image or its pairs into training.

**CLI:** pass an image from `data/input/test/` (or train/val during development):

```bash
python main.py reconstruct --method classical --config configs/classical.yaml \
  --input data/input/test/1-LINE_ALBUM_-1_220521_0_jpg.rf.fe8a263c6682800d79f518484f4fe1f7.jpg
```
