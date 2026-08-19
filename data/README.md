# Dataset

All puzzle data lives under `data/` (course spec). There is no top-level `detection/` folder.

```
data/
├── data.yaml              # class names (piece IDs 1–35) and split paths
├── splits.json            # frozen stem lists (do not edit by hand)
├── input/                 # scrambled puzzle images (model input)
│   ├── train/             # 4613 images — singles + most multi-piece boards
│   ├── val/               # 35 multi-piece boards (9–35 pieces)
│   └── test/              # 36 multi-piece boards (9–35 pieces); never train on these
├── ground_truth/          # labels, same filename stem as the image
│   ├── train/
│   ├── val/
│   └── test/
└── sample_pieces/         # tiny fixtures for unit tests and the demo
```

**Pairing rule:** `data/input/<split>/<stem>.jpg` ↔ `data/ground_truth/<split>/<stem>.txt`.

**Label format:** YOLO line `class cx cy w h` (normalized 0–1). `class` is the piece identity (`1`–`35` in `data.yaml`). These are piece boxes, not a finished assembled grid.

**Split rule:** val and test contain only reconstruction boards (at least 9 unique pieces). Isolated 1-piece photos stay in train. Frozen in `splits.json`; do not mix a test image into training.

**CLI:** pass a board from `data/input/test/` (or train/val during development):

```bash
python main.py reconstruct --method classical --config configs/classical.yaml \
  --input data/input/test/0718-1_Color_png.rf.f6b7f8ba974357f79903f0d9fcf4264e.jpg
```
