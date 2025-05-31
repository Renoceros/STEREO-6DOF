#  6D Pose Estimation with Stereo ConvNeXt

This project explores **6D pose estimation** using stereo images and **ConvNeXt V2 Nano** backbones. It compares different architectural approaches—including single-image, twin-view shared-weight, and early fusion methods—while logging evaluation metrics and training performance for rigorous experimentation.

---

##  Project Structure

```
.
├── ArUco/                   # Image refrences for the cube faces
├── dataset/                 # Stereo images + labels
├── model/                   # Trained .pth checkpoints, evaluation sheets (.xlsx/.md)
├── runs/                    # TensorBoard logs for each training run
├── streamlit/               # Streamlit dashboard app
├── utility/                 # Stereo + rotation helper functions
├── video/                   # Stereo video batches (left/right pair)
├── *.ipynb / *.py           # Training scripts per model variant
```

---

##  Model Variants

There are **three main architectural variants**, each targeting stereo information differently:

###  VANILA

* **Single-image model**
* Uses left stereo image only
* Standard 3-channel RGB input
* 640→256→128→9 head (default)
* Acts as the baseline

###  SW-TWIN

* **Shared-weight twin-image model**
* Splits stereo image into L/R views
* Passes each through the same ConvNeXt backbone
* Extracted features are concatenated (640×2)
* Allows stereo disparity learning via feature-level fusion

###  6CH

* **Early fusion**
* Combines L and R into a 6-channel image (lR, lG, lB, rR, rG, rB)
* First ConvNeXt conv layer is patched to accept 6 channels
* Stereo fused at input level → early joint feature learning

---

##  Evaluation Metrics

Each model variant logs the following per epoch:

* **Loss/Train**, **Loss/Val**
* **RMSE**:

  * Translation RMSE (mm)
  * Rotation RMSE (deg)
* **Accuracy** (optional)
* **Inference time** (ms/image)
* **VRAM usage** (approximate)
* **Training time per epoch**

All logs are accessible through:

* `model/EVAL.xlsx` or `.ods`
* TensorBoard logs in `runs/`
* Markdown summaries in `model/`

---

##  Streamlit Dashboard

Launch the dashboard:

```bash
cd streamlit
streamlit run app.py
```

### Navigation

* **Home**: Overview and context
* **Dashboard**: Training curves + evaluation graphs
* **Documentation**: Per-model `.md` report viewer
* **Implementation**: Architecture, logs, dataset parameters

---

##  Training

Each model has its own training script:

* `S-ConvNeXt6DP.py` for Vanilla
* `S-SW-ConvNeXt6DP.py` for Double
* `S6ch-ConvNeXt6DP.py` for 6CH

Each script supports:

* Resume from checkpoint
* AMP (Automatic Mixed Precision)
* TensorBoard logging
* Early stopping
* L1 unstructured pruning every 5 epochs

---

##  Dataset Format

Stereo images are stored as **side-by-side composites**:

```
dataset/
└── batch{n}/
    ├── images/
    │   ├── 00001.png
    │   ├── ...
    ├── labels.csv
    └── metadata_{n}.md   ← Per-batch metadata
```

Labels are 9 values per image:
`[tx, ty, tz, qx, qy, qz, qw, is_translation_valid, is_rotation_valid]`

---

##  Dependencies

* PyTorch
* `timm`
* `torchvision`
* `openpyxl`, `pyexcel-ods3` (for evaluation)
* `streamlit`, `matplotlib`, `pandas`

---

##  Insights

This repo was built to **experiment with stereo-based improvements** to 6D pose estimation. It aims to be lightweight, modular, and flexible for comparing architectures with minimal overhead.

By centralizing logs, checkpoints, and metadata in a consistent structure, it allows for scalable and repeatable experimentation.

