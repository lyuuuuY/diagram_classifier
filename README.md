# Diagram Image Classifier

A three-class image classifier built with Python and PyTorch. It recognizes:

- `lewis`: Lewis structure diagrams
- `graph`: connected graph diagrams
- `equation`: calculus equations

Inputs may be handwritten, screenshot from webpage, or computer-generated. The
model uses an ImageNet-pretrained ResNet18 and transfer learning with no more
than 500 training images. The project includes synthetic data generation, image
processing, hyperparameter search, final model training, test visualization, a
command-line classification function, and a local web interface.

## Required Classification Interface

The interface required by the assignment is defined in `classifier.py`:

```python
from classifier import classify

label = classify(r"path\to\image.png")
print(label)
```

`classify(filepath)` has the following contract:

- `filepath` must be a Python `str`.
- The file must be a PNG.
- The image must be exactly `224 × 224`.
- The image must use 8-bit color channels in the range 0–255.
- The return value is exactly one of:

```text
lewis
graph
equation
```

PowerShell example:

```powershell
.\.venv\Scripts\python.exe -c "from classifier import classify; print(classify(r'merged_data\testing\lewis\YOUR_IMAGE.png'))"
```

## Model and Training Strategy

- Architecture: `torchvision.models.resnet18`
- Initial weights: ImageNet pretrained weights
- Input size: `224 × 224`
- Normalization:
  - Mean: `(0.485, 0.456, 0.406)`
  - Standard deviation: `(0.229, 0.224, 0.225)`
- Optimizer: AdamW
- Loss: cross-entropy loss
- Primary model-selection metric: validation macro F1

Training has two phases:

1. Freeze the ResNet18 backbone and train only the new classification head.
2. Unfreeze `layer4` and the classification head for fine-tuning.

Hyperparameter search and final training are deliberately separated:

- `parameter_search.py` searches, compares, and saves hyperparameter results.
  It does not read the testing set and does not write `best_model.pth`.
- `train_best_model.py` trains one final model using manually entered
  hyperparameters and fixed epoch counts, then evaluates the testing set once.

## Current Dataset

Each source is shuffled and split independently using an 8:1:1 ratio. The
current standard dataset contains:

| Split | Equation | Graph | Lewis | Total |
|---|---:|---:|---:|---:|
| Training | 141 | 167 | 152 | 460 |
| Validation | 18 | 22 | 19 | 59 |
| Testing | 18 | 20 | 19 | 57 |

The training set contains 460 images and therefore satisfies the assignment
limit of at most 500 training images.

An additional 34 images are stored in `merged_data/testing_new_scope` to
measure generalization to new visual styles. This directory uses a flat layout.
Every filename must begin with exactly one of `equation`, `graph`, or `lewis`;
`evaluate.py` uses that filename prefix as the true label.

Standard dataset layout:

```text
merged_data/
├── training/
│   ├── equation/
│   ├── graph/
│   └── lewis/
├── validation/
│   ├── equation/
│   ├── graph/
│   └── lewis/
├── testing/
│   ├── equation/
│   ├── graph/
│   └── lewis/
└── testing_new_scope/
```

## Project Structure

```text
classifier/
├── app.py                         # Local Flask web backend
├── classifier.py                  # Required classify(filepath) interface
├── evaluate.py                    # Evaluation and result visualization
├── parameter_search.py            # Grid search and multi-seed confirmation
├── train_best_model.py            # Manual parameters and fixed-epoch training
├── requirements.txt               # Exact direct dependency versions
├── models/
│   ├── best_model.pth             # Final ResNet18 checkpoint
│   ├── best_hyperparameters.json  # Hyperparameter-selection report
│   └── class_names.json           # Class-to-index mapping
├── utilities/
│   ├── calculus_equation_generator.py
│   ├── connected_graph_generator.py
│   ├── connected_graph_generator_style2.py
│   ├── lewis_generator.py
│   ├── plot_convert.py
│   ├── split_dataset.py
│   └── split_handwritten.py
├── data_calculus_equations/
├── data_Connected_graph/
├── data_lewis/
├── merged_data/
├── templates/
├── static/
├── training_results/
└── test_results/
```

## Environment and Dependency Versions

Development and validation environment:

| Component | Version |
|---|---|
| Python | 3.11.8 |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| Flask | 3.1.3 |
| Werkzeug | 3.1.8 |
| Matplotlib | 3.11.1 |
| NetworkX | 3.6.1 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |

`requirements.txt` uses the PyTorch CUDA 12.8 wheels to reproduce the current
development environment. At runtime, the program selects CUDA automatically
when it is available and otherwise falls back to CPU. The CUDA packages still
require substantially more disk space even when inference runs on CPU.

## Installation

The commands below use Windows PowerShell.

Create a virtual environment:

```powershell
cd E:\Programe__AA\classifier
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the model is tracked with Git LFS, download it after cloning:

```powershell
git lfs install
git lfs pull
```

## Data Generation

Run all commands from the project root.

Generate Lewis structures:

```powershell
.\.venv\Scripts\python.exe utilities\lewis_generator.py --count 130
```

Generate connected graphs:

```powershell
.\.venv\Scripts\python.exe utilities\connected_graph_generator.py --count 150
```

Generate labeled, colored connected graphs with optional arrows:

```powershell
.\.venv\Scripts\python.exe utilities\connected_graph_generator_style2.py --count 40
```

Generate calculus equations:

```powershell
.\.venv\Scripts\python.exe utilities\calculus_equation_generator.py --count 180
```

Each generator supports `--output-dir`, `--size`, and `--seed`. Use `--help`
to see every option:

```powershell
.\.venv\Scripts\python.exe utilities\lewis_generator.py --help
```

## Image Conversion and Handwritten Image Splitting

`plot_convert.py` removes large empty background areas, proportionally resizes
the diagram, centers it on a padded canvas, and writes a 224×224 RGB PNG:

```powershell
.\.venv\Scripts\python.exe utilities\plot_convert.py `
  --input-dir data_lewis\generated `
  --output-dir data_lewis\converted `
  --size 224
```

Explanatory text beneath a Lewis structure is removed by default. Use
`--keep-annotations` to preserve it.

Split handwritten grid pages into individual images:

```powershell
.\.venv\Scripts\python.exe utilities\split_handwritten.py `
  --input-dir data_lewis\hand_written `
  --output-dir data_lewis\han_written_split `
  --rows 6 `
  --columns 2 `
  --size 224
```

## Rebuilding the 8:1:1 Dataset

```powershell
.\.venv\Scripts\python.exe utilities\split_dataset.py
```

The default seed is `20260729`. The script recreates
`merged_data/training`, `merged_data/validation`, and `merged_data/testing`.
The original source directories are not modified.

The configured sources are:

- Lewis:
  - `data_lewis/converted`
  - `data_lewis/generated`
  - `data_lewis/han_written_split`
- Connected graph:
  - `data_Connected_graph/generated`
  - `data_Connected_graph/generated_style2`
  - `data_Connected_graph/han_written_split`
- Calculus equation:
  - `data_calculus_equations/generated`
  - `data_calculus_equations/han_written_split`

## Hyperparameter Search

Run all 16 hyperparameter combinations and confirm the top configurations with
multiple seeds:

```powershell
.\.venv\Scripts\python.exe parameter_search.py `
  --device auto `
  --batch-size 16 `
  --num-workers 4
```

Default parameter grid:

```python
PARAM_GRID = {
    "head_learning_rate": (3e-4, 1e-3),
    "finetune_learning_rate": (1e-5, 3e-5),
    "weight_decay": (1e-4, 1e-3),
    "dropout": (0.2, 0.5),
}
```

Default confirmation seeds:

```text
17, 42, 2026
```

Primary outputs:

```text
models/best_hyperparameters.json
training_results/grid_search_results.csv
```

The search process never evaluates the testing set and does not overwrite an
existing `models/best_model.pth`.

## Training the Final Model with Manual Settings

Open `train_best_model.py` and edit the settings near the top:

```python
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-3
DROPOUT = 0.5
SEED = 17

HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 15

BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "auto"
```

`HEAD_EPOCHS` and `FINETUNE_EPOCHS` are exact training lengths. Early stopping
is disabled for final training, so both phases run for the complete number of
epochs specified. The saved weights are the checkpoint with the strongest
validation performance among those fixed epochs.

Start final training:

```powershell
.\.venv\Scripts\python.exe train_best_model.py
```

Primary outputs:

```text
models/best_model.pth
models/class_names.json
training_results/final_training_config.json
training_results/final_training_history.json
training_results/test_metrics.json
training_results/training_curves.png
training_results/confusion_matrix.png
```

`train_best_model.py` evaluates the standard testing set once. If the Flask
application is already running when the checkpoint is replaced, restart
`app.py`; otherwise, the web process may continue using its cached model.

## Evaluation and Visualization

Evaluate the saved model on both the standard and new-scope testing sets:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Reports are saved separately:

```text
test_results/testing/
test_results/testing_new_scope/
```

Each report directory contains:

```text
test_metrics.json
test_predictions.csv
test_summary.png
test_predictions_gallery.png
misclassified_examples.png
```

Example with explicit options:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --model-path models\best_model.pth `
  --test-dir merged_data\testing `
  --new-scope-dir merged_data\testing_new_scope `
  --batch-size 16 `
  --num-workers 4 `
  --device auto
```

## Local Web Interface

Start Flask:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:5000/
```

The page accepts a PNG of any dimensions through drag-and-drop or file upload.
The backend:

1. Resizes the image while preserving its aspect ratio.
2. Estimates the background color from the four corners.
3. Centers and pads the result to `224 × 224`.
4. Calls `classifier.classify()`.

Upload limits:

- PNG only
- Smaller than 5 MB
- No more than 50 million source pixels

Health endpoint:

```text
GET http://127.0.0.1:5000/api/health
```

Classification endpoint:

```text
POST http://127.0.0.1:5000/api/classify
Content-Type: multipart/form-data
Form field: image
```

## GitHub and Model Files

`best_model.pth` is a binary model file and should be tracked with Git LFS:

```powershell
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

Do not commit `.venv`, `__pycache__`, or local cache directories. After cloning,
install the dependencies and use Git LFS to download the model before running
`classify()`, `evaluate.py`, or the web application.

## Reproducibility

- Dataset splitting uses a fixed seed.
- Every DataLoader worker receives a deterministic seed.
- Python, NumPy, and PyTorch seeds are set together.
- cuDNN benchmarking is disabled.
- PyTorch deterministic algorithms are enabled with `warn_only=True`.
- Small floating-point differences may still occur across hardware, CUDA
  versions, drivers, and low-level operators.

## License

No open-source license has been added yet. Without a license, the repository
does not automatically grant others permission to copy, modify, or redistribute
the code.
