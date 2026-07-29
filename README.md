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

The interface is defined in `classifier.py`:

```python
from classifier import classify

label = classify(r"path\to\image.png")
print(label)
```

`The return value is exactly one of:

```text
lewis
graph
equation
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

Hyperparameter search and final training are separated:

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

An additional 34 images are stored in `merged_data/testing_new_scope` to
measure generalization to new visual styles. These images differ substantially from the training data: 
they contain a wider range of colors, more complex equations, and, in some cases, multiple Lewis structures within a single image.

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
├── merged_data/                   # Final dataset for training, validation and testing
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

The dependency files are separated by compute platform:

- `requirements.txt` contains the common dependencies shared by all systems.
- `requirements-cpu.txt` installs the CPU build of PyTorch and TorchVision.
- `requirements-cu128.txt` installs the CUDA 12.8 builds used in the original
  development and training environment.

## Installation

The commands below use Windows PowerShell. Python 3.11 is recommended because
the project was developed and tested with Python 3.11.8.

Create a virtual environment:

```powershell
cd E:\Programe__AA\classifier
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a computer without an NVIDIA GPU, install the CPU dependencies:

```powershell
python -m pip install -r requirements-cpu.txt
```

For a computer with a compatible NVIDIA GPU, install the CUDA 12.8
dependencies:

```powershell
python -m pip install -r requirements-cu128.txt
```

To install only the common dependencies, without installing PyTorch or
TorchVision:

```powershell
python -m pip install -r requirements.txt
```

Only one of `requirements-cpu.txt` or `requirements-cu128.txt` should be used
for a given virtual environment.
```

This repository uses Git LFS to track the trained model checkpoint. After
cloning the repository, install Git LFS and download the model:

```powershell
git lfs install
git lfs pull
```


## Local Web Interface

Start Flask:

```powershell
.\.venv\Scripts\python.exe app.py
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




