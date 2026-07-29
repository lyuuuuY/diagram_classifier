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

## Installation and Usage

The trained model is included through Git LFS. Users do not need to run
hyperparameter search or train the model again.

Clone the repository and download the trained checkpoint:

```powershell
git lfs install
git clone https://github.com/lyuuuuY/diagram_classifier.git
cd diagram_classifier
git lfs pull
```

Create and activate a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the CPU dependencies on a computer without an NVIDIA GPU:

```powershell
python -m pip install -r requirements_cpu.txt
```

Alternatively, install the CUDA 12.8 dependencies on a computer with a
compatible NVIDIA GPU:

```powershell
python -m pip install -r requirements_cu128.txt
```


### Classify an Image

The direct `classify()` interface requires a 224×224 PNG:

```powershell
python -c "from classifier import classify; print(classify(r'path\to\image.png'))"
```

The output is one of:

```text
lewis
graph
equation
```

### Use the Web Interface

The web interface accepts PNG images of any dimensions and automatically
resizes and pads them to 224×224:

```powershell
python app.py
```

Open the following address in a browser:

```text
http://127.0.0.1:5000/
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


## Model Limitations

The training data is relatively simple: most images contain one Lewis
structure, one equation, or a connected graph made of basic nodes and edges.
Therefore, the model performs well on test images with similar styles.

Performance decreases on the new-scope dataset. Colored 3D molecules may be
classified as graphs because atoms and bonds resemble nodes and edges. Lewis
structures with dark backgrounds or multiple structures may be classified as
equations because their layout resembles mathematical notation. Some incorrect
predictions also have high confidence, showing that the model is less reliable
on unfamiliar image styles. A more diverse training set would improve
generalization.



