# Handwritten Digit Recognition with a CNN (MNIST, PyTorch)

A small Convolutional Neural Network (CNN), built with PyTorch, that recognises handwritten digits (0–9).
It is trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and then tested on real handwritten digits.



## Results

| Evaluation | Result |
|---|---|
| MNIST test set (10,000 images) | **98.73 %** accuracy after 3 epochs |
| Own handwritten samples (`samples/`) | **10 / 10** correct |

Training uses a fixed random seed (`42`), so re-running the notebook gives the same numbers on the same hardware.

## Repository contents

| File / folder | Description |
|---|---|
| `MNIST_2.ipynb` | Notebook covering data loading, training, evaluation, saving the model and predicting new images |
| `cnn_mnist_model.pt` | Trained model weights (PyTorch `state_dict`, about 1.7 MB) |
| `samples/` | Hand-drawn digits 0–9 (black on white) used to test the model |
| `predictions.png` | Model predictions on the sample images |
| `requirements.txt` | Python dependencies |

The MNIST dataset is not included. `torchvision` downloads it into `data/` the first time the notebook runs.

## Model

```
Input 1×28×28
 → Conv2d(1→32, 3×3) → ReLU → MaxPool(2)    # 32×14×14
 → Conv2d(32→64, 3×3) → ReLU → MaxPool(2)   # 64×7×7
 → Flatten → Linear(3136→128) → ReLU
 → Linear(128→10)                            # one score per digit
```

- **Parameters:** 421,642
- **Loss / optimiser:** Cross-entropy, Adam (learning rate 0.003), batch size 64, 3 epochs
- **Data augmentation:** random rotation (±15°), shifts (±10%) and scaling (0.9–1.1). This helps the model cope with real handwriting that is tilted or off-centre.

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook MNIST_2.ipynb
```

Run all cells. The notebook uses an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS) if one is available, and the CPU otherwise.
Training takes a few minutes on a GPU and somewhat longer on a CPU.

You can also open the notebook in [Google Colab](https://colab.research.google.com/). The last cell, which is commented out, lets you upload an image there.

## Using the trained model without retraining

```python
import torch
from PIL import Image

# Paste the `Net` class from the notebook first
model = Net()
model.load_state_dict(torch.load("cnn_mnist_model.pt", map_location="cpu"))
model.eval()
```

Then use the notebook's `preprocess()` / `predict_digit()` functions to classify an image file.

## Predicting your own digits

1. Write a digit in dark ink on white paper (or draw one digitally) and save it as a `.png`.
2. Put the file in `samples/`.
3. Re-run the last section of the notebook.

Each image is first converted into MNIST format:

1. Convert to grayscale and invert the colours (MNIST digits are white on black).
2. Remove faint background noise.
3. Crop to the digit and scale it so its longer side is 20 px.
4. Centre it on a 28×28 black canvas and normalise it with the MNIST mean and standard deviation.

Steps 2–4 make a big difference. If the whole photo is simply resized to 28×28, the digit shrinks to a few faint pixels. With that simpler approach, only 5 of the 10 samples were classified correctly.

## Requirements

- Python 3.9+
- PyTorch, torchvision, matplotlib, Pillow, Jupyter (see `requirements.txt`)

## Author

Walusimbi Uthman
