# 🧠 CNN & RNN from Scratch in NumPy

This project implements both **Convolutional Neural Networks (CNNs)** for image classification and **Recurrent Neural Networks (RNNs)** for sequence generation, built entirely from scratch using **NumPy**. No external ML libraries like PyTorch or TensorFlow are used. This is an educational codebase to deeply understand the mechanics of deep learning at the matrix level.

---

## 🧩 Models Implemented

### 🔹 1. CNN for Image Classification

Implemented in: `cnn.py`, `cnn_layers.py`, `layers.py`

**Architecture**:
```
conv → ReLU → maxpool → conv → ReLU → maxpool → fc → ReLU → fc → softmax
```

Supports:
- Custom weight initialization based on fan-in
- Max pooling and convolution implemented via loops
- Softmax classification loss
- Modular layer-wise forward and backward passes

Training loop managed via `Solver` class.

---

### 🔹 2. RNN for Image Captioning

Implemented in: `rnn.py`, `rnn_layers.py`, `layers.py`

**Architecture**:
```
Image Feature (via fc) → Initial Hidden State → Word Embedding → RNN → Vocabulary Scores (via fc)
```

Features:
- Implements caption generation with vanilla RNN
- Temporal softmax loss with masking support for padded captions
- Includes forward and backward pass of embedding, RNN, and output projection layers
- Test-time `sample()` method to auto-generate captions from image features

---

## 🛠️ Training & Optimization

### 🔧 `solver.py`

A modular training engine inspired by CS231n:
- Supports SGD, Adam (`optim.py`)
- Training/validation accuracy tracking
- Learning rate decay
- Batch sampling and weight updates per epoch

---

## 📁 Project Structure

```
.
├── cnn.py               # CNN model class
├── cnn_layers.py        # Conv and pooling layers (forward/backward)
├── rnn.py               # Captioning RNN class
├── rnn_layers.py        # RNN step-wise forward/backward logic
├── layers.py            # Shared layers (ReLU, FC, softmax)
├── solver.py            # Training engine for both models
├── optim.py             # Optimizers: SGD, Adam
├── cnn.ipynb            # CNN demo notebook
├── rnn.ipynb            # RNN demo notebook
```

---

## 🧪 Example Usage

### Train a CNN model

```python
from cnn import ConvNet
from solver import Solver

model = ConvNet()
solver = Solver(model, data={
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val
}, num_epochs=10, batch_size=64, optim_config={'learning_rate': 1e-3})
solver.train()
```

### Train a Captioning RNN

```python
from rnn import CaptioningRNN
from solver import Solver

model = CaptioningRNN(word_to_idx)
solver = Solver(model, data={
    'X_train': features_train, 'y_train': captions_train,
    'X_val': features_val, 'y_val': captions_val
}, num_epochs=5, batch_size=32, optim_config={'learning_rate': 1e-3})
solver.train()
```

---

## 🎯 Learning Goals

- Understand how forward/backward passes work in CNN and RNN layers
- Implement softmax loss over time series (temporal loss)
- Apply training from scratch without black-box libraries
- Learn how image features can drive sequence generation

---

## 🧱 Dependencies

- Python 3.7+
- NumPy

Install:
```bash
pip install numpy
```

---

## 👨‍💻 Author

**Hao-Chun Shih (Oscar)**  
📧 oscar10408@gmail.com  
🎓 University of Michigan – Master in Data Science

---

## 📜 License

This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).
