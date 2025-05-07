# 🧠 CNN & RNN from Scratch in NumPy

This project implements both **Convolutional Neural Networks (CNNs)** for image classification and **Recurrent Neural Networks (RNNs)** for sequence generation, built entirely from scratch using **NumPy**. No external ML libraries like PyTorch or TensorFlow are used. This is an educational codebase to deeply understand the mechanics of deep learning at the matrix level.

---

## 🧩 Models Implemented

### 🔹 1. CNN for Image Classification

Implemented in: `cnn.py`, `cnn_layers.py`, `layers.py`

### 🧠 CNN: Training Loss & Accuracy

This figure shows the CNN model’s learning progress. The **left** plot illustrates how the training loss decreases over iterations, and the **right** plot shows training and validation accuracy across epochs.

![CNN Training Performance](./Images/cnn.png)

- The training loss steadily decreases, indicating effective optimization.
- Validation accuracy closely follows training accuracy, showing minimal overfitting.

---

### 🧠 CNN: Learned Convolutional Filters

Below are the convolutional filters learned by the second convolutional layer in the CNN. Each small grid corresponds to a filter kernel visualized as an RGB patch.

![CNN Learned Filters](./Images/Learned_Features_CNN.png)

- The variety of colors and patterns suggests that the model has learned diverse low-level features such as edges, color gradients, and textures.
- These filters help capture visual cues useful for classification.

---


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

This figure showcases sample outputs from the RNN-based image captioning model. For each image, it shows the model-generated caption (`Sample`) and the ground-truth human-written caption (`GT`).

![RNN Captioning Samples](./Images/RNN_Test_Sample.png)

- Some generated captions are close to the ground truth, but `<UNK>` tokens highlight vocabulary limitations.
- The captions demonstrate the model’s ability to associate image features with meaningful sentence structures.

---

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
