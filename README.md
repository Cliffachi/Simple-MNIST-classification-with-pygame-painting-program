# MNIST Digit Classifier with Pygame 🎨

A simple handwritten digit recognition project built with **PyTorch, Convolutional Neural Networks (CNNs), and Pygame**. Draw a digit on the screen and let the trained model predict which number it represents.

## ✨ Features

- **Interactive drawing canvas** — Draw digits from 0 to 9 using your mouse.
- **CNN-powered classification** — Use a trained convolutional neural network to recognize handwritten digits.
- **Real-time interface** — View your drawing and the model's prediction in a desktop application.
- **MNIST dataset** — Train and evaluate the model using the classic handwritten digit dataset.
- **PyTorch implementation** — Explore how a CNN processes images and makes predictions.

## 🖥️ How It Works

The project combines a trained deep learning model with a simple graphical interface.

1. **Draw:** Use your mouse to draw a digit on the Pygame canvas.
2. **Preprocess:** Save the drawing as an image, convert it to grayscale, resize it to 28 × 28 pixels, and normalize its pixel values.
3. **Classify:** Pass the processed image through the trained CNN.
4. **Predict:** Select the digit with the highest model output score and display the prediction.

```text
Mouse Input
    │
    ▼
Pygame Drawing Canvas
    │
    ▼
Image Preprocessing
Grayscale → Resize → Normalize
    │
    ▼
Convolutional Neural Network
    │
    ▼
Predicted Digit (0–9)
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Building and running the CNN |
| Torchvision | Loading the MNIST dataset and image transformations |
| Pygame | Interactive drawing interface |
| Pillow (PIL) | Image loading and grayscale conversion |
| NumPy | Numerical operations |

## 📁 Project Structure

```text
Simple-MNIST-classification-with-pygame-painting-program/
├── app.py              # Pygame drawing interface
├── cnn.py              # CNN architecture and prediction function
├── main.py             # Main entry point
├── cnn.pth             # Saved model weights
├── train_model.ipynb   # Model training notebook
├── drawing.png         # Example or saved drawing
├── requirements.txt    # Python dependencies
└── data/
    └── MNIST/          # MNIST dataset files
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Cliffachi/Simple-MNIST-classification-with-pygame-painting-program.git
```

Navigate into the project directory:

```bash
cd Simple-MNIST-classification-with-pygame-painting-program
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

If you want to explore the main script or training workflow, inspect `main.py` and `train_model.ipynb` as well.

## 🎮 How to Use

1. Launch the application.
2. Draw a handwritten digit on the black canvas using your mouse.
3. Choose a brush size from the side menu if needed.
4. Click the save button to process your drawing and run the classifier.
5. Check the prediction displayed at the top of the window.
6. Draw another digit and repeat!

**Tip:** For better results, draw a clear, centered digit with strokes resembling those in the MNIST dataset.

## 🧠 Model Architecture

The project uses a convolutional neural network designed for 28 × 28 grayscale images.

| Layer | Operation | Output shape |
|---|---|---|
| Input | Grayscale image | 1 × 28 × 28 |
| Convolution 1 | 32 filters, 3 × 3 kernel, ReLU | 32 × 28 × 28 |
| Max pooling 1 | 2 × 2 pooling | 32 × 14 × 14 |
| Convolution 2 | 64 filters, 3 × 3 kernel, ReLU | 64 × 14 × 14 |
| Max pooling 2 | 2 × 2 pooling | 64 × 7 × 7 |
| Flatten | Convert feature maps to a vector | 3,136 |
| Fully connected | 128 neurons, ReLU | 128 |
| Output | 10 class scores | 10 |

The output layer produces scores for the ten possible digits, from 0 through 9. The predicted digit is the class with the highest score.

## 📚 Training

The `train_model.ipynb` notebook is provided for exploring the model training process.

The project uses the MNIST dataset, which contains handwritten digit images and their corresponding labels.

To experiment with the model:

1. Explore the dataset and preprocessing steps.
2. Inspect the CNN architecture in `cnn.py`.
3. Train the model using the notebook.
4. Evaluate its performance on unseen test images.
5. Save the trained model weights and use them in the Pygame application.

The repository also includes `cnn.pth`, a saved model checkpoint used by the application.

## 🔍 Limitations

- The model is trained on MNIST-style digits, so handwriting that differs significantly from the training data may reduce accuracy.
- Drawings are resized to 28 × 28 pixels, which can distort digits if they are poorly centered or scaled.
- Predictions depend on the quality of the drawing and the preprocessing pipeline.
- The project is intended as a learning and demonstration project rather than a production-grade OCR system.

## 🎯 Learning Objectives

This project is useful for learning how to:

- Build a CNN for image classification with PyTorch.
- Work with image datasets using Torchvision.
- Preprocess images for deep learning models.
- Load trained model weights for inference.
- Integrate a machine learning model into a desktop application.
- Connect a graphical user interface to a deep learning pipeline.

## 🔮 Future Improvements

Potential improvements include:

- Display the top three predictions with confidence scores.
- Add a clear-canvas button.
- Improve digit centering and scaling during preprocessing.
- Display model accuracy and training metrics.
- Add a confusion matrix to analyze classification errors.
- Compare the CNN against a simple multilayer perceptron (MLP).

## 📄 License

No license information is specified here. If you plan to distribute or reuse this project, consider adding an appropriate license to the repository.

---

Built as a hands-on project to explore deep learning and interactive handwritten digit recognition with Python.
