# Handwritten Digit Recognition using Streamlit

A simple AI-powered handwritten digit recognition web app built using Python, TensorFlow/Keras, and Streamlit.

Users can draw a digit on an external drawing canvas, and the trained deep learning model predicts the digit in real time.

---

## Features

- Handwritten digit prediction (0–9)
- Deep learning model trained on the MNIST dataset
- Streamlit web interface
- Image preprocessing pipeline
- Beginner-friendly project structure
- Fast prediction results

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Streamlit
- Pillow

---

## Project Structure

```text
handwritten-digit-recognition/
│
├── app.py
├── train_model.py
├── preprocess.py
├── model.h5
├── requirements.txt
├── README.md
└── sample_images/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/poojarysohan6361-star/handwritten-digit-recognition.git
```

Go to the project folder:

```bash
cd handwritten-digit-recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Model Information

The model is trained using the MNIST handwritten digit dataset using a Convolutional Neural Network (CNN).

### Input
- 28x28 grayscale image

### Output
- Predicted digit from 0–9

---

## Future Improvements

- Draw directly inside the Streamlit app
- Better UI/UX
- Confidence score visualization
- Support for custom datasets
- Model optimization for faster inference

---

## Author

Sohan Poojary

GitHub: https://github.com/poojarysohan6361-star
