import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

data = np.load("model_weights.npz")
W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ)

def predict(img_array):
    image = Image.fromarray(img_array).convert("L").resize((28, 28))
    arr = np.array(image).reshape(784, 1) / 255.0
    Z1 = W1.dot(arr) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return int(np.argmax(A2)), float(np.max(A2)) * 100

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below. Built from scratch with NumPy — no external AI used.")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    if st.button("Predict"):
        pred, conf = predict(canvas_result.image_data.astype(np.uint8)[:, :, :3])
        st.write(f"### Predicted digit: {pred} ({conf:.1f}% confident)")