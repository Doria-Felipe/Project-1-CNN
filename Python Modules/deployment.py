import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

MODEL_PATH = "challenger8_fixed.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model input: (1, 32, 32, 3) float32 in [0,1]."""
    img = img.convert("RGB").resize((32, 32))
    x = np.array(img).astype("float32") / 255.0
    return x[None, ...]

def predict(img: Image.Image):
    x = preprocess(img)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    pred_label = CIFAR10_CLASSES[idx]
    confidence = float(probs[idx])

    prob_dict = {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(10)}

    return pred_label, confidence, prob_dict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Label(num_top_classes=10, label="All class probabilities"),
    ],
    title="CIFAR-10 Classifier (challenger8_fixed.keras)",
    description="Upload an image. The app resizes it to 32×32, normalizes it, and predicts a CIFAR-10 class."
)

if __name__ == "__main__":
    demo.launch()