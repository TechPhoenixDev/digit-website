import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from tensorflow import keras

st.set_page_config(page_title="Digit Recognition", page_icon="✍️")

@st.cache_resource
def load_model():
    return keras.models.load_model("digit_cnn .keras")

model = load_model()

def otsu_threshold(arr):
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total = arr.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = w_b = 0.0
    best, best_var = 0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best = var, t
    return best

def preprocess(img):
    img = ImageOps.exif_transpose(img).convert("L")
    img.thumbnail((600, 600))
    img = img.filter(ImageFilter.GaussianBlur(1))
    arr = np.array(img).astype(float)
    bg = np.array(img.filter(ImageFilter.GaussianBlur(25))).astype(float)
    diff = np.clip(bg - arr, 0, 255)
    if diff.max() < 1:
        return np.zeros((28, 28), dtype="float32")
    d = (diff / diff.max() * 255).astype(np.uint8)
    mask = d > otsu_threshold(d)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype="float32")
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
    x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
    crop = Image.fromarray((mask[y0:y1+1, x0:x1+1] * 255).astype(np.uint8))
    k = max(3, (max(crop.size) // 15) * 2 + 1)
    crop = crop.filter(ImageFilter.MaxFilter(k))
    w, h = crop.size
    s = 20.0 / max(w, h)
    crop = crop.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28-crop.size[0])//2, (28-crop.size[1])//2))
    a = np.array(canvas).astype("float32") / 255.0
    tot = a.sum()
    if tot > 0:
        cy = (a.sum(1) * np.arange(28)).sum() / tot
        cx = (a.sum(0) * np.arange(28)).sum() / tot
        a = np.roll(a, (int(round(14-cy)), int(round(14-cx))), axis=(0, 1))
    return a

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a photo of **ONE digit** written with a dark marker on white paper.")

file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file)
    x = preprocess(img)
    probs = model.predict(x.reshape(1, 28, 28, 1), verbose=0)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Your upload", width=250)
    with col2:
        st.image(x, caption="What the model sees (28x28)", clamp=True, width=150)

    st.success(f"Prediction: {probs.argmax()}   (confidence {probs.max()*100:.1f}%)")
    st.bar_chart(probs)
