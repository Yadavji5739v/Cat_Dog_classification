import streamlit as st
import numpy as np
from PIL import Image
import os
import time

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PawPredict AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy-import TensorFlow so the page renders even before the model loads ────
from pathlib import Path

@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    base = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = [
        base / "models" / "cats_dogs_cnn_model.h5",
        base / "models" / "cats_dogs_cnn_model.keras",
        base / "cats_dogs_cnn_model.h5",
        base / "cats_dogs_cnn_model.keras",
        cwd / "models" / "cats_dogs_cnn_model.h5",
        cwd / "models" / "cats_dogs_cnn_model.keras",
        cwd / "cats_dogs_cnn_model.h5",
        cwd / "cats_dogs_cnn_model.keras",
        cwd.parent / "models" / "cats_dogs_cnn_model.h5",
        cwd.parent / "models" / "cats_dogs_cnn_model.keras",
    ]

    seen = set()
    tried = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(str(candidate))
        if candidate.exists():
            try:
                return tf.keras.models.load_model(str(candidate), compile=False), str(candidate)
            except Exception as e:
                st.warning(f"Failed to load model at {candidate}: {e}")
                continue

    return None, tried


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

/* ── Root & Background ── */
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1628 40%, #0a1520 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c2e 0%, #0a1420 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }

/* ── Hide default Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero title ── */
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #63b3ed, #b794f4, #fc8181);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.hero-sub {
    text-align: center;
    color: #90cdf4;
    font-size: 1.15rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.5px;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 1.5rem;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: rgba(99,179,237,0.05) !important;
    border: 2px dashed rgba(99,179,237,0.4) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(183,148,244,0.7) !important;
    background: rgba(183,148,244,0.07) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #4299e1, #9f7aea) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(66,153,225,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(66,153,225,0.55) !important;
}

/* ── Result banner ── */
.result-cat {
    background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(183,148,244,0.15));
    border: 1.5px solid rgba(99,179,237,0.5);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
}
.result-dog {
    background: linear-gradient(135deg, rgba(252,129,74,0.2), rgba(246,173,85,0.15));
    border: 1.5px solid rgba(246,173,85,0.5);
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
}
.result-label {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.confidence-text {
    font-size: 1.1rem;
    color: #e2e8f0;
    font-weight: 400;
}

/* ── Progress bar override ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #4299e1, #9f7aea) !important;
    border-radius: 999px !important;
}

/* ── Stat boxes ── */
.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #b794f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.82rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.2rem;
}

/* ── Info text ── */
.info-text { color: #a0aec0; font-size: 0.95rem; line-height: 1.7; }

/* ── Tip badges ── */
.tip-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    color: #90cdf4;
    margin: 0.2rem;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Warning box ── */
.warn-box {
    background: rgba(246,173,85,0.1);
    border: 1px solid rgba(246,173,85,0.35);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #f6ad55;
    font-size: 0.92rem;
}

/* ── Fix Streamlit image container ── */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
CLASS_MAP  = {0: ("Cat", "🐱", "cat"), 1: ("Dog", "🐶", "dog")}
CONFIDENCE_THRESHOLD = 60.0   # % below which we show a "low confidence" note


# ── Helper: preprocess ────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Helper: predict ───────────────────────────────────────────────────────────
def predict(model, pil_img):
    tensor = preprocess(pil_img)
    raw    = model.predict(tensor, verbose=0)[0][0]   # sigmoid output
    dog_conf = float(raw) * 100
    cat_conf = 100 - dog_conf
    label_idx = 1 if dog_conf >= 50 else 0
    confidence = dog_conf if label_idx == 1 else cat_conf
    return label_idx, confidence, cat_conf, dog_conf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:3rem;'>🐾</div>
        <div style='font-size:1.4rem; font-weight:700; color:#63b3ed;'>PawPredict AI</div>
        <div style='font-size:0.78rem; color:#4a5568; margin-top:0.3rem;'>CNN · Cat & Dog Classifier</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 How to use")
    st.markdown("""
    <div class='info-text'>
    1. Upload a <b>JPG / PNG / JPEG</b> photo<br>
    2. Click <b>Analyse Image</b><br>
    3. View the prediction & confidence<br>
    4. Try another image anytime!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("### 🧠 Model Info")
    st.markdown("""
    <div class='info-text'>
    <b>Architecture:</b> Custom CNN<br>
    <b>Input Size:</b> 128 × 128 px<br>
    <b>Classes:</b> Cat · Dog<br>
    <b>Output:</b> Sigmoid (binary)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("### 💡 Tips")
    tips = [
        "Clear, well-lit photos",
        "Single pet in frame",
        "Avoid blurry images",
        "Frontal or side view",
    ]
    for t in tips:
        st.markdown(f"<span class='tip-badge'>✓ {t}</span>", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#2d3748; font-size:0.75rem; padding-top:0.5rem;'>
        Built with TensorFlow + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>🐾 PawPredict AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Deep Learning · Binary Image Classification · Cat vs Dog</div>",
            unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading neural network…"):
    model, model_path = load_model()

if model is None:
    st.markdown("""
    <div class='warn-box'>
    ⚠️ <b>Model not found.</b><br>
    Place your trained model file (<code>cats_dogs_cnn_model.keras</code> or
    <code>cats_dogs_cnn_model.h5</code>) inside a <code>models/</code> folder
    next to this <code>app.py</code>, then restart.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-text' style='margin-top:1rem;'>
    <b>Deployment check:</b> the app is searching for the model in the current
    working directory and around the location of <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)

    paths_html = "<br>".join(f"<code>{p}</code>" for p in model_path)
    st.markdown(f"""
    <div class='info-text' style='font-size:0.85rem; margin-top:0.5rem;'>
    Expected model locations:<br>
    {paths_html}
    </div>
    """, unsafe_allow_html=True)

    # Show demo mode notice
    st.info("📌 Running in **Demo mode** – upload an image and the interface will appear, "
            "but predictions require the trained model file.")
    model_loaded = False
else:
    model_loaded = True
    fname = os.path.basename(model_path)
    st.success(f"✅ Model loaded — `{fname}`", icon="🧠")

st.markdown("")

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, "CNN",     "Architecture"),
    (c2, "128×128", "Input Size"),
    (c3, "2",       "Classes"),
    (c4, "Binary",  "Output Type"),
]:
    with col:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{val}</div>
            <div class='stat-label'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ── Upload + Result columns ───────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True, caption=f"📁 {uploaded.name}")

        st.markdown("")
        analyse_btn = st.button("🔍 Analyse Image", use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color:#4a5568;'>
            <div style='font-size:3rem;'>🖼️</div>
            <div style='margin-top:0.5rem; font-size:0.95rem;'>No image uploaded yet</div>
            <div style='font-size:0.8rem; margin-top:0.3rem;'>Supports JPG · PNG · JPEG · WEBP</div>
        </div>
        """, unsafe_allow_html=True)
        analyse_btn = False

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Prediction Result")

    if uploaded and analyse_btn:
        if not model_loaded:
            st.warning("⚠️ Model not loaded. Please add your trained model file and restart.")
        else:
            with st.spinner("🧠 Analysing…"):
                time.sleep(0.6)   # tiny pause for UX feel
                label_idx, confidence, cat_pct, dog_pct = predict(model, image)

            label, emoji, css_cls = CLASS_MAP[label_idx]
            result_class = "result-cat" if label_idx == 0 else "result-dog"
            color        = "#63b3ed" if label_idx == 0 else "#f6ad55"

            st.markdown(f"""
            <div class='{result_class}'>
                <div class='result-label'>{emoji} {label}</div>
                <div class='confidence-text'>
                    Confidence: <b style='color:{color};'>{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if confidence < CONFIDENCE_THRESHOLD:
                st.markdown("""
                <div style='margin-top:0.8rem;'>
                <div class='warn-box'>
                    ⚡ Low confidence — try a clearer, better-lit photo.
                </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Probability breakdown**")

            # Cat bar
            st.markdown(f"🐱 Cat — **{cat_pct:.1f}%**")
            st.progress(int(cat_pct))

            # Dog bar
            st.markdown(f"🐶 Dog — **{dog_pct:.1f}%**")
            st.progress(int(dog_pct))

            # Raw score
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-text' style='font-size:0.82rem;'>
                🔢 Raw sigmoid score: <code>{dog_pct/100:.6f}</code><br>
                📐 Input shape: <code>128 × 128 × 3</code><br>
                🏷️ Predicted class index: <code>{label_idx}</code>
            </div>
            """, unsafe_allow_html=True)

    elif uploaded and not analyse_btn:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color:#4a5568;'>
            <div style='font-size:3rem;'>👆</div>
            <div style='margin-top:0.5rem;'>Click <b style='color:#63b3ed;'>Analyse Image</b></div>
            <div style='font-size:0.82rem; margin-top:0.3rem;'>to run the classifier</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 1rem; color:#4a5568;'>
            <div style='font-size:3rem;'>🐾</div>
            <div style='margin-top:0.5rem;'>Results appear here</div>
            <div style='font-size:0.82rem; margin-top:0.3rem;'>Upload an image to begin</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── About section ─────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

with st.expander("📖 About this application"):
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        **🧠 Model**
        A Convolutional Neural Network trained on 25,000 labelled images
        (cats and dogs). Binary classification using a sigmoid output layer.
        """)
    with a2:
        st.markdown("""
        **⚙️ Preprocessing**
        Images are resized to 128×128 pixels and pixel values are
        normalised to the [0, 1] range (÷ 255) before inference.
        """)
    with a3:
        st.markdown("""
        **📊 Interpretation**
        The sigmoid output represents the probability of the image being
        a **Dog** (≥ 0.5 → Dog, < 0.5 → Cat). Confidence is the
        distance from the 50% decision boundary.
        """)
